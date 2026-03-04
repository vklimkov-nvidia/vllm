// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// C++ extension for high-performance shared-memory IPC.
// Provides atomic flag operations, futex-based waiting, and batched
// memcpy with GIL release for minimal-latency cross-process tensor transfer.
//
// All flag operations use proper acquire/release memory ordering via GCC
// __atomic builtins, and the GIL is released during wait and copy phases
// so other Python threads can run concurrently.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstring>
#include <cerrno>
#include <cstdint>
#include <vector>

#include <sys/syscall.h>
#include <linux/futex.h>
#include <sys/mman.h>
#include <unistd.h>
#include <time.h>

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

// Number of spin iterations before falling back to futex.
// Each iteration is ~5 ns (atomic load + PAUSE), so 200 iterations ≈ 1 µs.
// This covers the common case where the other side signals almost immediately,
// avoiding the ~2-5 µs futex wake latency.
static constexpr int SPIN_ITERS = 200;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

struct CopyDesc {
    void*       dst;
    const void* src;
    size_t      nbytes;
};

// Spin + futex wait on a uint32 flag.  Caller must NOT hold the GIL.
static bool
wait_flag_impl(uint32_t* flag, double timeout_s) {
    // Phase 1: fast spin with CPU yield hints
    for (int i = 0; i < SPIN_ITERS; i++) {
        if (__atomic_load_n(flag, __ATOMIC_ACQUIRE) != 0)
            return true;
#if defined(__x86_64__)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#endif
    }

    // Phase 2: kernel-assisted futex sleep
    if (timeout_s < 0) {
        // Infinite wait
        while (true) {
            if (__atomic_load_n(flag, __ATOMIC_ACQUIRE) != 0)
                return true;
            syscall(SYS_futex, flag, FUTEX_WAIT, 0, nullptr, nullptr, 0);
        }
    }

    struct timespec start_ts;
    clock_gettime(CLOCK_MONOTONIC, &start_ts);
    const double start = start_ts.tv_sec + start_ts.tv_nsec * 1e-9;

    while (true) {
        if (__atomic_load_n(flag, __ATOMIC_ACQUIRE) != 0)
            return true;

        struct timespec now_ts;
        clock_gettime(CLOCK_MONOTONIC, &now_ts);
        double remaining = timeout_s - ((now_ts.tv_sec + now_ts.tv_nsec * 1e-9) - start);
        if (remaining <= 0)
            return __atomic_load_n(flag, __ATOMIC_ACQUIRE) != 0;

        struct timespec ts;
        ts.tv_sec  = static_cast<time_t>(remaining);
        ts.tv_nsec = static_cast<long>((remaining - ts.tv_sec) * 1e9);

        long ret = syscall(SYS_futex, flag, FUTEX_WAIT, 0, &ts, nullptr, 0);
        if (__atomic_load_n(flag, __ATOMIC_ACQUIRE) != 0)
            return true;
        if (ret == -1 && errno == ETIMEDOUT)
            return false;
        // EAGAIN: flag changed between load and FUTEX_WAIT → retry
    }
}

static inline void
set_and_wake(uint32_t* flag) {
    __atomic_store_n(flag, 1, __ATOMIC_RELEASE);
    syscall(SYS_futex, flag, FUTEX_WAKE, 1, nullptr, nullptr, 0);
}

// Parse a Python list[tuple[int,int,int]] into a CopyDesc vector.
// Returns false and sets a Python exception on error.
static bool
parse_copies(PyObject* list, std::vector<CopyDesc>& out) {
    if (!PyList_Check(list)) {
        PyErr_SetString(PyExc_TypeError, "expected a list of (dst, src, nbytes) tuples");
        return false;
    }
    Py_ssize_t n = PyList_GET_SIZE(list);
    out.resize(n);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* tup = PyList_GET_ITEM(list, i);
        out[i].dst    = reinterpret_cast<void*>(
            PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(tup, 0)));
        out[i].src    = reinterpret_cast<const void*>(
            PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(tup, 1)));
        out[i].nbytes = static_cast<size_t>(
            PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(tup, 2)));
        if (PyErr_Occurred()) return false;
    }
    return true;
}

static inline void
exec_copies(const std::vector<CopyDesc>& ops) {
    for (const auto& op : ops)
        std::memcpy(op.dst, op.src, op.nbytes);
}

// ---------------------------------------------------------------------------
// Python-facing functions
// ---------------------------------------------------------------------------

// wait_flag(addr: int, timeout_s: float) -> bool
// Spin + futex wait.  GIL is released during the entire wait.
// timeout_s < 0 means wait forever.
static PyObject*
py_wait_flag(PyObject* /*self*/, PyObject* args) {
    unsigned long long addr;
    double timeout_s;
    if (!PyArg_ParseTuple(args, "Kd", &addr, &timeout_s))
        return nullptr;

    auto* flag = reinterpret_cast<uint32_t*>(addr);
    bool ready;

    Py_BEGIN_ALLOW_THREADS
    ready = wait_flag_impl(flag, timeout_s);
    Py_END_ALLOW_THREADS

    return PyBool_FromLong(ready);
}

// wait_and_clear_flag(addr: int, timeout_s: float) -> bool
// Atomic wait + clear in a single GIL-free section.
static PyObject*
py_wait_and_clear_flag(PyObject* /*self*/, PyObject* args) {
    unsigned long long addr;
    double timeout_s;
    if (!PyArg_ParseTuple(args, "Kd", &addr, &timeout_s))
        return nullptr;

    auto* flag = reinterpret_cast<uint32_t*>(addr);
    bool ready;

    Py_BEGIN_ALLOW_THREADS
    ready = wait_flag_impl(flag, timeout_s);
    if (ready)
        __atomic_store_n(flag, 0, __ATOMIC_RELEASE);
    Py_END_ALLOW_THREADS

    return PyBool_FromLong(ready);
}

// set_flag_and_wake(addr: int) -> None
// Atomic store(1) + FUTEX_WAKE.
static PyObject*
py_set_flag_and_wake(PyObject* /*self*/, PyObject* args) {
    unsigned long long addr;
    if (!PyArg_ParseTuple(args, "K", &addr))
        return nullptr;

    set_and_wake(reinterpret_cast<uint32_t*>(addr));
    Py_RETURN_NONE;
}

// clear_flag(addr: int) -> None
static PyObject*
py_clear_flag(PyObject* /*self*/, PyObject* args) {
    unsigned long long addr;
    if (!PyArg_ParseTuple(args, "K", &addr))
        return nullptr;

    __atomic_store_n(reinterpret_cast<uint32_t*>(addr), 0, __ATOMIC_RELEASE);
    Py_RETURN_NONE;
}

// check_flag(addr: int) -> bool
static PyObject*
py_check_flag(PyObject* /*self*/, PyObject* args) {
    unsigned long long addr;
    if (!PyArg_ParseTuple(args, "K", &addr))
        return nullptr;

    uint32_t val = __atomic_load_n(reinterpret_cast<uint32_t*>(addr),
                                   __ATOMIC_ACQUIRE);
    return PyBool_FromLong(val != 0);
}

// batch_copy(copies: list[tuple[int, int, int]]) -> None
// Each tuple is (dst_addr, src_addr, nbytes).
// GIL is released during the memory copies.
static PyObject*
py_batch_copy(PyObject* /*self*/, PyObject* args) {
    PyObject* copies_list;
    if (!PyArg_ParseTuple(args, "O", &copies_list))
        return nullptr;

    std::vector<CopyDesc> ops;
    if (!parse_copies(copies_list, ops))
        return nullptr;

    Py_BEGIN_ALLOW_THREADS
    exec_copies(ops);
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

// batch_copy_and_signal(copies: list[tuple[int,int,int]], flag_addr: int) -> None
// Batched memcpy + atomic flag set + futex wake, all GIL-free.
static PyObject*
py_batch_copy_and_signal(PyObject* /*self*/, PyObject* args) {
    PyObject* copies_list;
    unsigned long long flag_addr;
    if (!PyArg_ParseTuple(args, "OK", &copies_list, &flag_addr))
        return nullptr;

    std::vector<CopyDesc> ops;
    if (!parse_copies(copies_list, ops))
        return nullptr;

    auto* flag = reinterpret_cast<uint32_t*>(flag_addr);

    Py_BEGIN_ALLOW_THREADS
    exec_copies(ops);
    set_and_wake(flag);
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

// prefault_mlock(addr: int, size: int) -> bool
// Touch every page then mlock.  Returns True on success.
static PyObject*
py_prefault_mlock(PyObject* /*self*/, PyObject* args) {
    unsigned long long addr;
    unsigned long long size;
    if (!PyArg_ParseTuple(args, "KK", &addr, &size))
        return nullptr;

    auto* p = reinterpret_cast<volatile char*>(addr);
    for (size_t off = 0; off < size; off += 4096)
        (void)p[off];

    int ret = mlock(reinterpret_cast<void*>(addr), static_cast<size_t>(size));
    return PyBool_FromLong(ret == 0);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

static PyMethodDef shm_channel_methods[] = {
    {"wait_flag",              py_wait_flag,              METH_VARARGS,
     "wait_flag(addr, timeout_s) -> bool\n"
     "Spin + futex wait on a uint32 flag. GIL released. timeout<0 = infinite."},

    {"wait_and_clear_flag",    py_wait_and_clear_flag,    METH_VARARGS,
     "wait_and_clear_flag(addr, timeout_s) -> bool\n"
     "Wait then clear flag atomically. GIL released."},

    {"set_flag_and_wake",      py_set_flag_and_wake,      METH_VARARGS,
     "set_flag_and_wake(addr) -> None\n"
     "Atomic store(1) + FUTEX_WAKE."},

    {"clear_flag",             py_clear_flag,             METH_VARARGS,
     "clear_flag(addr) -> None\n"
     "Atomic store(0)."},

    {"check_flag",             py_check_flag,             METH_VARARGS,
     "check_flag(addr) -> bool\n"
     "Atomic load, returns True if non-zero."},

    {"batch_copy",             py_batch_copy,             METH_VARARGS,
     "batch_copy(copies) -> None\n"
     "Batched memcpy with GIL released. copies: list[(dst,src,n)]."},

    {"batch_copy_and_signal",  py_batch_copy_and_signal,  METH_VARARGS,
     "batch_copy_and_signal(copies, flag_addr) -> None\n"
     "Batched memcpy + atomic set + futex wake. GIL released."},

    {"prefault_mlock",         py_prefault_mlock,         METH_VARARGS,
     "prefault_mlock(addr, size) -> bool\n"
     "Prefault pages and mlock. Returns True on success."},

    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef shm_channel_module = {
    PyModuleDef_HEAD_INIT,
    "_shm_channel_cpp",
    "C++ helpers for low-latency shared-memory IPC:\n"
    "atomic flags, futex wait/wake, batched memcpy with GIL release.",
    -1,
    shm_channel_methods,
};

PyMODINIT_FUNC
PyInit__shm_channel_cpp(void) {
    return PyModule_Create(&shm_channel_module);
}
