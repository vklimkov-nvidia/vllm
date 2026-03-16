FROM nvcr.io/nvidia/tritonserver:26.02-py3

# 1. System dependencies for audio libraries
RUN apt-get update && \
    apt-get install -y sox libsox-fmt-all && \
    rm -rf /var/lib/apt/lists/*

# 2a. Install torch + torchaudio from the CUDA 12.8 index
RUN pip install --no-cache-dir \
    torch torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 2b. Python packages for checkpoint conversion (from PyPI)
RUN pip install --no-cache-dir \
    transformers==4.57.3 \
    onnxruntime-gpu \
    librosa \
    sox \
    onnxscript \
    accelerate

# 3. Build-time deps that the base image may not have
#    (needed because we use --no-build-isolation below)
RUN pip install --no-cache-dir \
    "setuptools>=77.0.3,<80.0.0" \
    "setuptools-scm>=8.0" \
    ninja \
    cmake \
    jinja2

# 4. Clone fork, build from source, clean up
#    --no-build-isolation: reuse the base image's torch/CUDA for compilation
#    --no-deps: don't install/upgrade any runtime deps (preserves TRT-LLM, TensorRT, torch, etc.)
#    WARNING: TORCH_CUDA_ARCH_LIST="8.6" is for RTX6000, modify if needed
RUN git clone --single-branch --branch vklimkov/qwen3_tts \
        https://github.com/vklimkov-nvidia/vllm.git /tmp/vllm-fork && \
    cd /tmp/vllm-fork && \
    MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="8.6" pip install --no-build-isolation --no-deps . && \
    cd / && rm -rf /tmp/vllm-fork

# 5. vllm runtime deps omitted by --no-deps
RUN pip install --no-cache-dir \
    setproctitle \
    cloudpickle \
    compressed-tensors==0.11.0 \
    depyf==0.19.0 \
    diskcache==5.6.3 \
    gguf==0.13.0 \
    gpt-oss==0.0.7 \
    lm-format-enforcer==0.11.3 \
    msgspec \
    outlines_core==0.2.11 \
    py-cpuinfo \
    pybase64 \
    cbor2 \
    cachetools \
    prometheus_client \
    einops \
    lark==1.2.2 \
    blake3 \
    Pillow \
    pyzmq \
    tensorrt==10.15.1.29

# 6. Verify the shm_channel extension compiled and loads
RUN python3 -c 'import vllm._shm_channel_cpp as m; print("OK:", m.__file__)'

# 7. Clone the official Qwen3-TTS repository
RUN git clone https://github.com/QwenLM/Qwen3-TTS.git /workspace/Qwen3-TTS
RUN pip install -e /workspace/Qwen3-TTS/ --no-deps

WORKDIR /workspace