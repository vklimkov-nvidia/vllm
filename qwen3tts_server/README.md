Qwen3-TTS CustomVoice server using vLLM + Triton.

Uses built-in speaker "Aiden" — no reference audio needed.

## 1. Convert checkpoint

```bash
python qwen3tts_scripts/convert_qwen3tts_checkpoint.py \
    ~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/<hash>/ \
    qwen3tts_server/models/qwen3_tts_vllm_model
```

## 2. Build and run the Triton container

```bash
cd qwen3tts_server
docker build -t qwen3tts_server_py .
docker run --rm -it --gpus all \
    --shm-size=8g \
    -p 8000:8000 \
    -v "$(pwd):/workspace/server" \
    qwen3tts_server_py \
    /bin/bash
```

## 3. Export model components (inside container)

```bash
# trace prefill encoder (embeds speaker + text into prefill)
python3 export_prefill_encoder.py \
    --model-path models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/<hash>/ \
    --speaker Aiden \
    --text "hello world!" \
    --language auto \
    --output models/prefill.pt \
    --device cuda --dtype bfloat16 \
    --torchscript-path models/encoder.jit

# trace codec (converts audio tokens → waveform)
python3 export_codec.py \
    --tokenizer-path models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/<hash>/ \
    --onnx-path models/codec.onnx \
    --trt-path model_repository/codec_decoder/1/codec.trt \
    --trt-fp16 \
    --trt-frames-profile 1 35 64 \
    --frames 35
```

## 4. Start the server

```bash
tritonserver --model-repository=model_repository
```

## 5. Send requests

See `run_request.ipynb`. Just send text + optional language, speaker is Aiden by default.

To change the default speaker, edit `default_speaker` in `model_repository/qwen3_tts/config.pbtxt`.
Available speakers: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee.
