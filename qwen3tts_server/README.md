this branch has qwen3tts impl. to run it, convert the checkpoint:
```
python qwen3tts_scripts/convert_qwen3tts_checkpoint.py ~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3/ qwen3tts_server/models/qwen3_tts_vllm_model
```

see `demo_qwen3_tts.ipynb` on how to run it or `qwen3tts_scripts/benchmark_qwen3_tts.py` to benchmark the implementation.


in order to serve the model, we need tritron inference server.
```
cd qwen3tts_server
docker build -t qwen3tts_server_py .
```

Run it
```
docker run --rm -it --gpus all \
    --shm-size=8g \
    -p 8000:8000 \
    -v "$(pwd):/workspace/server" \
    qwen3tts_server_py \
    /bin/bash
```

Inside the container, trace other model components:
```
# capture reference speaker information
python3 extract_reference.py --model-path models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3/ --ref-text "I felt like, you know, you can be both, right? I mean, you can be both. What he was saying to me is like, are you here to do the work? Um, and you know, I in I internalize that. If your end goal is just to be a movie star, well" --ref-audio referencespeaker.wav --language english --output models/reference.pt --dtype float32

# trace encoder that combines reference speaker info and text for synthesis.
# this is a lighweight module that is run per request.
python3 export_prefill_encoder.py --model-path models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3/ --ref-data models/reference.pt --text "hello world!" --language "english" --output models/prefill.pt --device cuda --dtype bfloat16 --torchscript-path models/encoder.jit

# trace codec that converts audio tokens to audio
python3 export_codec.py --tokenizer-path models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/2069d3478828c9135fff015cd13613975dfa4ba8/ --onnx-path models/codec.onnx --trt-path model_repository/codec_decoder/1/codec.trt
```

Finally you can start a server with 

```
tritonserver --model-repository=model_repository
```

and send requests as shown in `run_request.ipynb`
