# Hugging Face visual-language model ensemble

legopolitics can run multiple Hugging Face visual-language models against whole
frames, detected figure crops, and contextual crops. Every raw response is kept
separately before agreement and voting calculations.

The generic adapter uses the Transformers `image-text-to-text` pipeline and
supports plain prompts and chat-template models. It also supports:

- CPU execution;
- FP32, FP16, and BF16;
- Accelerate `device_map` loading;
- CPU or disk offload;
- `max_memory` limits;
- bitsandbytes 4-bit and 8-bit loading;
- optional FlashAttention 2;
- sequential model loading and unloading.

## Integrated model families

| Adapter name | Hugging Face model ID | Typical use |
|---|---|---|
| `smolvlm_500m` | `HuggingFaceTB/SmolVLM-500M-Instruct` | CPU and very small GPUs |
| `smolvlm2_500m` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | low-memory image/video analysis |
| `smolvlm2_2_2b` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | efficient detailed description |
| `qwen2_5_vl_3b` | `Qwen/Qwen2.5-VL-3B-Instruct` | visual reasoning, OCR, localization |
| `idefics3_8b` | `HuggingFaceM4/Idefics3-8B-Llama3` | OCR, document understanding, reasoning |
| `llava_onevision_7b` | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | image, multi-image, and video reasoning |

The adapters are model-ID driven, so another compatible Hugging Face
`image-text-to-text` checkpoint can be added in YAML without editing package
source.

## Select a hardware profile

Run:

```bash
legopolitics recommend-vlm-profile
```

Then create the suggested configuration:

```bash
legopolitics config-template \
  --profile hf-midrange-gpu \
  --output analysis.yaml
```

Available templates:

| Template | Intended environment | Default model strategy |
|---|---|---|
| `hf-cpu` | CPU-only | SmolVLM 500M, FP32 |
| `hf-small-gpu` | under roughly 6 GB VRAM | SmolVLM2 500M, FP16 |
| `hf-midrange-gpu` | roughly 6–14 GB VRAM | SmolVLM2 2.2B plus 4-bit Qwen 3B |
| `hf-high-end-gpu` | roughly 14–28 GB VRAM | SmolVLM2 and Qwen in BF16; 8B optional |
| `hf-maximum` | roughly 28+ GB VRAM | sequential 3B, 7B, and 8B ensemble |

The thresholds are conservative package defaults. Exact memory requirements
vary with image resolution, frame count, generation length, Transformers
version, attention implementation, and model revision.

## Installation

Standard Hugging Face support:

```bash
python -m pip install -e ".[huggingface]"
```

For 4-bit or 8-bit loading:

```bash
python -m pip install -e ".[huggingface,quantization]"
```

FlashAttention is deliberately not installed automatically because supported
builds depend on the operating system, GPU, CUDA, and PyTorch versions.

## Example mixed-GPU configuration

```yaml
performance:
  device: auto
  hardware_profile: midrange_gpu
  unload_models_between_stages: true
  inference_batch_size: 1

vision:
  whole_frame: true
  crops: true
  models:
    huggingface_vlms:
      - name: smolvlm2
        enabled: true
        model_id: HuggingFaceTB/SmolVLM2-2.2B-Instruct
        input_mode: chat
        dtype: float16
        generation:
          max_new_tokens: 192

      - name: qwen
        enabled: true
        model_id: Qwen/Qwen2.5-VL-3B-Instruct
        input_mode: chat
        dtype: float16
        quantization: 4bit
        device_map: auto
        max_memory:
          "0": 7GiB
          cpu: 32GiB
        offload_folder: cache/qwen_offload
        processor_kwargs:
          min_pixels: 100352
          max_pixels: 401408
        generation:
          max_new_tokens: 192
```

## Why sequential loading is the default

With `unload_models_between_stages: true`, legopolitics fully processes one VLM,
unloads it, clears the safe CUDA cache, and then loads the next. This is slower
than keeping every model resident but permits an ensemble to run on a wider
range of GPUs.

## Output and agreement

Each response records:

- model ID and revision;
- adapter name;
- dtype and quantization;
- hardware profile and device map;
- prompt version;
- raw generated text;
- target frame or detection;
- pass type;
- parsed output when available.

The ensemble layer can then compute majority voting, weighted voting,
maximum-confidence selection, score variance, and disagreement-based human
review priorities.

Review each model card, license, access restriction, and intended use before
running it on research data.
