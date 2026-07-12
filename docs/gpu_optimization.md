# GPU and CPU optimization

legopolitics is designed to run on CPU-only systems and on GPUs ranging from
small consumer cards to high-memory workstations.

## Automatic recommendation

```bash
legopolitics recommend-vlm-profile
```

The command reports the detected GPU, total memory, BF16 support, and a
conservative configuration-template recommendation.

## Hardware profiles

```yaml
performance:
  hardware_profile: auto
```

Valid values are:

- `auto`
- `cpu`
- `small_gpu`
- `midrange_gpu`
- `high_end_gpu`
- `maximum`

The profile controls default dtype and offload behavior. Per-model YAML settings
always take precedence.

## Memory-control settings

```yaml
vision:
  models:
    huggingface_vlms:
      - name: qwen
        enabled: true
        model_id: Qwen/Qwen2.5-VL-3B-Instruct
        dtype: float16
        quantization: 4bit
        device_map: auto
        max_memory:
          "0": 7GiB
          cpu: 32GiB
        offload_folder: cache/offload

performance:
  inference_batch_size: 1
  low_vram_mode: true
  unload_models_between_stages: true
```

Important controls:

- reduce `processor_kwargs.max_pixels` for visual models;
- reduce YOLO `image_size`;
- use one-frame batches;
- shorten `max_new_tokens`;
- load VLMs sequentially;
- use 4-bit or 8-bit loading where supported;
- use `device_map: auto` for CPU offload;
- set explicit `max_memory` values;
- save disk offload on a fast local SSD rather than a network drive.

## CPU mode

CPU mode is slow but useful for validation and small datasets:

```bash
legopolitics config-template --profile hf-cpu --output analysis.yaml
```

Use the smallest integrated model, FP32, one-frame batches, and a low sampling
rate.

## Multi-GPU machines

A custom device map or `max_memory` dictionary may reference more than one GPU:

```yaml
device_map: auto
max_memory:
  "0": 22GiB
  "1": 22GiB
  cpu: 64GiB
```

The package records the effective loading plan in model provenance.

## Avoiding out-of-memory failures

Start with the smallest profile and a short video. Increase image resolution,
model size, and generation length separately. If an OOM occurs, legopolitics
records the failure; configured pipeline recovery can reduce batch size or skip
the failed model without discarding successful stages.
