from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from legopolitics.exceptions import DependencyUnavailableError, ModelInferenceError
from legopolitics.schemas import ModelProvenance
from legopolitics.utils.device import HardwareProfile, resolve_vlm_runtime_plan
from legopolitics.vision.base import ImageInput, VisionRequest, VisionResponseData
from legopolitics.vision.common import to_pil


class HuggingFaceVLMAdapter:
    """Generic adapter for Hugging Face ``image-text-to-text`` models.

    The adapter supports ordinary image-plus-prompt models and chat-template
    VLMs such as Qwen2.5-VL, SmolVLM/SmolVLM2, Idefics3, and
    LLaVA-OneVision. Models are loaded lazily and can use CPU execution,
    Accelerate device maps, CPU/disk offload, or bitsandbytes 4/8-bit loading.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        trust_remote_code: bool = False,
        *,
        revision: str | None = None,
        dtype: str = "auto",
        input_mode: Literal["auto", "plain", "chat"] = "auto",
        generation: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        pipeline_kwargs: dict[str, Any] | None = None,
        adapter_name: str | None = None,
        hardware_profile: HardwareProfile = "auto",
        quantization: Literal["none", "4bit", "8bit"] = "none",
        device_map: str | dict[str, int | str] | None = None,
        max_memory: dict[str, int | str] | None = None,
        offload_folder: str | None = None,
        use_flash_attention_2: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.dtype_name = dtype
        self.input_mode = input_mode
        self.default_generation = dict(generation or {})
        self.processor_kwargs = dict(processor_kwargs or {})
        self.model_kwargs = dict(model_kwargs or {})
        self.pipeline_kwargs = dict(pipeline_kwargs or {})
        self.adapter_name = adapter_name or "hf_vlm"
        self.hardware_profile = hardware_profile
        self.quantization = quantization
        self.device_map = device_map
        self.max_memory = dict(max_memory or {})
        self.offload_folder = offload_folder
        self.use_flash_attention_2 = use_flash_attention_2
        self.pipeline: Any = None
        self.runtime_plan = resolve_vlm_runtime_plan(
            requested_device=device,
            requested_dtype=dtype,
            requested_profile=hardware_profile,
            quantization=quantization,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
        )

    @staticmethod
    def _resolve_dtype(dtype: str) -> Any:
        if dtype == "auto":
            return "auto"
        try:
            import torch
        except ImportError:
            return dtype
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(dtype.lower(), dtype)

    def _quantization_config(self, dtype: Any) -> Any:
        if self.quantization == "none":
            return None
        if self.runtime_plan.device == "cpu":
            raise DependencyUnavailableError(
                "bitsandbytes 4/8-bit VLM loading requires a supported accelerator. "
                "Use quantization: none for CPU execution."
            )
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise DependencyUnavailableError(
                "Quantized Hugging Face VLM loading requires "
                "`pip install legopolitics[quantization]`."
            ) from exc
        if self.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        return BitsAndBytesConfig(load_in_8bit=True)

    def load(self) -> None:
        try:
            from transformers import AutoProcessor, pipeline
        except ImportError as exc:
            raise DependencyUnavailableError(
                "VLM support requires `pip install legopolitics[huggingface]`."
            ) from exc

        dtype = self._resolve_dtype(self.runtime_plan.dtype)
        model_kwargs = dict(self.model_kwargs)
        model_kwargs.setdefault("low_cpu_mem_usage", True)
        quantization_config = self._quantization_config(dtype)
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if self.runtime_plan.max_memory:
            model_kwargs["max_memory"] = self.runtime_plan.max_memory
        if self.runtime_plan.offload_folder:
            folder = Path(self.runtime_plan.offload_folder)
            folder.mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = str(folder)
            model_kwargs.setdefault("offload_state_dict", True)
        if self.use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        kwargs: dict[str, Any] = {
            "task": "image-text-to-text",
            "model": self.model_id,
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": dtype,
        }
        if self.runtime_plan.device_map is not None:
            kwargs["device_map"] = self.runtime_plan.device_map
        else:
            kwargs["device"] = -1 if self.runtime_plan.device == "cpu" else self.runtime_plan.device
        if self.revision:
            kwargs["revision"] = self.revision
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs
        if self.processor_kwargs:
            processor_load_kwargs: dict[str, Any] = {
                "trust_remote_code": self.trust_remote_code,
                **self.processor_kwargs,
            }
            if self.revision:
                processor_load_kwargs["revision"] = self.revision
            kwargs["processor"] = AutoProcessor.from_pretrained(
                self.model_id, **processor_load_kwargs
            )
        kwargs.update(self.pipeline_kwargs)
        try:
            self.pipeline = pipeline(**kwargs)
        except Exception as exc:
            raise ModelInferenceError(
                f"Could not load Hugging Face VLM {self.model_id} with "
                f"profile={self.runtime_plan.profile}, dtype={self.runtime_plan.dtype}, "
                f"quantization={self.quantization}: {exc}"
            ) from exc

    @staticmethod
    def _generated_text(result: Any) -> str:
        item = result[0] if isinstance(result, list) and result else result
        if isinstance(item, dict):
            generated = item.get("generated_text", item.get("text", item))
            if isinstance(generated, str):
                return generated
            if isinstance(generated, list):
                for message in reversed(generated):
                    if not isinstance(message, dict):
                        continue
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        text_parts = [
                            str(part.get("text", ""))
                            for part in content
                            if isinstance(part, dict) and part.get("type") == "text"
                        ]
                        if text_parts:
                            return "".join(text_parts)
            return str(generated)
        return str(item)

    def _plain_call(self, image: ImageInput, prompt: str, generation: dict[str, Any]) -> Any:
        assert self.pipeline is not None
        return self.pipeline(images=to_pil(image), text=prompt, **generation)

    def _chat_call(self, image: ImageInput, prompt: str, generation: dict[str, Any]) -> Any:
        assert self.pipeline is not None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": to_pil(image)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        generation.setdefault("return_full_text", False)
        return self.pipeline(text=messages, **generation)

    def analyze_image(self, image: ImageInput, request: VisionRequest) -> VisionResponseData:
        if self.pipeline is None:
            self.load()
        generation = {**self.default_generation, **request.generation}
        generation.setdefault("max_new_tokens", 256)

        errors: list[str] = []
        modes = [self.input_mode] if self.input_mode != "auto" else ["chat", "plain"]
        for mode in modes:
            try:
                result = (
                    self._chat_call(image, request.prompt, dict(generation))
                    if mode == "chat"
                    else self._plain_call(image, request.prompt, dict(generation))
                )
                return VisionResponseData(
                    raw_text=self._generated_text(result),
                    metadata={
                        "input_mode": mode,
                        "model_id": self.model_id,
                        "hardware_profile": self.runtime_plan.profile,
                    },
                )
            except Exception as exc:
                errors.append(f"{mode}: {exc}")

        raise ModelInferenceError(
            f"Hugging Face VLM inference failed for {self.model_id}. " + " | ".join(errors)
        )

    def analyze_images(
        self, images: Sequence[ImageInput], request: VisionRequest
    ) -> list[VisionResponseData]:
        return [self.analyze_image(x, request) for x in images]

    def metadata(self) -> ModelProvenance:
        return ModelProvenance(
            adapter=self.adapter_name,
            model_id=self.model_id,
            revision=self.revision,
            device=self.runtime_plan.device,
            dtype=self.runtime_plan.dtype,
            metadata={
                "input_mode": self.input_mode,
                "trust_remote_code": self.trust_remote_code,
                "source": "huggingface",
                "hardware_profile": self.runtime_plan.profile,
                "quantization": self.quantization,
                "device_map": self.runtime_plan.device_map,
                "max_memory": self.runtime_plan.max_memory,
                "offload_folder": self.runtime_plan.offload_folder,
                "flash_attention_2": self.use_flash_attention_2,
            },
        )

    def unload(self) -> None:
        self.pipeline = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
