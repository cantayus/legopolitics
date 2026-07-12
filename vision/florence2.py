from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from legopolitics.exceptions import DependencyUnavailableError, ModelInferenceError
from legopolitics.schemas import ModelProvenance
from legopolitics.utils.device import detect_device, resolve_torch_dtype
from legopolitics.vision.base import ImageInput, VisionRequest, VisionResponseData
from legopolitics.vision.common import to_pil

TASK_PROMPTS = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
    "object_detection": "<OD>",
    "dense_region_caption": "<DENSE_REGION_CAPTION>",
    "region_proposal": "<REGION_PROPOSAL>",
    "ocr": "<OCR>",
    "ocr_with_region": "<OCR_WITH_REGION>",
    "phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "referring_expression_segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
    "region_to_description": "<REGION_TO_DESCRIPTION>",
    "region_to_category": "<REGION_TO_CATEGORY>",
}


class Florence2Adapter:
    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-large-ft",
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device_request = device
        self.dtype_name = dtype
        self.trust_remote_code = trust_remote_code
        self.model: Any = None
        self.processor: Any = None
        self.device = "cpu"

    def load(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise DependencyUnavailableError(
                "Florence-2 requires legopolitics[huggingface]"
            ) from exc
        info = detect_device(self.device_request)
        self.device = info.device
        dtype = resolve_torch_dtype(self.dtype_name, self.device)
        kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.revision:
            kwargs["revision"] = self.revision
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        self.processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs).to(self.device)
        self.model.eval()

    def analyze_image(self, image: ImageInput, request: VisionRequest) -> VisionResponseData:
        if self.model is None or self.processor is None:
            self.load()
        import torch

        pil = to_pil(image)
        prefix = TASK_PROMPTS.get(request.task, request.prompt or "<MORE_DETAILED_CAPTION>")
        prompt = prefix
        if request.prompt and request.task in {
            "phrase_grounding",
            "referring_expression_segmentation",
            "region_to_description",
            "region_to_category",
        }:
            prompt += request.prompt
        try:
            inputs = self.processor(text=prompt, images=pil, return_tensors="pt")
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            generation = {
                "max_new_tokens": 1024,
                "num_beams": 3,
                "do_sample": False,
                **request.generation,
            }
            with torch.inference_mode():
                generated = self.model.generate(**inputs, **generation)
            text = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
            parsed = None
            if hasattr(self.processor, "post_process_generation"):
                parsed = self.processor.post_process_generation(
                    text, task=prefix, image_size=pil.size
                )
            return VisionResponseData(
                raw_text=text, parsed=parsed if isinstance(parsed, dict) else {"result": parsed}
            )
        except Exception as exc:
            raise ModelInferenceError(f"Florence-2 inference failed: {exc}") from exc

    def analyze_images(
        self, images: Sequence[ImageInput], request: VisionRequest
    ) -> list[VisionResponseData]:
        return [self.analyze_image(image, request) for image in images]

    def metadata(self) -> ModelProvenance:
        return ModelProvenance(
            adapter="florence2",
            model_id=self.model_id,
            revision=self.revision,
            device=self.device,
            dtype=self.dtype_name,
        )

    def unload(self) -> None:
        self.model = None
        self.processor = None
