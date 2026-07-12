# Limitations

Brick-built anatomy, stylized accessories, occlusion, repeated character designs, rapid stop-motion edits, subtitles, and low-resolution uploads can reduce detector, pose, OCR, and tracking accuracy. A rectangular mask fallback is not equivalent to semantic segmentation. Simple IoU tracking can fragment characters or switch identities; appearance embeddings and manual review are recommended for demanding work.

Optional integrations depend on external package versions, model APIs, hardware, access tokens, and model licenses. They are implemented behind lazy adapters but were not all executed in the package build environment. Run `legopolitics doctor` and adapter-specific integration tests in the target environment.

Context-conditioned VLM outputs can reflect confirmation bias. Compare them with blind outputs and human coding.
