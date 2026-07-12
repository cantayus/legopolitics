# MIT Redistribution Guide

The original `legopolitics` source code and project-authored documentation are
licensed under the MIT License.

## What redistribution permits

You may use, copy, modify, merge, publish, distribute, sublicense, and sell
copies of the MIT-licensed package materials, including modified versions.

## What must accompany redistributed copies

Include:

1. The copyright notice.
2. The complete MIT License text from `LICENSE`.
3. The applicable notices from `NOTICE`.
4. Third-party and model notices relevant to the components you redistribute.

## Materials not automatically covered by MIT

The repository MIT License does not relicense:

- Third-party Python dependencies.
- FFmpeg or other external executables.
- Hugging Face model weights.
- Datasets or media analyzed by the package.
- The user-supplied YOLO checkpoint unless separately authorized.
- Ultralytics code or model components governed by separate terms.
- Third-party trademarks.

The PyPI wheel excludes the bundled YOLO checkpoint. Before attaching or
redistributing that checkpoint through GitHub Releases, Git LFS, or another
service, confirm that you have the necessary rights and satisfy the applicable
Ultralytics terms.

This document is a practical project notice, not legal advice.
