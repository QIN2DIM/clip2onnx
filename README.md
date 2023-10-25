# clip2onnx

## Installing

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/openai/CLIP.git@main
pip install open_clip_torch
```

## Quick Start

Export CLIP to ONNX. Supports models from `huggingface`, `clip`, `open_clip`.

```python
from pathlib import Path

from clip2onnx.model_card import ModelCard
from clip2onnx.templates import EVA02_L_14_336, HF_ViT_L_14

assets_dir = Path(__file__).parent.parent.joinpath("assets")
dummy_image_path = assets_dir.joinpath("hello-world.jpg")

model_dir = Path(__file__).parent.parent.joinpath("model")


def demo():
    for template in [HF_ViT_L_14, EVA02_L_14_336]:
        template.update({"model_dir": model_dir})
        model_card = ModelCard.from_template(template)
        model_card(dummy_image_path=dummy_image_path)


if __name__ == "__main__":
    demo()
```

## Example: Zero-shot Image Classification

→ [see more details](https://github.com/QIN2DIM/clip2onnx/blob/main/examples/auto_labeling.py)

## Example: ONNX Simplify

## Example: Quant INT8
