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
