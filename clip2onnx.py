import inspect
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import clip
import open_clip
import torch
from PIL import Image

from templates import EVA02_L_14_336

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class ModelCard:
    """
    Open-CLIP Benchmarks
    > https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv

    Open-CLIP Model Profile
    > https://github.com/mlfoundations/open_clip/blob/main/docs/model_profile.csv

    """

    model_name: str
    """
    model name
    """

    tag: str = ""
    """
    pretrained
    """

    onnx_visual: str = ""
    onnx_textual: str = ""
    """
    Merged ONNX Model Name.
    If you decide to export the model on the huggingface, 
    please customize the onnx_visual and onnx_textual fields.
    """

    onnx_visual_path: Path = field(default=Path)
    onnx_textual_path: Path = field(default=Path)
    model_dir: Path = Path("model")
    """
    Default export path:
      - visual: [project_dir]/model/*[self.model_name]/[VERSION]/[self.onnx_visual]
      - textual: [project_dir]/model/*[self.model_name]/[VERSION]/[self.onnx_textual]
    """

    DEFAULT_TEXTUAL_FIELDS = {
        "export_params": True,
        "input_names": ["TEXT"],
        "output_names": ["TEXT_EMBEDDING"],
        "dynamic_axes": {"TEXT": {0: "text_batch_size"}, "TEXT_EMBEDDING": {0: "text_batch_size"}},
    }
    """
    Template parameter of the textual-part appended to `torch.onnx.export`
    """

    DEFAULT_VISUAL_FIELDS = {
        "export_params": True,
        "input_names": ["IMAGE"],
        "output_names": ["IMAGE_EMBEDDING"],
        "dynamic_axes": {
            "IMAGE": {0: "image_batch_size"},
            "IMAGE_EMBEDDING": {0: "image_batch_size"},
        },
    }
    """
    Template parameter of the visual-part appended to `torch.onnx.export`
    """

    def __post_init__(self):
        _prefix = self.model_name

        if not self.tag and (not self.onnx_visual or not self.onnx_textual):
            logging.warning(
                "If you decide to export the model on the huggingface, "
                "please customize the onnx_visual and onnx_textual fields."
            )
            _prefix = self.model_name.split("/")[-1]

        if not self.tag:
            self.onnx_visual = f"visual_CLIP_{_prefix}.onnx"
            self.onnx_textual = f"textual_CLIP_{_prefix}.onnx"
        else:
            self.onnx_visual = f"visual_CLIP_{_prefix}.{self.tag}.onnx"
            self.onnx_textual = f"textual_CLIP_{_prefix}.{self.tag}.onnx"

        _suffix = 1
        for _ in range(100):
            pre_dir = self.model_dir.joinpath(self.model_name, f"v{_suffix}")
            if not pre_dir.exists():
                pre_dir.mkdir(parents=True, exist_ok=True)
                break
            _suffix += 1

        self.onnx_visual_path = self.model_dir.joinpath(
            self.model_name, f"v{_suffix}", self.onnx_visual
        )
        self.onnx_textual_path = self.model_dir.joinpath(
            self.model_name, f"v{_suffix}", self.onnx_textual
        )

    @classmethod
    def from_template(cls, template):
        return cls(
            **{
                key: (template[key] if val.default == val.empty else template.get(key, val.default))
                for key, val in inspect.signature(cls).parameters.items()
            }
        )

    def __call__(self, *args, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dummy_image_path = kwargs.get("dummy_image_path")

        if not self.tag:
            model, preprocess = open_clip.create_model_from_pretrained(
                self.model_name, device=device
            )
        else:
            model, preprocess = open_clip.create_model_from_pretrained(
                self.model_name, self.tag, device=device
            )

        self.to_onnx_visual(model, preprocess, device, dummy_image_path)
        self.to_onnx_textual(model, device)

        logging.info(f"Successfully exported model - path={self.model_dir}")

        return model, preprocess

    def to_onnx_visual(self, model, preprocess, device, dummy_image_path: Path = None):
        dummy_image_path = dummy_image_path or Path("franz-kafka.jpg")
        dummy_image = preprocess(Image.open(dummy_image_path)).unsqueeze(0).to(device)

        model.forward = model.encode_image
        torch.onnx.export(
            model=model,
            args=(dummy_image,),
            f=f"{self.onnx_visual_path}",
            **self.DEFAULT_VISUAL_FIELDS,
        )

    def to_onnx_textual(self, model, device):
        dummy_text = clip.tokenize(
            [
                "a photo taken during the day",
                "a photo taken at night",
                "a photo taken of Mickey Mouse",
            ]
        ).to(device)

        model.forward = model.encode_text
        torch.onnx.export(
            model=model,
            args=(dummy_text,),
            f=f"{self.onnx_textual_path}",
            **self.DEFAULT_TEXTUAL_FIELDS,
        )


def print_available_open_clip_models():
    def lookup(mol: Callable[[], list]):
        for model_name in mol():
            if tags := open_clip.list_pretrained_tags_by_model(model_name):
                for tag in tags:
                    card = {"model": model_name, "tag": tag}
                    print(f"ModelCard | {card}")
            card = {"model": model_name, "tag": ""}
            print(f"ModelCard | {card}")

    lookup(open_clip.list_openai_models)
    lookup(open_clip.list_models)


def main():
    assets_dir = Path(__file__).parent.joinpath("assets")
    dummy_image_path = assets_dir.joinpath("hello-world.jpg")

    model_card = ModelCard.from_template(EVA02_L_14_336)
    model_card(dummy_image_path=dummy_image_path)


if __name__ == "__main__":
    # print_available_open_clip_models()
    main()
