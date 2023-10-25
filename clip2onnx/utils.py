# -*- coding: utf-8 -*-
# Time       : 2023/10/25 7:17
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from typing import Callable


def print_available_open_clip_models():
    import open_clip

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
