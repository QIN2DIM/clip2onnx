# -*- coding: utf-8 -*-
# Time       : 2023/10/25 6:52
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
"""
Open-CLIP Benchmarks
> https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv

Open-CLIP Model Profile
> https://github.com/mlfoundations/open_clip/blob/main/docs/model_profile.csv

Awesome list for production on CLIP
> https://github.com/QIN2DIM/awesome-clip-production
"""

HF_ViT_L_14 = {
    # 2023-10-25, 1.79GB, 0.7921 % ImageNet-1k
    # https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K
    "model_name": "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
}

EVA02_L_14_336 = {
    # 2023-10-25,  0.8039 % ImageNet-1k
    # https://github.com/baaivision/EVA/tree/master/EVA-CLIP
    "model_name": "EVA02-L-14-336",
    "tag": "merged2b_s6b_b61k",
}

HF_ViT_H_14_CLIPA_336 = {
    # 2023-10-25, 3.94GB, 0.818 % ImageNet-1k
    # https://huggingface.co/UCSC-VLAA/ViT-H-14-CLIPA-336-datacomp1B
    "model_name": "hf-hub/UCSC-VLAA/ViT-H-14-CLIPA-336-datacomp1B"
}
