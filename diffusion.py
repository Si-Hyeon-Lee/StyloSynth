import os, sys
import argparse
import copy
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
from huggingface_hub import hf_hub_download

def preprop_for_diffusion(image, vis_output_model):
    image_t = image.transpose(2, 0, 1)
    array_transposed1 = np.transpose(image_t, (1, 2, 0))

    image1 = np.rot90(array_transposed1, k=3) # 이미지 회전: 시계 방향으로 90도 회전 (실제로는 반시계 방향으로 세 번 회전)

    array_transposed2 = vis_output_model

    mask_image1 = np.rot90(array_transposed2, k=3) # 마스크 회전: 이미지와 같은 방식.

    image1 = image1 * 256
    image1 = image1.astype(np.uint8)
    mask_image1 = mask_image1.astype(np.uint8)

    image_source_pil = Image.fromarray(image1)
    image_mask_pil = Image.fromarray(mask_image1)

    display(*[image_source_pil, image_mask_pil])
    return image_source_pil, image_mask_pil


def generate_image(image, mask, prompt, negative_prompt, pipe, seed, device):
    w, h = image.size
    in_image = image.resize((512, 512))
    in_mask = mask.resize((512, 512))

    generator = torch.Generator(device).manual_seed(seed)

    result = pipe(image=in_image, mask_image=in_mask, prompt=prompt, negative_prompt=negative_prompt, generator=generator) # 허깅페이스 Stable Diffusion 파이프라인을 사용하여 이미지 생성
    result = result.images[0]
    return result.resize((w, h))
