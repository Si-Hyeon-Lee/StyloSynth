# StyloSynth

A fun project that uses a PyTorch-based segmentation model (SMP) to detect and segment clothing from user-uploaded images, 
then transforms the outfit using a powerful LLM plus Stable Diffusion.

StyloSynth 는 **Pythorch 의 3rd party libraries** 만을 활용한 
**의상 변경** 프로젝트 입니다.

## Preview
|Input|Output|
|------|---|

## Overview

1. - [DeepFashion](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) Dataset 으로 최적화된
Segmentation 모델로 의상의 Mask를 얻어냅니다.

2. - [Google Gemini](https://gemini.google.com/) 사용자의 간단한 의상 변경 요청을 gemini 에게 복잡하고 다채로운 의상 묘사로 바꾸도록 요청합니다.

3. - [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) Binary Mask 에 의상을 채우도록 최적화된,
Pretrained Stable Diffusion Model 을 huggingface 에서 불러와 의상을 변경합니다.


## Key Libraries

- [segmentation_models.pytorch (SMP)](https://github.com/qubvel/segmentation_models.pytorch)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [PyTorch](https://pytorch.org)
