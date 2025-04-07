# StyloSynth

StyloSynth 는 **Pythorch 의 3rd party libraries** 만을 활용한 
**의상 변경** 프로젝트 입니다.
## Preview
|Input|Output|
|------|---|
|![original](https://github.com/user-attachments/assets/8c3d8899-f58d-47e9-ad5e-c14691febc2f)|![output](https://github.com/user-attachments/assets/03584e99-f24e-417d-a12a-7df569d0f4ad)|


## Overview

1. - [DeepFashion](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) Dataset 으로 최적화된
Segmentation 모델로 의상의 Mask를 얻어냅니다.


2. - [Google Gemini](https://gemini.google.com/) 사용자의 간단한 의상 변경 요청을 gemini 에게 복잡하고 다채로운 의상 묘사로 바꾸도록 요청합니다.

3. - [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) Binary Mask 에 의상을 채우도록 최적화된,
Pretrained Stable Diffusion Model 을 huggingface 에서 불러와 의상을 변경합니다.


## Key Libraries

- [segmentation_models.pytorch (SMP)](https://github.com/qubvel/segmentation_models.pytorch)
  옷의 binary mask 를 얻기 위한 segmentation model 에 사용했습니다.
- [google generativeai](https://ai.google.dev/gemini-api/docs/models?hl=ko)
  gemini 에게 사용자 프롬프트를 다채롭게 하기 위해 사용했습니다.
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
  Masked Area 에 의상을 채우기 위한 pretrained Model 을 위해 사용했습니다.
