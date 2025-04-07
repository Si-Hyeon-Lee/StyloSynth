import numpy as np
import torch
import torch.nn as nn
import base64
from PIL import Image
import segmentation_models_pytorch as smp


class Segmentation_model(nn.Module):
    def __init__(self, num_classes,encoder=None,pre_weight=None):
        super().__init__()
        self.model = smp.Unet( classes = num_classes,
                              encoder_name=encoder,
                              encoder_weights=pre_weight,
                              in_channels=3)
    
    def forward(self, x):
        y = self.model(x)
        return y

def get_cloth_mask(image_np, seg_pipe):
    """
    image_np: 세그멘테이션할 이미지 (numpy 배열)
    seg_pipe: Hugging Face의 세그멘테이션 파이프라인
    return: 의상 부분을 1로 하는 바이너리 마스크 (numpy 배열)
    """
    image_pil = Image.fromarray(image_np.astype(np.uint8))

    seg_results = seg_pipe(image_pil)
    target_labels = {"dress", "shirt", "t-shirt", "pants", "skirt", "coat", "jacket", "sweater", "top"}
    mask = np.zeros((image_pil.height, image_pil.width), dtype=np.uint8)

    for r in seg_results:
        label = r["label"].lower()
        if any(t in label for t in target_labels):

            base64_mask = r["mask"].split(",")[-1]  # 'data:image/png;base64,' 부분 제거
            decoded_mask = base64.b64decode(base64_mask)
            mask_pil = Image.open(io.BytesIO(decoded_mask)).convert("L")
            mask_np = np.array(mask_pil)

            mask = np.maximum(mask, np.where(mask_np > 128, 1, 0))

    return mask

def segmentation_output(mask, num_classes=7):
    label_colours = [(0, 0, 0), (0, 0, 0), (0, 0, 0),(256, 256, 256), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
    # 0: 전신, 1: 머리카락, 2: 머리~목, 3: 상의, 4: 바지, 5: 배경, 6: 팔

    h, w = mask.shape
    img = Image.new('RGB', (w, h))
    pixels = img.load()
    for j_, j in enumerate(mask[:, :]):
        for k_, k in enumerate(j):
            if k < num_classes:
                pixels[k_, j_] = label_colours[k]
    output = np.array(img)

    return output


def segementation(image, model):
    test_conv = image.transpose(2, 0 ,1)
    test_conv1 = test_conv[np.newaxis, :, :, :]
    test_conv_tensor = torch.from_numpy(test_conv1.copy()).float().to(device =torch.device('cuda'))
    conv_out_model = model(test_conv_tensor)
    output_model = torch.argmax(conv_out_model, dim=1)
    vis_output_model = segmentation_output(output_model[0].data.cpu().numpy())
    return vis_output_model