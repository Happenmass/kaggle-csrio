from urllib.request import urlopen
from PIL import Image
import timm
import torch
import numpy as np

img = Image.open("csiro-biomass/test/ID1001187975.jpg")

model = timm.create_model(
    'vit_7b_patch16_dinov3.lvd1689m',
    pretrained=False,
    num_classes=0,  # remove classifier nn.Linear
    checkpoint_path="timm/vit_7b_patch16_dinov3.lvd1689m/model.safetensors"
)
model = model.eval().to(torch.bfloat16).to("cuda")

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

print(img.size)

#convert img to tensor
img_tensor = torch.tensor(np.array(img)).unsqueeze(0)

output = model(transforms(img).unsqueeze(0).to(torch.bfloat16).to("cuda"))  # output is (batch_size, num_features) shaped tensor

print(output.shape)
