# GAN.py
import os
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

# Generator (U-Net with skip connections)
class UNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ngf=64):
        super(UNetGenerator, self).__init__()

        def down_block(in_channels, out_channels, apply_batchnorm=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
            if apply_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up_block(in_channels, out_channels, apply_dropout=False):
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU(inplace=True)]
            if apply_dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        self.down1 = down_block(input_nc, ngf, apply_batchnorm=False)
        self.down2 = down_block(ngf, ngf * 2)
        self.down3 = down_block(ngf * 2, ngf * 4)
        self.down4 = down_block(ngf * 4, ngf * 8)
        self.down5 = down_block(ngf * 8, ngf * 8)
        self.down6 = down_block(ngf * 8, ngf * 8)
        self.down7 = down_block(ngf * 8, ngf * 8)
        self.down8 = down_block(ngf * 8, ngf * 8, apply_batchnorm=False)

        self.up1 = up_block(ngf * 8, ngf * 8, apply_dropout=True)
        self.up2 = up_block(ngf * 8 * 2, ngf * 8, apply_dropout=True)
        self.up3 = up_block(ngf * 8 * 2, ngf * 8, apply_dropout=True)
        self.up4 = up_block(ngf * 8 * 2, ngf * 8)
        self.up5 = up_block(ngf * 8 * 2, ngf * 4)
        self.up6 = up_block(ngf * 4 * 2, ngf * 2)
        self.up7 = up_block(ngf * 2 * 2, ngf)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        return self.final(torch.cat([u7, d1], 1))


# Globals
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"

# Load models
model_pencil = None
model_edge = None
var = 10

def load_trained_model(path):
    model = UNetGenerator().to(DEVICE)
    try:
        checkpoint = torch.load(path, map_location=DEVICE)
        model.load_state_dict(checkpoint["generator_state"])
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Failed to load model from {path}: {e}")
        return None


def initialize_models(pencil_path, edge_path):
    global model_pencil, model_edge
    model_pencil = load_trained_model(pencil_path)
    model_edge = load_trained_model(edge_path)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(img).unsqueeze(0).to(DEVICE)

def postprocess_image(tensor):
    tensor = tensor.squeeze().detach().cpu().numpy()
    tensor = (tensor * 0.5) + 0.5
    return np.clip(tensor * 255, 0, 255).astype(np.uint8)

def generate_image(model, filename, suffix):
    input_path = os.path.join(UPLOAD_DIR, filename)
    output_filename = f"{suffix}_gan_{filename}"
    output_path = os.path.join(RESULTS_DIR, output_filename)

    if not os.path.exists(input_path):
        raise FileNotFoundError("File not found")

    img_tensor = preprocess_image(input_path)
    with torch.no_grad():
        output_tensor = model(img_tensor)
    output_image = postprocess_image(output_tensor)
    cv2.imwrite(output_path, output_image)
    return output_filename
