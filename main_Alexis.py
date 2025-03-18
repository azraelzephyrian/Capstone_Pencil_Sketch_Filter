import torch
import cv2
import numpy as np
import os
import shutil
import re
import time
from PIL import Image
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from mimetypes import guess_type

# Import your trained model architecture
import torch
import torch.nn as nn
import torch.optim as optim

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

        # Encoder (Downsampling layers)
        self.down1 = down_block(input_nc, ngf, apply_batchnorm=False)
        self.down2 = down_block(ngf, ngf * 2)
        self.down3 = down_block(ngf * 2, ngf * 4)
        self.down4 = down_block(ngf * 4, ngf * 8)
        self.down5 = down_block(ngf * 8, ngf * 8)
        self.down6 = down_block(ngf * 8, ngf * 8)
        self.down7 = down_block(ngf * 8, ngf * 8)
        self.down8 = down_block(ngf * 8, ngf * 8, apply_batchnorm=False)

        # Decoder (Upsampling layers)
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


# PatchGAN Discriminator
class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64):
        super(PatchGANDiscriminator, self).__init__()

        def disc_block(in_channels, out_channels, stride=2, apply_batchnorm=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False)]
            if apply_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            disc_block(input_nc + output_nc, ndf, apply_batchnorm=False),
            disc_block(ndf, ndf * 2),
            disc_block(ndf * 2, ndf * 4),
            disc_block(ndf * 4, ndf * 8, stride=1),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, img, sketch):
        input = torch.cat((img, sketch), 1)  # Concatenate input and output
        return self.model(input)


# Loss Functions
class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.criterion = nn.BCELoss()
    
    def forward(self, pred, target_is_real):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.criterion(pred, target)


import torch
import cv2
import numpy as np
import os
import shutil
import re
import time
from PIL import Image
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from mimetypes import guess_type

# Import trained model

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)  
os.makedirs(RESULTS_DIR, exist_ok=True)  

# Load trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "pix2pix_checkpoint_pencil.pth"
#CHECKPOINT_PATH = "pix2pix_checkpoint_edge_detect.pth"

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

def load_trained_model(checkpoint_path):
    """Loads the trained Pix2Pix generator model."""
    generator = UNetGenerator(input_nc=3, output_nc=1).to(DEVICE)  
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    generator.load_state_dict(checkpoint["generator_state"])
    generator.eval()
    return generator

model = load_trained_model(CHECKPOINT_PATH)

def preprocess_image(image_path):
    """Loads and preprocesses an image for the model."""
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Fix: Normalize all channels
    ])
    return transform(img).unsqueeze(0).to(DEVICE)

def postprocess_image(tensor):
    """Converts a model output tensor (grayscale) back to a displayable image."""
    tensor = tensor.squeeze().detach().cpu().numpy()  # Fix: Only one squeeze()
    tensor = (tensor * 0.5) + 0.5  # Undo normalization
    return np.clip(tensor * 255, 0, 255).astype(np.uint8)

app = FastAPI()
import os
from fastapi.staticfiles import StaticFiles

frontend_path = "frontend"
if not os.path.exists(frontend_path):
    os.makedirs(frontend_path)  # Create directory if missing

app.mount("/frontend", StaticFiles(directory=frontend_path), name="frontend")


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Handles image uploads and saves the file to 'uploads/'."""
    if not file.filename.lower().endswith(("png", "jpg", "jpeg")):
        raise HTTPException(status_code=400, detail="File type not allowed. Only .png, .jpg, and .jpeg are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return JSONResponse(content={"message": "File uploaded successfully", "filename": file.filename})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


@app.post("/generate-sketch/")
async def generate_sketch_with_model(filename: str):
    """Uses the trained model to generate a sketch from the uploaded image."""
    input_path = os.path.join(UPLOAD_DIR, filename)
    output_path = os.path.join(RESULTS_DIR, "gan_sketch_" + filename)

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        img_tensor = preprocess_image(input_path)
        with torch.no_grad():
            generated_sketch = model(img_tensor)

        sketch_image = postprocess_image(generated_sketch)

        cv2.imwrite(output_path, sketch_image)

        return JSONResponse(content={"message": "Sketch generated successfully", "processed_filename": "gan_sketch_" + filename})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate sketch: {str(e)}")

@app.get("/result/{filename}")
async def get_result(filename: str):
    """Fetches the processed sketch image from the results directory."""
    result_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Processed image not found")
    
    media_type, _ = guess_type(result_path)
    return FileResponse(result_path, media_type=media_type)

