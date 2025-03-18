import os
import time
import io
import torch
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware  
from torchvision import transforms
import torch.nn as nn
from torchvision import models
from fastapi import FastAPI
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

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)  
os.makedirs(RESULTS_DIR, exist_ok=True)  

# Load trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "pix2pix_checkpoint_pencil.pth"
#CHECKPOINT_PATH = "pix2pix_checkpoint_edge_detect.pth"

app = FastAPI()

# Define allowed origins (for CORS)
origins = [
    "http://localhost:3000",  # Allow frontend on localhost:3000
    "https://example.com",    # Allow example.com
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow specific origins or set to "*" for all
    allow_credentials=True,
    allow_methods=["*"],    # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)

# Create directories for uploaded files and processed results if they don't exist
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure upload directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)  # Ensure results directory exists

# Mount the "uploads" directory to serve static files
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# Mount the "frontend" directory to serve static HTML, CSS, JS files
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Helper functions
def is_allowed_file(filename: str) -> bool:
    """Check if the file is allowed based on the file extension."""
    allowed_extensions = {"png", "jpg", "jpeg"}
    return filename.split(".")[-1].lower() in allowed_extensions

def sanitize_filename(filename: str) -> str:
    """Sanitize the filename to avoid security issues."""
    return filename.replace(" ", "_").lower()

def cleanup_old_files(directory: str, max_age_seconds: int = 86400):
    """Delete files older than `max_age_seconds` in the given directory."""
    current_time = time.time()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                os.remove(file_path)

# Clean up old files on startup
cleanup_old_files(UPLOAD_DIR)
cleanup_old_files(RESULTS_DIR)


@app.post("/upload_GAN/")
async def upload_file_GAN(file: UploadFile = File(...)):
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


@app.post("/generate-sketch_GAN/")
async def generate_sketch_with_model_GAN(filename: str):
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

@app.get("/result_GAN/{filename}")
async def get_result_GAN(filename: str):
    """Fetches the processed sketch image from the results directory."""
    result_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Processed image not found")
    
    media_type, _ = guess_type(result_path)
    return FileResponse(result_path, media_type=media_type)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Handles image uploads and saves the file to 'uploads/'."""
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="File type not allowed. Only .png, .jpg, and .jpeg are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Log file path to verify
    print(f"File uploaded and saved to: {file_path}")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": "File uploaded successfully", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

   
# Route for converting the uploaded image to pencil sketch
@app.post("/convert/")
async def convert_image(filename: str, intensity: int = 50, stroke_size: int = 3, colorize: bool = False):


    """Converts the uploaded image to pencil sketch."""
    input_path = os.path.join(UPLOAD_DIR, filename)
    output_path = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        img = cv2.imread(input_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid image.")

        # Convert the image to HSV to extract hue & saturation for colorization
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue_channel, saturation_channel, _ = cv2.split(hsv_image)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply intensity adjustment (higher intensity = darker sketch)
        adjusted_gray = np.clip(gray_image * (intensity / 50.0), 0, 255).astype(np.uint8)



        # Invert the grayscale image
        inverted_image = 255 - gray_image

        # Blur the inverted image
        blurred_image = cv2.GaussianBlur(inverted_image, (111, 111), 0)

        # Invert the blurred image
        inverted_blurred_image = 255 - blurred_image

        # Create the pencil sketch by blending the grayscale and inverted blurred images
                # Create the pencil sketch by blending the adjusted grayscale and inverted blurred images
        pencil_sketch = cv2.divide(adjusted_gray, inverted_blurred_image, scale=256.0)

        # Save the resulting pencil sketch
        # Apply stroke effect by blurring based on stroke size
        if stroke_size > 1:
            pencil_sketch = cv2.GaussianBlur(pencil_sketch, (stroke_size * 2 + 1, stroke_size * 2 + 1), 0)

        # Save the resulting pencil sketch
        if colorize:
            # Combine hue & saturation with sketch intensity
            colorized_hsv = cv2.merge((hue_channel, saturation_channel, pencil_sketch))
            colorized_image = cv2.cvtColor(colorized_hsv, cv2.COLOR_HSV2BGR)

            # Save the colorized sketch
            cv2.imwrite(output_path, colorized_image)
        else:
            # Save the regular pencil sketch
            cv2.imwrite(output_path, pencil_sketch)



        # Return a success message with the filename of the processed image
        return JSONResponse(content={"message": "Image converted successfully", "processed_filename": filename})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to convert image: {str(e)}")

@app.post("/update-image/")
async def update_image(filename: str, intensity: int, stroke: int):
    """Updates the processed image with new intensity and stroke values."""
    input_path = os.path.join(RESULTS_DIR, filename)
    output_path = os.path.join(RESULTS_DIR, f"updated_{intensity}_{stroke}_" + filename)

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Apply intensity adjustment (scale grayscale values)
        img = cv2.multiply(img, (intensity / 100.0))

        # Apply stroke size effect (Gaussian blur)
        if stroke > 1:
            img = cv2.GaussianBlur(img, (2 * stroke + 1, 2 * stroke + 1), 0)

        # Save the updated image
        cv2.imwrite(output_path, img)

        return {"message": "Image updated successfully", "processed_filename": f"updated_{intensity}_{stroke}_" + filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update image: {str(e)}")

# Route to rotate the image 90 degrees to the left (counterclockwise)

@app.post("/rotate-left/")
async def rotate_left(filename: str):
    """Rotates the most recent sketch 90 degrees left (counterclockwise)."""
    input_path = os.path.join(RESULTS_DIR, filename)  # Ensure rotation uses the processed sketch
    output_path = os.path.join(RESULTS_DIR, "rotated_left_" + filename)

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Processed sketch file not found")

    try:
        img = cv2.imread(input_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid image.")

        # Rotate the image 90 degrees counterclockwise
        rotated_image = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Save the rotated image
        cv2.imwrite(output_path, rotated_image)

        return JSONResponse(content={"message": "Image rotated left successfully", "processed_filename": "rotated_left_" + filename})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rotate image left: {str(e)}")

# Route to rotate the image 90 degrees to the right (clockwise)
@app.post("/rotate-right/")
async def rotate_right(filename: str):
    """Rotates the most recent sketch 90 degrees right (clockwise)."""
    input_path = os.path.join(RESULTS_DIR, filename)  # Ensure rotation uses the processed sketch
    output_path = os.path.join(RESULTS_DIR, "rotated_right_" + filename)

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Processed sketch file not found")

    try:
        img = cv2.imread(input_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid image.")

        # Rotate the image 90 degrees clockwise
        rotated_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        # Save the rotated image
        cv2.imwrite(output_path, rotated_image)

        return JSONResponse(content={"message": "Image rotated right successfully", "processed_filename": "rotated_right_" + filename})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rotate image right: {str(e)}")


# Route for fetching the converted result
@app.get("/result/{filename}")
async def get_result(filename: str):
    """Fetches the processed sketch image from the results directory."""
    result_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Processed image not found")
    
    media_type, _ = guess_type(result_path)
    return FileResponse(result_path, media_type=media_type)

# Route to serve the homepage
@app.get("/")
def home():
    return {"message": "FastAPI backend is running"}

# Include routes from the model app
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
# You can add other endpoints in main.py as needed
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)