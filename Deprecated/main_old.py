from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2 as cv
import numpy as np
import io
from PIL import Image

app = FastAPI()

def convert_to_pencil_sketch(image):
    """Applies a pencil sketch effect using the Dodge Blend Technique."""
    
    # Convert image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Invert the grayscale image
    invert_image = cv.bitwise_not(gray_image)

    # Apply Gaussian blur to the inverted image
    blur_image = cv.GaussianBlur(invert_image, (21, 21), 0)

    # Invert the blurred image
    invert_blur = cv.bitwise_not(blur_image)

    # Blend the grayscale image with the inverted blurred image
    sketch = cv.divide(gray_image, invert_blur, scale=256.0)

    return sketch

@app.get("/")
def home():
    return {"message": "FastAPI backend is running"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Handles image uploads and returns a processed pencil sketch image."""
    contents = await file.read()
    image = np.array(Image.open(io.BytesIO(contents)))  # Convert to NumPy array
    
    # Convert RGBA (if present) to BGR format for OpenCV compatibility
    if image.shape[-1] == 4:
        image = cv.cvtColor(image, cv.COLOR_RGBA2BGR)
    else:
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # Apply pencil sketch filter
    sketch = convert_to_pencil_sketch(image)
    
    # Encode the processed image as PNG
    _, buffer = cv.imencode(".png", sketch)
    
    return JSONResponse(content={"message": "Image processed", "image_data": buffer.tobytes().hex()})