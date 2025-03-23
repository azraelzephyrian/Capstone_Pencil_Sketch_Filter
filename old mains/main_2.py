from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2 as cv
import numpy as np
import io
from PIL import Image
import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure upload directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)  # Ensure results directory exists

print("🚀 FastAPI is starting up!")
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


@app.post("/convert/")
async def convert_image(filename: str):
    """Loads an uploaded image, applies the pencil sketch filter, and saves the result."""
    
    input_path = os.path.join(UPLOAD_DIR, filename)
    output_path = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Read the image using OpenCV
    image = cv.imread(input_path)
    if image is None:
        raise HTTPException(status_code=400, detail="Error reading image file")

    # Apply pencil sketch effect
    sketch = convert_to_pencil_sketch(image)

    # Save the processed image
    cv.imwrite(output_path, sketch)

    print(f"✅ Processed image saved: {output_path}")  # Debugging

    return JSONResponse(content={"message": "Image converted successfully", "processed_filename": filename})

@app.get("/result/{filename}")
async def get_result(filename: str):
    """Fetches the processed sketch image from the results directory."""
    
    result_path = os.path.join(RESULTS_DIR, filename)
    
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Processed image not found")

    return FileResponse(result_path, media_type="image/png")

@app.get("/test/")
async def test():
    print("✅ FastAPI received a test request!")  # Debug output
    return {"message": "FastAPI is working"}

@app.get("/")
def home():
    return {"message": "FastAPI backend is running"}
'''
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Handles image uploads and saves the file to 'uploads/' before processing."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # Open the file in write mode and write the contents
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"✅ File saved: {file_path}")  # Debug: Check if file saves successfully

        # Read the saved file back for processing
        image = cv.imread(file_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Error reading image after saving.")

        # Convert image to pencil sketch
        sketch = convert_to_pencil_sketch(image)

        # Save the processed image
        processed_path = os.path.join("results", file.filename)
        os.makedirs("results", exist_ok=True)  # Ensure results folder exists
        cv.imwrite(processed_path, sketch)
        print(f"✅ Processed image saved: {processed_path}")

        return JSONResponse(content={"message": "Image uploaded and processed successfully", "filename": file.filename})

    except Exception as e:
        print(f"❌ Error: {e}")  # Debugging
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

#-o NUL
'''
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure the folder exists
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Handles image uploads and saves the file to 'uploads/' before processing."""
    print(f"📥 Received request to upload: {file.filename}")  # Debug

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        print(f"📂 Saving file to: {file_path}")  # Debug

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"✅ File saved successfully: {file_path}")  # Debug
        
        return {"message": "File uploaded successfully", "filename": file.filename}

    except Exception as e:
        print(f"❌ Error saving file: {e}")  # Debug
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

