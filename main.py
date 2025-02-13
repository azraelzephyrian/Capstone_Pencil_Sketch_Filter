import cv2
import numpy as np
import os
import shutil
import re
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from mimetypes import guess_type

# Create directories for uploaded files and processed results if they don't exist
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure upload directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)  # Ensure results directory exists

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = FastAPI()

# Mount the "frontend" directory to serve static HTML, CSS, JS files
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Helper functions
def is_allowed_file(filename: str) -> bool:
    """Check if the file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_filename(filename: str) -> str:
    """Sanitize the filename to prevent path traversal."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)

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

# Route to handle file upload
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Handles image uploads and saves the file to 'uploads/'."""
    if not is_allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed. Only .png, .jpg, and .jpeg are supported.")

    sanitized_filename = sanitize_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, sanitized_filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": "File uploaded successfully", "filename": sanitized_filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

# Route for converting the uploaded image to pencil sketch
@app.post("/convert/")
async def convert_image(filename: str):
    """Converts the uploaded image to pencil sketch."""
    input_path = os.path.join(UPLOAD_DIR, filename)
    output_path = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        img = cv2.imread(input_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid image.")

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Invert the grayscale image
        inverted_image = 255 - gray_image

        # Blur the inverted image
        blurred_image = cv2.GaussianBlur(inverted_image, (111, 111), 0)

        # Invert the blurred image
        inverted_blurred_image = 255 - blurred_image

        # Create the pencil sketch by blending the grayscale and inverted blurred images
        pencil_sketch = cv2.divide(gray_image, inverted_blurred_image, scale=256.0)

        # Save the resulting pencil sketch
        cv2.imwrite(output_path, pencil_sketch)

        # Return a success message with the filename of the processed image
        return JSONResponse(content={"message": "Image converted successfully", "processed_filename": filename})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to convert image: {str(e)}")

# Route to rotate the image 90 degrees to the left (counterclockwise)
@app.post("/rotate-left/")
async def rotate_left(filename: str):
    """Rotates the image 90 degrees left (counterclockwise)."""
    input_path = os.path.join(UPLOAD_DIR, filename)
    output_path = os.path.join(RESULTS_DIR, "rotated_left_" + filename)

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        img = cv2.imread(input_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid image.")

        # Rotate the image 90 degrees counterclockwise
        rotated_image = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Save the rotated image
        cv2.imwrite(output_path, rotated_image)

        # Return a success message with the filename of the processed image
        return JSONResponse(content={"message": "Image rotated left successfully", "processed_filename": "rotated_left_" + filename})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rotate image left: {str(e)}")

# Route to rotate the image 90 degrees to the right (clockwise)
@app.post("/rotate-right/")
async def rotate_right(filename: str):
    """Rotates the image 90 degrees right (clockwise)."""
    input_path = os.path.join(UPLOAD_DIR, filename)
    output_path = os.path.join(RESULTS_DIR, "rotated_right_" + filename)

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        img = cv2.imread(input_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid image.")

        # Rotate the image 90 degrees clockwise
        rotated_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        # Save the rotated image
        cv2.imwrite(output_path, rotated_image)

        # Return a success message with the filename of the processed image
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