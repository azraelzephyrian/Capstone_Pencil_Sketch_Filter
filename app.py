# flask_image_editor.py
import os
import uuid
from flask import Flask, request, session, send_file, jsonify, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import torch

print("✅ Flask app running from correct file!")

app = Flask(__name__)
from GAN import (
    model_pencil,
    model_edge,
    preprocess_image,
    postprocess_image,
    initialize_models,
    load_trained_model,
    preprocess_image,
    postprocess_image
)


# Set paths to your trained model checkpoint files
PENCIL_MODEL_PATH = "pix2pix_checkpoint_pencil.pth"
EDGE_MODEL_PATH = "pix2pix_checkpoint_edge_detect.pth"

# Initialize the models
initialize_models(PENCIL_MODEL_PATH, EDGE_MODEL_PATH)



PENCIL_MODEL_PATH = "pix2pix_checkpoint_pencil.pth"
EDGE_MODEL_PATH = "pix2pix_checkpoint_edge_detect.pth"

model_pencil = load_trained_model(PENCIL_MODEL_PATH)
model_edge = load_trained_model(EDGE_MODEL_PATH)

print("model_pencil:", model_pencil)
print("preprocess_image:", preprocess_image)

app.secret_key = "your-secret-key"  # Replace in production

UPLOAD_DIR = "uploads"
WORKSPACE_DIR = "workspace"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(WORKSPACE_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_user_id():
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    return session["user_id"]

def get_active_image_path():
    user_id = get_user_id()
    user_folder = os.path.join(WORKSPACE_DIR, user_id)
    os.makedirs(user_folder, exist_ok=True)
    return os.path.join(user_folder, "active.png")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/test", methods=["POST"])
def test_route():
    print("✅ /test route hit!")
    return "Test OK", 200

@app.route("/ping")
def ping():
    print("Ping route hit!")
    return "pong"


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_DIR, filename)
        file.save(file_path)

        # Copy to user's active image
        img = cv2.imread(file_path)
        active_path = get_active_image_path()
        cv2.imwrite(active_path, img)

        return redirect(url_for("index"))
    return "Invalid file type", 400

@app.route("/active-image")
def active_image():
    active_path = get_active_image_path()
    if not os.path.exists(active_path):
        return "No active image", 404
    return send_file(active_path, mimetype="image/png")

@app.route("/convert/pencil", methods=["POST"])
def pencil_sketch():
    path = get_active_image_path()
    if not os.path.exists(path):
        return "No image uploaded", 400

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    inv_blur = 255 - blur
    sketch = cv2.divide(gray, inv_blur, scale=256.0)
    sketch_colored = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(path, sketch_colored)
    return redirect(url_for("index"))

@app.route("/rotate/<direction>", methods=["POST"])
def rotate(direction):
    path = get_active_image_path()
    if not os.path.exists(path):
        return "No image uploaded", 400

    img = cv2.imread(path)
    if direction == "left":
        rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif direction == "right":
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        return "Invalid direction", 400

    cv2.imwrite(path, rotated)
    return redirect(url_for("index"))
# flask_image_editor.py (snippet)

@app.route("/download")
def download():
    path = get_active_image_path()
    if not os.path.exists(path):
        return "No image uploaded", 400
    return send_file(path, as_attachment=True, download_name="sketch.png")

import cv2
import numpy as np
from flask import Flask, request, session, send_file, render_template, redirect, url_for
from scipy.spatial import Voronoi
from sklearn.cluster import KMeans

# Make sure these exist or are imported
def sample_points_by_detail(image, num_points=1000):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    prob_map = magnitude / (np.sum(magnitude) + 1e-8)
    prob_map_flat = prob_map.flatten()

    ys, xs = np.indices((image.shape[:2]))
    coords = np.column_stack((xs.flatten(), ys.flatten()))
    indices = np.random.choice(len(coords), size=num_points, p=prob_map_flat)
    return coords[indices]

def voronoi_mosaic_adaptive(image, num_cells=1000, edge_thickness=1, edge_color=(0, 0, 0), padding=20):
    h, w = image.shape[:2]
    padded = cv2.copyMakeBorder(image, padding, padding, padding, padding, borderType=cv2.BORDER_REFLECT)
    hp, wp = padded.shape[:2]

    sampled_coords = sample_points_by_detail(padded, num_cells)
    points = sampled_coords
    vor = Voronoi(points)
    output = np.zeros_like(padded)

    for region_idx in vor.point_region:
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue

        polygon = [vor.vertices[i] for i in region]
        polygon_np = np.array([polygon], dtype=np.int32)

        mask = np.zeros((hp, wp), dtype=np.uint8)
        cv2.fillPoly(mask, polygon_np, 255)

        mean_color = cv2.mean(padded, mask=mask)[:3]
        mean_color_bgr = tuple(map(int, mean_color))
        cv2.fillPoly(output, polygon_np, mean_color_bgr)

        if edge_thickness > 0:
            cv2.polylines(output, polygon_np, isClosed=True, color=edge_color, thickness=edge_thickness)

    return output[padding:-padding, padding:-padding]

def gaussian_splat(image, num_splats=2000, sigma=3.0, spread=5):
    """Applies Gaussian splatting across the entire image with better coverage."""
    height, width, _ = image.shape
    output = np.zeros_like(image, dtype=np.float32)

    # Generate random points for Gaussian splatting
    x_coords = np.random.randint(0, width, size=num_splats)
    y_coords = np.random.randint(0, height, size=num_splats)

    for i in range(num_splats):
        x, y = x_coords[i], y_coords[i]
        # Sample color from original image
        color = image[y, x]

        # Create Gaussian splat over a larger spread
        for dx in range(-spread, spread + 1):
            for dy in range(-spread, spread + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    weight = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                    output[ny, nx] += weight * color

    # Apply Gaussian blur to fill gaps
    output = cv2.GaussianBlur(output, (7, 7), sigma)
    # Normalize colors
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

################################################################################
# ROUTES FOR THE EXTRA FILTERS
################################################################################

@app.route('/mosaic', methods=['POST'])
def mosaic_route():
    """Applies Voronoi mosaic to the active image."""
    print('mosaic route')
    path = get_active_image_path()
    if not os.path.exists(path):
        return "No active image found", 400

    # Gather optional form parameters
    num_cells = int(request.form.get('num_cells', 2000))
    edge_thickness = int(request.form.get('edge_thickness', 1))

    try:
        image = cv2.imread(path)
        mosaic = voronoi_mosaic_adaptive(
            image,
            num_cells=num_cells,
            edge_thickness=edge_thickness,
            edge_color=(0, 0, 0),
            padding=20
        )
        cv2.imwrite(path, mosaic)
        return redirect(url_for("index"))
    except Exception as e:
        return f"Error applying mosaic: {str(e)}", 500


@app.route('/gaussian-sketch', methods=['POST'])
def gaussian_sketch_route():
    """Converts the uploaded image to a colored sketch using Gaussian splatting and edge blending."""
    path = get_active_image_path()
    if not os.path.exists(path):
        return "No active image found", 400

    # Parse optional parameters
    blur_ksize = int(request.form.get('blur_ksize', 5))
    canny_thresh1 = int(request.form.get('canny_thresh1', 50))
    canny_thresh2 = int(request.form.get('canny_thresh2', 150))
    dilate_iter = int(request.form.get('dilate_iter', 1))
    alpha_blend = float(request.form.get('alpha_blend', 0.6))
    num_gaussians = int(request.form.get('num_gaussians', 5000))
    sigma = float(request.form.get('sigma', 2.0))
    edge_alpha = float(request.form.get('edge_alpha', 0.85))

    try:
        img = cv2.imread(path)
        if img is None:
            return "Could not read active image", 400

        # Convert to grayscale, blur, detect edges
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)

        # Dilation
        kernel = np.ones((2, 2), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=dilate_iter)

        # Gaussian splatting
        splatted_colors = gaussian_splat(img, num_splats=num_gaussians, sigma=sigma)

        # Invert edges & colorify
        edges_inv = cv2.bitwise_not(edges_dilated)
        edges_inv = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)

        # Blend edges over splatted colors
        colored_sketch = cv2.addWeighted(splatted_colors, alpha_blend, img, 1 - alpha_blend, 0)
        colored_sketch = cv2.addWeighted(colored_sketch, edge_alpha, edges_inv, 1 - edge_alpha, 0)

        # Save result
        cv2.imwrite(path, colored_sketch)
        return redirect(url_for("index"))
    except Exception as e:
        return f"Error applying gaussian sketch: {str(e)}", 500


@app.route('/stroke', methods=['POST'])
def stroke_route():
    """Overlays tapered pencil strokes perpendicular to edges."""
    path = get_active_image_path()
    if not os.path.exists(path):
        return "No active image found", 400

    # Parse optional parameters
    stroke_length = int(request.form.get('stroke_length', 5))
    stroke_thickness = int(request.form.get('stroke_thickness', 1))
    inv_density = float(request.form.get('inv_density', 0.7))

    try:
        # Load image in grayscale
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return "Could not read active image", 400

        # Detect edges using Canny
        edges = cv2.Canny(image, 100, 200)

        # Compute gradient direction
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        angles = np.arctan2(sobely, sobelx)  # Edge direction
        angles_perp = angles + np.pi / 2    # Perp direction

        # Create an output color image
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Find edge coordinates
        edge_points = np.argwhere(edges > 0)

        # Draw perpendicular strokes
        for y, x in edge_points:
            if np.random.rand() < inv_density:
                continue
            angle = angles_perp[y, x]
            dx = round(np.cos(angle) * stroke_length / 2)
            dy = round(np.sin(angle) * stroke_length / 2)
            pt1 = (x - dx, y - dy)
            pt2 = (x + dx, y + dy)
            cv2.line(output, pt1, pt2, (0, 0, 0), thickness=stroke_thickness, lineType=cv2.LINE_AA)

        # Save the result
        cv2.imwrite(path, output)
        return redirect(url_for("index"))
    except Exception as e:
        return f"Error applying stroke transformation: {str(e)}", 500
from sklearn.cluster import MiniBatchKMeans

@app.route('/cel-shade', methods=['POST'])
def cel_shade_route():
    from sklearn.cluster import MiniBatchKMeans

    path = get_active_image_path()
    if not os.path.exists(path):
        return "No active image found", 400

    try:
        # Parameters
        bilateral_d = int(request.form.get('bilateral_d', 9))
        bilateral_sigmaColor = int(request.form.get('bilateral_sigmaColor', 75))
        bilateral_sigmaSpace = int(request.form.get('bilateral_sigmaSpace', 75))
        median_blur_ksize = int(request.form.get('median_blur_ksize', 7))
        canny_thresh1 = int(request.form.get('canny_thresh1', 50))
        canny_thresh2 = int(request.form.get('canny_thresh2', 150))
        num_colors = int(request.form.get('num_colors', 12))  # ~12–20 gives stylized color without grayscale

        # Load and preprocess
        img = cv2.imread(path)
        if img is None:
            return "Could not read active image", 400

        smoothed = cv2.bilateralFilter(img, d=bilateral_d,
                                       sigmaColor=bilateral_sigmaColor,
                                       sigmaSpace=bilateral_sigmaSpace)

        # Color quantization with MiniBatchKMeans
        def quantize_image(image, k=8):
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            small = cv2.resize(img_rgb, (0, 0), fx=0.2, fy=0.2)
            pixels = small.reshape(-1, 3)
            kmeans = MiniBatchKMeans(n_clusters=k, batch_size=1000, random_state=42).fit(pixels)
            labels = kmeans.predict(img_rgb.reshape(-1, 3))
            quantized = kmeans.cluster_centers_[labels].reshape(img_rgb.shape)
            return cv2.cvtColor(quantized.astype(np.uint8), cv2.COLOR_RGB2BGR)

        quantized = quantize_image(smoothed, k=num_colors)

        # Edge detection on quantized image
        gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, median_blur_ksize)
        edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)

        # NEW: Edge thickness control
        edge_thickness = int(request.form.get('edge_thickness', 1))
        kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        edges_inv = cv2.bitwise_not(edges)
        edges_inv_colored = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)

        # Overlay black edges over quantized image
        cel_shaded = cv2.multiply(quantized, edges_inv_colored, scale=1/255.0)

        cv2.imwrite(path, cel_shaded)
        return redirect(url_for("index"))
    except Exception as e:
        return f"Error applying cel-shade transformation: {str(e)}", 500







@app.route('/poster', methods=['POST'])
def poster_route():
    """Converts the uploaded image to a poster-style image with color quantization + bold edges."""
    path = get_active_image_path()
    if not os.path.exists(path):
        return "No active image found", 400

    # Parse optional parameters
    posterize_levels = int(request.form.get('posterize_levels', 16))
    color_intensity = float(request.form.get('color_intensity', 0.9))
    gamma = float(request.form.get('gamma', 1.3))

    try:
        img = cv2.imread(path)
        if img is None:
            return "Could not read active image", 400

        # Bilateral filter
        bilateral_filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

        # Posterization via K-Means
        def quantize_image(image, k=16):
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Downsample for faster clustering
            small = cv2.resize(img_rgb, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
            pixels = small.reshape(-1, 3)

            # Fit KMeans on small image
            kmeans = MiniBatchKMeans(n_clusters=k, batch_size=1000, random_state=42).fit(pixels)

            # Apply those clusters to full-res image
            labels = kmeans.predict(img_rgb.reshape(-1, 3))
            quantized_img = kmeans.cluster_centers_[labels].reshape(img_rgb.shape)

            return cv2.cvtColor(quantized_img.astype(np.uint8), cv2.COLOR_RGB2BGR)


        quantized = quantize_image(bilateral_filtered, k=posterize_levels)

        # Grayscale + median blur
        gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 7)

        # Edge detection & enhancement
        edges = cv2.Canny(bilateral_filtered, 50, 150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        edges = cv2.convertScaleAbs(edges, alpha=1.5, beta=0)
        sharp_edges = cv2.addWeighted(edges, 2, cv2.GaussianBlur(edges, (5, 5), 0), -1, 0)
        edges_inv = cv2.bitwise_not(sharp_edges)
        edges_inv_colored = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)

        # Softening + gamma correction
        gray_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        desaturated = cv2.addWeighted(quantized, color_intensity, gray_colored, 1 - color_intensity, 0)
        gamma_corrected = np.array(255 * (desaturated / 255) ** (1 / gamma), dtype=np.uint8)

        # Final blend
        poster_style = cv2.bitwise_and(gamma_corrected, edges_inv_colored)
        cv2.imwrite(path, poster_style)
        return redirect(url_for('index'))
    except Exception as e:
        return f'Error applying poster transformation: {str(e)}', 500
    


from flask import redirect, url_for

@app.route("/generate_pencil_sketch_GAN/", methods=["POST"])
def generate_pencil_sketch():
    path = get_active_image_path()
    if not os.path.exists(path):
        return "No active image found", 400

    try:
        img_tensor = preprocess_image(path)
        with torch.no_grad():
            output = model_pencil(img_tensor)
        result = postprocess_image(output)
        cv2.imwrite(path, result)
        return redirect(url_for('index'))
    except Exception as e:
        return f"Error generating pencil sketch: {str(e)}", 500


@app.route("/generate_edge_sketch_GAN/", methods=["POST"])
def generate_edge_sketch():
    path = get_active_image_path()
    if not os.path.exists(path):
        return "No active image found", 400

    try:
        img_tensor = preprocess_image(path)
        with torch.no_grad():
            output = model_edge(img_tensor)
        result = postprocess_image(output)
        cv2.imwrite(path, result)
        return redirect(url_for('index'))
    except Exception as e:
        return f"Error generating edge sketch: {str(e)}", 500
    
from flask import request, redirect, url_for

@app.route("/colorize_with_image", methods=["POST"])
def colorize_with_image():
    path = get_active_image_path()
    if not os.path.exists(path):
        return "No active image found", 400

    if 'color_source' not in request.files:
        return "No color source image uploaded", 400

    file = request.files['color_source']
    if file.filename == '':
        return "No file selected", 400

    try:
        # Load active image
        base = cv2.imread(path)
        if base is None:
            return "Failed to read active image", 400

        # Load uploaded color image from memory
        file_bytes = np.frombuffer(file.read(), np.uint8)
        color_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if color_img is None:
            return "Failed to decode uploaded image", 400

        # Resize color source to match active image
        color_img = cv2.resize(color_img, (base.shape[1], base.shape[0]))

        # Convert both images to HSV
        base_hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
        color_hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

        # Replace H + S channels in base with those from uploaded image
        result_hsv = base_hsv.copy()
        result_hsv[..., 0] = color_hsv[..., 0]  # Hue
        result_hsv[..., 1] = color_hsv[..., 1]  # Saturation

        # Convert back to BGR and save to active image
        result_bgr = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(path, result_bgr)

        return redirect(url_for("index"))
    except Exception as e:
        return f"Error colorizing image: {str(e)}", 500




if __name__ == "__main__":
    app.run(debug=True)
