document.addEventListener("DOMContentLoaded", function () {
    let uploadedFile = null;
    let uploadedImageElement = new Image();
    let rotationAngle = 0;
    let intensity = 50;  // Initial intensity value (range: 0-100)
    let strokeSize = 3;  // Initial stroke size (range: 1-10)

    console.log("DOMContentLoaded - Script initialized.");

    // Function to upload the image
    async function uploadImage() {
        const fileInput = document.getElementById("imageUpload");
        if (!fileInput.files.length) {
            alert("Please select an image first.");
            console.log("No file selected.");
            return;
        }

        uploadedFile = fileInput.files[0];
        console.log("Uploading file:", uploadedFile);

        const formData = new FormData();
        formData.append("file", uploadedFile);

        try {
            const response = await fetch("/upload/", {
                method: "POST",
                body: formData
            });

            if (!response.ok) throw new Error(`Upload failed: ${response.statusText}`);
            
            const data = await response.json();
            if (!data.filename) throw new Error("Filename missing in response");

            console.log("Upload success:", data);
            fetchConvertedImage(data.filename);

        } catch (error) {
            handleError("uploading the image", error);
        }
    }

    // Function to fetch the converted image from the server
    async function fetchConvertedImage(filename) {
        try {
            console.log("Fetching converted image with filename:", filename);
            const response = await fetch(`/convert/?filename=${filename}`, { method: "POST" });

            if (!response.ok) throw new Error(`Conversion failed: ${response.statusText}`);

            const data = await response.json();
            if (!data.processed_filename) throw new Error("Processed filename missing in response");

            console.log("Image converted:", data);
            
            // Force reload to prevent caching issues
            const imageUrl = `/result/${data.processed_filename}?t=${new Date().getTime()}`;
            console.log("Processed image URL:", imageUrl);

            uploadedImageElement.src = imageUrl;
            
            uploadedImageElement.addEventListener("load", () => {
                console.log("Image successfully loaded, drawing on canvas...");
                drawImageToCanvas();
                document.querySelector('.result').style.display = "block";
                document.getElementById("save-btn").style.display = "inline-block";
            });

        } catch (error) {
            handleError("processing the image", error);
        }
    }

    // Function to draw the image on the canvas
    function drawImageToCanvas() {
        const canvas = document.getElementById("imageCanvas");
        const ctx = canvas.getContext("2d");

        if (!uploadedImageElement.src) {
            console.log("No image source available.");
            return;
        }

        console.log("Drawing image on canvas...");
        canvas.width = uploadedImageElement.width;
        canvas.height = uploadedImageElement.height;

        // Reset the canvas state
        ctx.save();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Center and rotate the image
        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.rotate(rotationAngle * Math.PI / 180);
        ctx.drawImage(uploadedImageElement, -uploadedImageElement.width / 2, -uploadedImageElement.height / 2);
        ctx.restore();

        // Apply pencil sketch with the current intensity and stroke size
        convertToPencilSketch(ctx, canvas);
    }

    // Convert the image to pencil sketch
    function convertToPencilSketch(ctx, canvas) {
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;

        // Apply grayscale and intensity
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];     // Red
            const g = data[i + 1]; // Green
            const b = data[i + 2]; // Blue

            // Calculate grayscale value
            const gray = 0.3 * r + 0.59 * g + 0.11 * b;

            // Apply intensity adjustment (higher intensity = darker sketch)
            const adjustedGray = gray * (intensity / 100);

            // Invert grayscale to get a pencil sketch effect
            data[i] = data[i + 1] = data[i + 2] = 255 - adjustedGray;
        }

        ctx.putImageData(imageData, 0, 0);

        // Apply stroke size effect (blur)
        if (strokeSize > 1) {
            const blurRadius = strokeSize; // Adjust blur radius based on stroke size
            ctx.filter = `blur(${blurRadius}px)`;
            ctx.drawImage(canvas, 0, 0);
            ctx.filter = "none"; // Reset filter
        }

        console.log("Pencil sketch applied with intensity:", intensity, "and stroke size:", strokeSize);
    }

    // Handle errors
    function handleError(action, error) {
        console.error(`Error ${action}:`, error);
        alert(`An error occurred while ${action}. Check console for details.`);
    }

    // Rotate the image by a specific angle
    function rotateImage(angle) {
        rotationAngle += angle;
        console.log(`Rotating image by ${angle} degrees. Current rotation angle: ${rotationAngle}`);
        drawImageToCanvas();
    }

    // Event listeners for sliders to adjust intensity and stroke size
    document.getElementById("intensitySlider").addEventListener("input", (e) => {
        intensity = e.target.value;
        console.log("Intensity adjusted to:", intensity);
        drawImageToCanvas(); // Redraw the canvas with new intensity
    });

    document.getElementById("strokeSlider").addEventListener("input", (e) => {
        strokeSize = e.target.value;
        console.log("Stroke size adjusted to:", strokeSize);
        drawImageToCanvas(); // Redraw the canvas with new stroke size
    });

    // Save the adjusted image
    document.getElementById("save-btn").addEventListener("click", () => {
        const canvas = document.getElementById("imageCanvas");
        const imageUrl = canvas.toDataURL("image/png");

        // Create a link element to download the image
        const link = document.createElement("a");
        link.href = imageUrl;
        link.download = "pencil_sketch.png";
        link.click();
    });

    // Event listeners for upload and rotation buttons
    document.getElementById("uploadBtn").addEventListener("click", uploadImage);
    document.getElementById("rotateLeftBtn").addEventListener("click", () => rotateImage(-90));
    document.getElementById("rotateRightBtn").addEventListener("click", () => rotateImage(90));

    // Exposing the uploadImage function globally for testing
    window.uploadImage = uploadImage;
});
