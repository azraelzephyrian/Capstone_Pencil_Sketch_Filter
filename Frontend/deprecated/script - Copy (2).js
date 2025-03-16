document.addEventListener("DOMContentLoaded", function () {
    let uploadedFile = null;
    let uploadedImageElement = new Image();
    let rotationAngle = 0;
    let intensity = 50;  // Initial intensity value (range: 0-100)
    let strokeSize = 3;  // Initial stroke size (range: 1-10)

    const imageState = {
        originalFile: null,  // Stores the original uploaded image file
        currentImage: null,  // Stores the current transformed image URL
        rotationAngle: 0,    // Stores the current rotation state
        intensity: 50,       // Default intensity
        strokeSize: 3,       // Default stroke size
    };
    
    console.log("DOMContentLoaded - Script initialized.");

    // Function to upload the image
    async function uploadImage() {
        const fileInput = document.getElementById("imageUpload");
        if (!fileInput.files.length) {
            alert("Please select an image first.");
            console.log("No file selected.");
            return;
        }

        imageState.originalFile = fileInput.files[0]; // Store original file
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
            imageState.currentImage = `/result/${data.filename}`; // Save uploaded image path
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
    
            // Update state and UI
            imageState.currentImage = imageUrl;
            updateImageDisplay();
    
        } catch (error) {
            handleError("processing the image", error);
        }
    }
    

    // Function to draw the image on the canvas
    // Function to draw the image on the canvas
    function updateImageDisplay() {
        if (!imageState.currentImage) {
            console.log("No processed image available.");
            return;
        }
    
        console.log("Updating displayed image to:", imageState.currentImage);
        document.getElementById("sketchImage").src = imageState.currentImage;
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
            const adjustedGray = gray * (imageState.intensity / 100);


            // Invert grayscale to get a pencil sketch effect
            data[i] = data[i + 1] = data[i + 2] = 255 - adjustedGray;
        }

        ctx.putImageData(imageData, 0, 0);

        // Apply stroke size effect (blur)
        if (imageState.strokeSize > 1) {
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
    async function rotateImage(direction) {
        if (!imageState.currentImage) {
            alert("No image has been processed yet.");
            return;
        }
    
        const filename = imageState.currentImage.split('/').pop(); // Extract filename from URL
    
        try {
            const response = await fetch(`/rotate-${direction}/?filename=${filename}`, { method: "POST" });
    
            if (!response.ok) throw new Error(`Rotation failed: ${response.statusText}`);
    
            const data = await response.json();
            if (!data.processed_filename) throw new Error("Processed filename missing in response");
    
            console.log("Image rotated:", data);
    
            // Update state with new rotated image URL
            imageState.currentImage = `/result/${data.processed_filename}?t=${new Date().getTime()}`;
            
            updateImageDisplay(); // Update the UI to show the rotated image
        } catch (error) {
            handleError("rotating the image", error);
        }
    }
    
    // Event listeners for rotation buttons
    document.getElementById("rotateLeftBtn").addEventListener("click", () => rotateImage("left"));
    document.getElementById("rotateRightBtn").addEventListener("click", () => rotateImage("right"));
    
    

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
