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

    async function fetchConvertedImage(filename, intensity = imageState.intensity, strokeSize = imageState.strokeSize, colorize = imageState.colorize) {

        try {
            console.log(`Fetching processed image with intensity=${intensity} and strokeSize=${strokeSize}`);
            
            const response = await fetch(`/convert/?filename=${filename}&intensity=${intensity}&stroke_size=${strokeSize}&colorize=${imageState.colorize}`, { 
                method: "POST" 
            });
            
            if (!response.ok) throw new Error(`Conversion failed: ${response.statusText}`);
    
            const data = await response.json();
            if (!data.processed_filename) throw new Error("Processed filename missing in response");
    
            console.log("Image converted successfully:", data);
    
            // Force reload to prevent caching issues
            const imageUrl = `/result/${data.processed_filename}?t=${new Date().getTime()}`;
            imageState.currentImage = imageUrl;
    
            // Update the static image element
            // Update the static image element and canvas
            document.getElementById("sketchImage").src = imageUrl;
            setTimeout(() => updateCanvasFromImage(imageUrl), 100);

            updateCanvasFromImage(imageUrl);

        } catch (error) {
            console.error("Error processing image:", error);
            alert(`An error occurred while processing the image. Check the console for details.`);
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
        drawImageToCanvas();
    }
    
    
    function drawImageToCanvas() {
        const canvas = document.getElementById("imageCanvas");
        const ctx = canvas.getContext("2d");
    
        if (!imageState.currentImage) {
            console.log("No processed image available for canvas redraw.");
            return;
        }
    
        const tempImg = new Image();
        tempImg.src = imageState.currentImage;
        tempImg.onload = function () {
            canvas.width = tempImg.width;
            canvas.height = tempImg.height;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.save();
            ctx.translate(canvas.width / 2, canvas.height / 2);
            ctx.rotate(imageState.rotationAngle * Math.PI / 180);
            ctx.drawImage(tempImg, -tempImg.width / 2, -tempImg.height / 2);
            ctx.restore();
            
            // Apply the pencil sketch effect (intensity, stroke size)
            convertToPencilSketch(ctx, canvas);
        };
    }
    
    
    
    
    


    // Convert the image to pencil sketch
    

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
    


    document.getElementById("intensitySlider").addEventListener("input", async (e) => {
        imageState.intensity = parseInt(e.target.value, 10);
        console.log("New intensity:", imageState.intensity);
    
        if (imageState.currentImage) {
            const filename = imageState.currentImage.split('/').pop();
            await fetchConvertedImage(filename, imageState.intensity, imageState.strokeSize);
        }
    });
    
    document.getElementById("strokeSlider").addEventListener("input", async (e) => {
        imageState.strokeSize = parseInt(e.target.value, 10);
        console.log("New stroke size:", imageState.strokeSize);
    
        if (imageState.currentImage) {
            const filename = imageState.currentImage.split('/').pop();
            await fetchConvertedImage(filename, imageState.intensity, imageState.strokeSize);
        }
    });

    
    
    
    
    

    // Save the adjusted image
    document.getElementById("save-btn").addEventListener("click", async () => {
        const canvas = document.getElementById("imageCanvas");
    
        if (canvas.width === 0 || canvas.height === 0) {
            alert("No image to save. Please process an image first.");
            return;
        }
    
        // Ensure latest processed image is loaded before saving
        if (imageState.currentImage) {
            await updateCanvasFromImage(imageState.currentImage);
        }
    
        setTimeout(() => {
            const imageUrl = canvas.toDataURL("image/png");
            const link = document.createElement("a");
            link.href = imageUrl;
            link.download = imageState.colorize ? "colorized_sketch.png" : "pencil_sketch.png";
            link.click();
        }, 300); // Small delay to ensure image loads
    });
    
    

    // Event listeners for upload and rotation buttons
    document.getElementById("uploadBtn").addEventListener("click", uploadImage);

    // Exposing the uploadImage function globally for testing
    window.uploadImage = uploadImage;

    // Toggle colorization
    document.getElementById("colorize-btn").addEventListener("click", async () => {
        imageState.colorize = !imageState.colorize;  // Toggle colorization state
        console.log("Colorization toggled:", imageState.colorize);
    
        if (imageState.currentImage) {
            const filename = imageState.currentImage.split('/').pop();
            await fetchConvertedImage(filename, imageState.intensity, imageState.strokeSize, imageState.colorize);
            setTimeout(() => updateCanvasFromImage(imageState.currentImage), 300);
        }
    });
    

    function updateCanvasFromImage(imageUrl) {
        const canvas = document.getElementById("imageCanvas");
        const ctx = canvas.getContext("2d");
    
        const img = new Image();
        img.crossOrigin = "Anonymous"; // Ensures the image can be saved
        img.src = imageUrl + "?t=" + new Date().getTime(); // Prevent caching issues
    
        img.onload = function () {
            console.log("Updating canvas with new image:", imageUrl);
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
        };
    
        img.onerror = function () {
            console.error("Error loading image into canvas:", imageUrl);
        };
    }
    

});
