
const video = document.getElementById("webcam");
const canvas = document.getElementById("camera-canvas");
const resultImage = document.getElementById("webcam-result");
const cameraStatus = document.getElementById("camera-status");
const cameraBadge = document.getElementById("camera-badge");
const placeholder = document.getElementById("camera-placeholder");
const startButton = document.getElementById("start-camera");
const stopButton = document.getElementById("stop-camera");

let stream = null;
let timer = null;
let busy = false;

// Start webcam
startButton.addEventListener("click", async () => {
    try {
        if (stream) return;

        stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        });

        video.srcObject = stream;
        video.hidden = false;
        resultImage.hidden = true;
        placeholder.hidden = true;

        await video.play();

        startButton.disabled = true;
        stopButton.disabled = false;
        cameraStatus.textContent = "Camera running. Detecting faces...";
        cameraBadge.textContent = "CAMERA ACTIVE";

        // Send camera frames to Python for detection
        timer = setInterval(sendFrame, 700);
        sendFrame();

    } catch (error) {
        cameraStatus.textContent =
            "Camera error: Allow camera permission and try again.";
        stopCamera();
    }
});

// Detect faces and show the processed frame in the SAME camera area
async function sendFrame() {
    if (!stream || busy || video.videoWidth === 0) return;

    busy = true;

    try {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        canvas.getContext("2d").drawImage(
            video, 0, 0, canvas.width, canvas.height
        );

        const response = await fetch("/predict_frame", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                image: canvas.toDataURL("image/jpeg", 0.7)
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Detection failed");
        }

        // This image overlays the camera preview instead of appearing below it.
        resultImage.src = "data:image/jpeg;base64," + data.image;
        resultImage.hidden = false;

        cameraStatus.textContent = "Live detection is running.";

    } catch (error) {
        cameraStatus.textContent = "Detection error: " + error.message;
    } finally {
        busy = false;
    }
}

// Stop webcam
function stopCamera() {
    if (timer) clearInterval(timer);
    timer = null;

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }

    stream = null;
    video.srcObject = null;
    video.hidden = true;
    resultImage.hidden = true;
    placeholder.hidden = false;

    startButton.disabled = false;
    stopButton.disabled = true;
    cameraBadge.textContent = "CAMERA STANDBY";
}

stopButton.addEventListener("click", () => {
    stopCamera();
    cameraStatus.textContent = "Camera stopped.";
});

// Image upload preview
const imageInput = document.getElementById("image-input");
const dropZone = document.getElementById("drop-zone");
const fileRow = document.getElementById("file-row");
const fileName = document.getElementById("file-name");
const clearButton = document.getElementById("clear-image");
const previewImage = document.getElementById("preview-image");
const emptyResult = document.getElementById("empty-result");
const resultTag = document.getElementById("result-tag");
const uploadForm = document.getElementById("upload-form");
const detectButton = document.getElementById("detect-button");

function removePreviousResult() {
    document.getElementById("server-result")?.remove();
    document.getElementById("server-message")?.remove();

    emptyResult.hidden = true;
    previewImage.hidden = false;
    resultTag.textContent = "IMAGE SELECTED";
}

function selectImage(file) {
    if (!file) return;

    if (!file.type.startsWith("image/")) {
        alert("Please select a valid image.");
        return;
    }

    const transfer = new DataTransfer();
    transfer.items.add(file);
    imageInput.files = transfer.files;

    fileName.textContent = file.name;
    fileRow.hidden = false;
    previewImage.src = URL.createObjectURL(file);

    removePreviousResult();
}

imageInput.addEventListener("change", () => {
    selectImage(imageInput.files[0]);
});

// Drag and drop upload
["dragenter", "dragover"].forEach(eventName => {
    dropZone.addEventListener(eventName, event => {
        event.preventDefault();
        dropZone.classList.add("dragover");
    });
});

["dragleave", "drop"].forEach(eventName => {
    dropZone.addEventListener(eventName, event => {
        event.preventDefault();
        dropZone.classList.remove("dragover");
    });
});

dropZone.addEventListener("drop", event => {
    selectImage(event.dataTransfer.files[0]);
});

// Clear selected image
clearButton.addEventListener("click", () => {
    imageInput.value = "";
    previewImage.removeAttribute("src");
    previewImage.hidden = true;
    fileRow.hidden = true;
    emptyResult.hidden = false;
    resultTag.textContent = "WAITING";
    detectButton.disabled = false;
});

// Submit image to Flask
uploadForm.addEventListener("submit", () => {
    detectButton.disabled = true;
    detectButton.textContent = "Analysing image...";
    resultTag.textContent = "PROCESSING";
});
