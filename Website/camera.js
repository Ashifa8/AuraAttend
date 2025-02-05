// Add event listener for the camera button
document.getElementById('cameraButton').addEventListener('click', function () {
    const cameraFeed = document.getElementById('cameraFeed');
    const video = document.getElementById('video');

    // Show the camera feed section
    cameraFeed.style.display = 'flex';

    // Request access to the user's camera
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream; // Assign the camera stream to the video element
            video.play(); // Start the video feed
        })
        .catch(err => {
            console.error("Error accessing the camera:", err);
            alert("Could not access the camera. Please check your browser permissions.");
        });
});

// Add event listener for the capture button
document.getElementById('captureButton').addEventListener('click', function () {
    const video = document.getElementById('video');
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');

    // Set canvas size to match the video feed dimensions
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Capture the video frame and draw it onto the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert the canvas image to Base64 format
    const dataURL = canvas.toDataURL('image/png');
    const base64Image = dataURL.split(',')[1];

    console.log("Captured image (base64):", base64Image); // For debugging

    // Send the captured image to the backend for recognition
    fetch('http://127.0.0.1:5000/recognize-face', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Image })
    })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                const results = data.results;
                results.forEach(result => {
                    alert(`Face Recognition Result: ${result.name}`);
                });
            } else {
                alert(`Error: ${data.message}`);
            }
        })
        .catch(err => {
            console.error("Error during face recognition request:", err);
            alert("An error occurred while sending the image to the server.");
        });
});
