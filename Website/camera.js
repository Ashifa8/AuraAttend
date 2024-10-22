document.getElementById('cameraButton').addEventListener('click', function () {
    const cameraFeed = document.getElementById('cameraFeed');
    const video = document.getElementById('video');

    // Display the camera feed section
    cameraFeed.style.display = 'flex';

    // Access the user's camera
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(err => {
            console.error("Error accessing the camera: ", err);
            alert('Could not access the camera. Please check your camera settings.');
        });
});

document.getElementById('captureButton').addEventListener('click', function () {
    const video = document.getElementById('video');
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataURL = canvas.toDataURL('image/png');
    console.log('Captured Image:', dataURL);
    
    // Send dataURL to the server for face recognition processing (use fetch or ajax)
    // Example: fetch('/save-image', { method: 'POST', body: JSON.stringify({ image: dataURL }) });
});
