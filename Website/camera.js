// Add event listener for the camera button
// Function to check if the user is logged in and update the navbar with username
function checkLoginStatus() {
    fetch("http://127.0.0.1:5000/session-status", {
        method: "GET",
        credentials: "include"
    })
    .then(response => response.json())
    .then(data => {
        if (data.loggedIn) {
            document.getElementById("welcomeMessage").innerText = Welcome, ${data.username};
        } else {
            window.location.href = "index.html"; // Redirect to login if not logged in
        }
    })
    .catch(error => console.error("Error checking session:", error));
}

// Logout function with alert showing the username
function logout() {
    fetch("http://127.0.0.1:5000/logout", {
        method: "GET",
        credentials: "include"
    })
    .then(() => {
        let username = localStorage.getItem("username");  // Get username from local storage
        localStorage.removeItem("username");  // Clear username from storage
        
        // Show alert with the username
        alert(Session ended. ${username} has been logged out.);
        
        // Redirect to login page
        window.location.href = "index.html"; 
    })
    .catch(error => console.error("Error logging out:", error));
}
// Function to get cookie value by name
function getCookie(name) {
    let cookies = document.cookie.split("; ");
    for (let cookie of cookies) {
        let [key, value] = cookie.split("=");
        if (key === name) return decodeURIComponent(value);
    }
    return null;
}

function checkLoginStatus() {
    fetch("http://127.0.0.1:5000/session-status", { credentials: "include" })
    .then(response => response.json())
    .then(data => {
        let username = data.loggedIn ? data.username : getCookie("username"); // Use session first, then cookie
        
        if (username) {
            console.log("Username found:", username); // Debugging
            document.getElementById("welcomeMessage").innerText = Welcome, ${username};
            localStorage.setItem("username", username); // Store for logout use
        } else {
            console.log("No session or cookie found, redirecting...");
            window.location.href = "index.html"; // Redirect if not logged in
        }
    })
    .catch(error => console.error("Error checking session:", error));
}
function logout() {
    let username = localStorage.getItem("username") || getCookie("username");  // Get username from localStorage or cookie

    fetch("http://127.0.0.1:5000/logout", {
        method: "GET",
        credentials: "include"
    })
    .then(() => {
        document.cookie = "username=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;"; // Clear cookie
        localStorage.removeItem("username"); // Clear local storage

        // Show alert with the correct username
        alert(Session ended. ${username ? username : "Unknown User"} has been logged out.);
        
        // Redirect to login page
        window.location.href = "index.html"; 
    })
    .catch(error => console.error("Error logging out:", error));
}


// Run on page load
window.onload = checkLoginStatus;


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
                let unknownDetected = false;

                // Process results
                results.forEach(result => {
                    if (result.name === "Unknown") {
                        unknownDetected = true;
                    } else {
                        alert(Face Recognized: ${result.name});
                    }
                });

                // If an unknown face is detected, show the registration prompt
                if (unknownDetected) {
                    showRegistrationPrompt(video);
                }
            } else {
                alert(Error: ${data.message});
            }
        })
        .catch(err => {
            console.error("Error during face recognition request:", err);
            alert("An error occurred while sending the image to the server.");
        });
});

// Function to show the registration prompt
function showRegistrationPrompt(video) {
    // Check if a registration prompt is already present to avoid duplicates
    if (document.querySelector('.registration-prompt')) {
        console.warn("Registration prompt already displayed.");
        return;
    }

    const container = document.createElement('div');
    container.className = 'registration-prompt';
    container.style.position = 'fixed';
    container.style.top = '50%';
    container.style.left = '50%';
    container.style.transform = 'translate(-50%, -50%)';
    container.style.zIndex = '1000';
    container.style.border = '1px solid #ccc';
    container.style.padding = '20px';
    container.style.backgroundColor = 'white';
    container.style.textAlign = 'center';
    container.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.2)';

    container.innerHTML = `
        <p>Unrecognized face detected. Do you want to register?</p>
        <button id="yesButton" style="margin: 5px;">Yes</button>
        <button id="noButton" style="margin: 5px;">No</button>
    `;

    document.body.appendChild(container);

    // Event listener for "Yes" button
    document.getElementById('yesButton').addEventListener('click', function () {
        container.innerHTML = `
            <p>Enter your name:</p>
            <input type="text" id="nameInput" placeholder="Enter your name" style="margin-bottom: 10px; padding: 5px;">
            <button id="registerButton" style="margin: 5px;">Register</button>
        `;

        // Event listener for "Register" button
        document.getElementById('registerButton').addEventListener('click', function () {
            const name = document.getElementById('nameInput').value.trim();
            if (!name) {
                alert("Please enter a valid name.");
                return;
            }

            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');

            // Set canvas size to match the video feed dimensions
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            // Capture the video frame and draw it onto the canvas
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Convert the canvas image to Base64 format
            const newImage = canvas.toDataURL('image/png').split(',')[1];

            // Send the new face data to the backend for registration
            fetch('http://127.0.0.1:5000/register-face', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name, image: newImage })
            })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        alert(Registration successful for ${name});
                        container.remove(); // Remove registration prompt
                    } else {
                        alert(Error: ${data.message});
                    }
                })
                .catch(err => {
                    console.error("Error during registration request:", err);
                    alert("An error occurred while registering the face.");
                });
        });
    });

    // Event listener for "No" button
    document.getElementById('noButton').addEventListener('click', function () {
        container.remove(); // Remove registration prompt
    });
}
