// Function to check if the user is logged in and update the navbar with username
function checkLoginStatus() {
    fetch("http://127.0.0.1:5000/session-status", {
        method: "GET",
        credentials: "include"
    })
    .then(response => response.json())
    .then(data => {
        if (data.loggedIn) {
            document.getElementById("welcomeMessage").innerText = `Welcome, ${data.username}`;
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
        alert(`Session ended. ${username} has been logged out.`);
        
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
            document.getElementById("welcomeMessage").innerText = `Welcome, ${username}`;
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
        alert(`Session ended. ${username ? username : "Unknown User"} has been logged out.`);
        
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
                        alert(`Face Recognized: ${result.name}`);
                    }
                });

                // If an unknown face is detected, show the registration prompt
                if (unknownDetected) {
                    showRegistrationPrompt(video);
                }
            } else {
                alert(`Error: ${data.message}`);
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
            <p>Enter your class:</p>
            <input type="text" id="classInput" placeholder="Enter your class" style="margin-bottom: 10px; padding: 5px;">
            <button id="registerButton" style="margin: 5px;">Register</button>
        `;

        // Event listener for "Register" button
        document.getElementById('registerButton').addEventListener('click', function () {
            const name = document.getElementById('nameInput').value.trim();
            const className = document.getElementById('classInput').value.trim();
            
            if (!name || !className) {
                alert("Please enter a valid name and class.");
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
                body: JSON.stringify({
                    name: name,
                    class: className,
                    image: newImage
                })
            })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        alert(`Registration successful for ${name}`);
                        container.remove(); // Remove registration prompt
                    } else {
                        alert(`Error: ${data.message}`);
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
                        alert(`Face Recognized: ${result.name}`);
                        markAttendance(result.name);  // Call function to mark attendance
                    }
                });

                // If an unknown face is detected, show the registration prompt
                if (unknownDetected) {
                    showRegistrationPrompt(video);
                }
            } else {
                alert(`Error: ${data.message}`);
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
            <p>Enter your class:</p>
            <input type="text" id="classInput" placeholder="Enter your class" style="margin-bottom: 10px; padding: 5px;">
            <button id="registerButton" style="margin: 5px;">Register</button>
        `;

        // Event listener for "Register" button
        document.getElementById('registerButton').addEventListener('click', function () {
            const name = document.getElementById('nameInput').value.trim();
            const className = document.getElementById('classInput').value.trim();
            
            if (!name || !className) {
                alert("Please enter a valid name and class.");
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
                body: JSON.stringify({
                    name: name,
                    class: className,
                    image: newImage
                })
            })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        alert(`Registration successful for ${name}`);
                        container.remove(); // Remove registration prompt
                    } else {
                        alert(`Error: ${data.message}`);
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

// After recognition, pass the class along with the name
function markAttendance(studentName) {
    // Use the backend function to get the class name based on the student name
    fetch("http://127.0.0.1:5000/get_class_name_for_recognized_face", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            name: studentName
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.className) {
            // If class name is found, mark attendance
            console.log("Marking attendance for:", studentName, "Class:", data.className);
            sendAttendanceToBackend(studentName, data.className);  // Call function to send data to backend
        } else {
            alert("Class not found for " + studentName);
        }
    })
    .catch(error => {
        console.error("Error fetching class name:", error);
        alert("Error fetching class name.");
    });
}

function sendAttendanceToBackend(studentName, className) {
    fetch("http://127.0.0.1:5000/mark_attendance", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        credentials: "include",
        body: JSON.stringify({
            name: studentName,
            class: className
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log(data.message);
            alert("✅ " + data.message);
        } else {
            console.error("❌ Attendance failed:", data.message);
            alert("❌ " + data.message);
        }
    })
    .catch(error => {
        console.error("❌ Error while marking attendance:", error);
        alert("Something went wrong while marking attendance.");
    });
}
document.addEventListener("DOMContentLoaded", function () {
    loadExcels();

    const createBtn = document.getElementById("createExcelBtn");
    if (createBtn) {
        createBtn.addEventListener("click", function () {
            const sheetName = prompt("Enter Excel sheet name.\nPlease use the format: program_section_semNo (e.g., BSCS_SS1_8th):");
            if (!sheetName) return;

            fetch("http://127.0.0.1:5000/create_excel", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                credentials: "include",
                body: JSON.stringify({ sheet_name: sheetName })
            })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    displayExcel(data.sheet_data, data.excel_id);
                    document.getElementById("saveSheetBtn").setAttribute("data-excel-id", data.excel_id);
                    document.getElementById("saveSheetBtn").style.display = "block";
                    loadExcels();
                } else {
                    alert("Failed to create sheet: " + data.message);
                }
            })
            .catch(error => {
                console.error("Error creating sheet:", error);
                alert("Something went wrong while creating the sheet.");
            });
        });
    }

    const saveBtn = document.getElementById("saveSheetBtn");
    if (saveBtn) {
        saveBtn.addEventListener("click", saveSheet);
    }
});

function loadExcels() {
    fetch("http://127.0.0.1:5000/get_excels", {
        method: "GET",
        credentials: "include"
    })
    .then(response => response.json())
    .then(data => {
        const container = document.getElementById("excelContainer");
        container.innerHTML = "";

        if (data.success && Array.isArray(data.excels)) {
            data.excels.forEach(excel => {
                const sheetBtn = document.createElement("button");
                sheetBtn.textContent = excel.file_name;

                sheetBtn.className = "sheet-button";
                sheetBtn.onclick = function () {
                    fetchExcelContent(excel.id);
                };

                // Create delete button for the sheet
                const deleteBtn = document.createElement("button");
                deleteBtn.textContent = "Delete";
                deleteBtn.className = "delete-button";
                deleteBtn.onclick = function () {
                    deleteExcel(excel.id);
                };

                const buttonContainer = document.createElement("div");
                buttonContainer.className = "sheet-btn-container";
                buttonContainer.appendChild(sheetBtn);
                buttonContainer.appendChild(deleteBtn);

                container.appendChild(buttonContainer);
            });
        } else {
            container.innerHTML = "<p>No Excel files available.</p>";
        }
    })
    .catch(err => {
        console.error("Error loading Excel list:", err);
    });
}

function fetchExcelContent(excelId) {
    fetch(`http://127.0.0.1:5000/load_excel/${excelId}`, {
        method: "GET",
        credentials: "include"
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayExcel(data.sheet_data, excelId);
            document.getElementById("saveSheetBtn").setAttribute("data-excel-id", excelId);
            document.getElementById("saveSheetBtn").style.display = "block";
        } else {
            alert("Failed to load Excel content.");
        }
    })
    .catch(err => {
        console.error("Error loading Excel content:", err);
    });
}

function displayExcel(sheetData, excelId) {
    const fullScreen = document.getElementById("fullScreenExcel");
    const luckysheetDiv = document.getElementById("luckysheet");

    fullScreen.style.display = "block";
    luckysheetDiv.style.height = "100%";

    luckysheet.destroy();
    luckysheet.create({
        container: 'luckysheet',
        data: sheetData,
        showinfobar: true,
        lang: 'en',
        allowEdit: true,
        title: "Excel Sheet"
    });

    let closeButton = document.querySelector("#fullScreenExcel .close-button");
    if (!closeButton) {
        closeButton = document.createElement("button");
        closeButton.classList.add("close-button");
        closeButton.innerHTML = "Close";
        closeButton.onclick = closeExcel;
        fullScreen.appendChild(closeButton);
    } else {
        closeButton.style.display = "block";
    }

    const saveBtn = document.getElementById("saveSheetBtn");
    saveBtn.setAttribute("data-excel-id", excelId);
    saveBtn.style.display = "block";
}

function closeExcel() {
    document.getElementById("fullScreenExcel").style.display = "none";

    const closeBtn = document.querySelector(".close-button");
    if (closeBtn) closeBtn.style.display = "none";

    const saveBtn = document.getElementById("saveSheetBtn");
    if (saveBtn) saveBtn.style.display = "none";
}

function saveSheet() {
    const sheetData = luckysheet.getAllSheets();
    const excelId = document.getElementById("saveSheetBtn").getAttribute("data-excel-id");

    if (!excelId) {
        alert("No Excel sheet selected to save.");
        return;
    }

    fetch("http://127.0.0.1:5000/save_excel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ sheet_json: sheetData, excel_id: excelId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert("Sheet saved successfully!");
        } else {
            alert("Failed to save sheet.");
        }
    })
    .catch(error => {
        console.error("Error saving sheet:", error);
        alert("Something went wrong while saving the sheet.");
    });
}
function deleteExcel(excelId) {
    if (confirm("Are you sure you want to delete this Excel file?")) {
        fetch(`http://127.0.0.1:5000/delete_excel/${excelId}`, {
            method: "POST",
            credentials: "include"
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert("Excel file deleted successfully!");
                loadExcels();  // Reload the list of Excel files after deletion
            } else {
                alert("Failed to delete the file: " + data.message);
            }
        })
        .catch(err => {
            console.error("Error deleting Excel file:", err);
            alert("Something went wrong while deleting the file.");
        });
    }
}
