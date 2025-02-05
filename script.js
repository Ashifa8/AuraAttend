// Handle Signup Form Submission
// Handle Signup Form Submission
document.getElementById("signupForm")?.addEventListener("submit", function (event) {
    event.preventDefault();

    const username = document.getElementById("newUsername").value;
    const password = document.getElementById("newPassword").value;

    fetch("http://127.0.0.1:5000/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ newUsername: username, newPassword: password })
    })
    .then(response => response.json())
    .then(data => {
        showPopup(data.message, data.status === "success" ? "success" : "error");
    })
    .catch(error => {
        console.error("Error:", error);
        showPopup("An error occurred, please try again.", "error");
    });
});

// Function to Show Popup Message
function showPopup(message, status) {
    let popup = document.getElementById("popup");
    document.getElementById("popupMessage").textContent = message; // Display the message
    popup.style.color = status === "success" ? "green" : "red";
    popup.style.display = "block";

    // Create and append the OK button
    const okButton = document.createElement("button");
    okButton.textContent = "OK";
    okButton.onclick = function () {
        popup.style.display = "none";  // Hide the popup when OK is clicked
        // Redirect to camera.html after clicking OK
        window.location.href = "camera.html";
    };

    // Remove any previous OK buttons to avoid stacking
    const existingOkButton = popup.querySelector("button");
    if (existingOkButton) {
        existingOkButton.remove();
    }

    // Append the OK button to the popup
    popup.appendChild(okButton);
}

document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("loginForm")?.addEventListener("submit", function (event) {
        event.preventDefault(); // Prevent default form submission

        const username = document.getElementById("username").value.trim();
        const password = document.getElementById("password").value.trim();

        if (!username || !password) {
            showPopup("Please enter both username and password.", "error");
            return;
        }

        fetch("http://127.0.0.1:5000/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, password }),
            credentials: "include"  // Ensures cookies are included
        })
        .then(response => response.json())
        .then(data => {
            console.log("Login Response:", data); // Debugging

            if (data.status === "success") {
                // Show popup for successful login
                showPopup("Login successful!", "success", () => {
                    // Redirect to camera page after clicking "OK"
                    window.location.href = "camera.html"; 
                });
            } else {
                showPopup("Invalid username or password", "error");
            }
        })
        .catch(error => {
            console.error("Error:", error);
            showPopup("An error occurred, please try again.", "error");
        });
    });

    // Function to Show Popup Message
    function showPopup(message, status, callback) {
        let popup = document.getElementById("popup");
        document.getElementById("popupMessage").textContent = message; // Display the message
        popup.style.color = status === "success" ? "green" : "red";
        popup.style.display = "block";

        // Create and append the OK button
        const okButton = document.createElement("button");
        okButton.textContent = "OK";
        okButton.onclick = function () {
            popup.style.display = "none";  // Hide the popup when OK is clicked
            if (callback) callback();  // Call the callback function (redirect to camera.html)
        };

        // Remove any previous OK buttons to avoid stacking
        const existingOkButton = popup.querySelector("button");
        if (existingOkButton) {
            existingOkButton.remove();
        }

        // Append the OK button to the popup
        popup.appendChild(okButton);
    }
});
