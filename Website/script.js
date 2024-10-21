document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault(); 

    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    
    if (username === 'user' && password === 'pass') {
        alert('Login successful!');
        
    } else {
        alert('Invalid username or password.');
    }
});


document.getElementById('signupButton').addEventListener('click', function() {
    window.location.href = 'sign Up.html'; 
});
document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevents the form from submitting the traditional way

    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    // Basic validation (replace with actual validation or server-side validation)
    if (username === 'admin' && password === 'admin123') {
        alert('Login successful!');
        // Redirect to the camera.html page on successful login
        window.location.href = 'camera.html';
    } else {
        alert('Invalid username or password. Please try again.');
    }
});


