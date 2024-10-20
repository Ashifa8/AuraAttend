document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form submission

    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    // Simple validation (replace with actual authentication logic)
    if (username === 'user' && password === 'pass') {
        alert('Login successful!');
        // Redirect or proceed to the next page
    } else {
        alert('Invalid username or password.');
    }
});

// Redirect to the sign-in page
document.getElementById('signupButton').addEventListener('click', function() {
    window.location.href = 'sign Up.html'; // Redirect to the sign-in page
});

