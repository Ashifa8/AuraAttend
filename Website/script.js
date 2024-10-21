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

