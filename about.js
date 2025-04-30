document.getElementById("review-form").addEventListener("submit", function(event) {
    event.preventDefault();

    let userName = document.getElementById("user-name").value;
    let thoughts = document.getElementById("thoughts").value;
    let rating = document.querySelector('input[name="rating"]:checked')?.value;

    if (!rating) {
        alert("Please select a rating!");
        return;
    }

    // Debugging: Check the values before sending
    console.log("Form data:", { userName, thoughts, rating });

    fetch("/submit_review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_name: userName, thoughts: thoughts, rating: rating })
    })
    .then(response => response.json())
    .then(data => {
        console.log("Response from Flask:", data);  // Debugging: check the response from Flask
        if (data.message) {
            alert(data.message); // Show alert after successful submission
            document.getElementById("review-form").reset(); // Reset the form
        } else {
            alert("Something went wrong!"); // Handle case when no message is returned
        }
    })
    .catch(error => {
        console.error("Error:", error);
        alert("There was an error submitting the review. Please try again.");
    });
});
