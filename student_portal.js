// Function to search for files based on class name
function searchFiles() {
    const className = document.getElementById("searchInput").value.trim();
    console.log("Search initiated with class name:", className);
  
    if (className === "") {
      document.getElementById("fileList").innerHTML = "";
      return;
    }
  
    fetch(`http://127.0.0.1:5000/search_excel_files?class_name=${encodeURIComponent(className)}`)
      .then(response => response.json())
      .then(data => {
        console.log("Response data:", data);
        const fileList = document.getElementById("fileList");
        fileList.innerHTML = "";
  
        if (data.files.length === 0) {
          fileList.innerHTML = "<li>No files found matching your search</li>";
          return;
        }
  
        data.files.forEach((file) => {
          const li = document.createElement("li");
          li.textContent = file.file_name;
          li.onclick = () => loadExcelData(file.file_path); // Load file only in excel.html
          fileList.appendChild(li);
        });
  
        // Save file list
        window.excelFileList = data.files;
      })
      .catch(err => {
        console.error("Error fetching files:", err);
      });
  }
  
  // Function to handle file click and redirect to Luckysheet view
  function loadExcelData(filePath) {
    console.log("Opening LuckySheet for:", filePath);
    window.location.href = `excel.html?file_path=${encodeURIComponent(filePath)}`; // Redirect to excel.html
  }
  
  // Enter key triggers search
  document.getElementById("searchInput").addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
      searchFiles();
    }
  });
  
  // Search icon click
  document.querySelector(".search-icon").addEventListener("click", function () {
    searchFiles();
  });
  const form = document.querySelector('.faq-form');
  form.addEventListener('submit', async (e) => {
    e.preventDefault(); // Prevent default form behavior
  
    const formData = new FormData(form);
  
    try {
      const response = await fetch('http://127.0.0.1:5000/submit_query', {
        method: 'POST',
        body: formData,
      });
  
      const result = await response.json();
  
      if (response.ok) {
        alert("✅ " + result.message);
        form.reset();
      } else {
        alert("❌ " + result.message);
      }
    } catch (error) {
      alert("❌ Something went wrong: " + error.message);
    }
  });