// Elements
const form = document.getElementById("upload-form");
const fileInput = document.getElementById("image");
const previewImg = document.getElementById("preview-img");
const previewSection = document.getElementById("preview-section");
const resultDiv = document.getElementById("result");
const probDiv = document.getElementById("prob");
const heatmapImg = document.getElementById("heatmap-img");
const uploadAnotherBtn = document.getElementById("upload-another");
const submitBtn = document.getElementById("submitBtn");

// Preview image immediately after selecting
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;

  previewImg.src = URL.createObjectURL(file);
  previewSection.style.display = "block";

  // reset result if any
  resultDiv.style.display = "none";
  heatmapImg.src = "";
  probDiv.textContent = "";
  
  const scanLine = document.querySelector(".scan-line");
  if (scanLine) {
    scanLine.classList.remove("active");
  }
});

// Submit form and run prediction
form.addEventListener("submit", async function (e) {
  e.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    alert("Please select an image first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  submitBtn.disabled = true;
  submitBtn.textContent = "Analyzing...";
  
  const scanLine = document.querySelector(".scan-line");
  if (scanLine) {
    scanLine.classList.add("active");
  }

  try {
    const response = await fetch("http://127.0.0.1:8000/api/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(errText || "Prediction failed");
    }

    const data = await response.json();

    probDiv.innerHTML = `
            Prediction: <b>${data.label.toUpperCase()}</b><br>
            Confidence: <b>${(data.prob * 100).toFixed(2)}%</b>
        `;

    heatmapImg.src = data.heatmap; // data URL returned from backend

    resultDiv.style.display = "block";
  } catch (err) {
    alert("Error: " + err.message);
    console.error(err);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Upload & Predict";
    
    if (scanLine) {
      scanLine.classList.remove("active");
    }
  }
});

// Reset to analyze another image
uploadAnotherBtn.addEventListener("click", () => {
  form.reset();
  previewSection.style.display = "none";
  resultDiv.style.display = "none";
  heatmapImg.src = "";
  probDiv.textContent = "";
  
  const scanLine = document.querySelector(".scan-line");
  if (scanLine) {
    scanLine.classList.remove("active");
  }
});
