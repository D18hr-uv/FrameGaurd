// Elements
const form = document.getElementById("upload-form");
const fileInput = document.getElementById("image");
const previewImg = document.getElementById("preview-img");
const previewSection = document.getElementById("preview-section");
const resultDiv = document.getElementById("result");
const heatmapImg = document.getElementById("heatmap-img");
const uploadAnotherBtn = document.getElementById("upload-another");
const submitBtn = document.getElementById("submitBtn");

// Verdict elements
const verdictBanner = document.getElementById("verdict-banner");
const verdictIcon = document.getElementById("verdict-icon");
const verdictLabel = document.getElementById("verdict-label");
const verdictConfidence = document.getElementById("verdict-confidence");

// Branch meters
const spatialValue = document.getElementById("spatial-value");
const spatialBar = document.getElementById("spatial-bar");
const fftValue = document.getElementById("fft-value");
const fftBar = document.getElementById("fft-bar");
const fusionValue = document.getElementById("fusion-value");
const fusionBar = document.getElementById("fusion-bar");

// Preview image immediately after selecting
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;

  previewImg.src = URL.createObjectURL(file);
  previewSection.style.display = "block";

  // reset result if any
  resultDiv.style.display = "none";
  heatmapImg.src = "";
  resetMeters();
  
  const scanLine = document.querySelector(".scan-line");
  if (scanLine) {
    scanLine.classList.remove("active");
  }
});

function resetMeters() {
  spatialValue.textContent = "—";
  fftValue.textContent = "—";
  fusionValue.textContent = "—";
  spatialBar.style.width = "0%";
  fftBar.style.width = "0%";
  fusionBar.style.width = "0%";
  verdictBanner.className = "verdict-banner";
}

function animateMeter(barEl, valueEl, targetPercent, delay) {
  setTimeout(() => {
    barEl.style.width = targetPercent + "%";
    valueEl.textContent = targetPercent.toFixed(1) + "%";
  }, delay);
}

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

    // ---- Update Verdict Banner ----
    const isFake = data.label === "fake";
    verdictBanner.classList.add(isFake ? "verdict-fake" : "verdict-real");
    verdictIcon.textContent = isFake ? "⚠" : "✓";
    verdictLabel.textContent = data.label.toUpperCase();
    verdictConfidence.textContent = `${(data.prob * 100).toFixed(1)}% Confidence`;

    // ---- Animate Branch Meters ----
    const spatialPct = (data.spatial_fake_conf * 100);
    const fftPct = (data.fft_fake_conf * 100);
    const fusionPct = (data.fusion_score * 100);

    animateMeter(spatialBar, spatialValue, spatialPct, 200);
    animateMeter(fftBar, fftValue, fftPct, 400);
    animateMeter(fusionBar, fusionValue, fusionPct, 600);

    // Color the bars based on severity
    colorBar(spatialBar, spatialPct);
    colorBar(fftBar, fftPct);
    colorBar(fusionBar, fusionPct);

    // ---- Heatmap ----
    heatmapImg.src = data.heatmap;

    resultDiv.style.display = "block";
  } catch (err) {
    alert("Error: " + err.message);
    console.error(err);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Initialize Scan";
    
    if (scanLine) {
      scanLine.classList.remove("active");
    }
  }
});

function colorBar(barEl, pct) {
  // Low = green, mid = yellow, high = red
  if (pct < 35) {
    barEl.style.background = "linear-gradient(90deg, #00ff88, #00cc6a)";
  } else if (pct < 65) {
    barEl.style.background = "linear-gradient(90deg, #ffaa00, #ff8800)";
  } else {
    barEl.style.background = "linear-gradient(90deg, #ff4466, #ff1133)";
  }
}

// Reset to analyze another image
uploadAnotherBtn.addEventListener("click", () => {
  form.reset();
  previewSection.style.display = "none";
  resultDiv.style.display = "none";
  heatmapImg.src = "";
  resetMeters();
  
  const scanLine = document.querySelector(".scan-line");
  if (scanLine) {
    scanLine.classList.remove("active");
  }
});
