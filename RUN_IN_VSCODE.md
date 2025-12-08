# How to Run the Deepfake Detection Project in VS Code

This guide provides step-by-step instructions to set up and run the **FrameGuard** deepfake detection system using Visual Studio Code (VS Code).

## Prerequisites

*   **VS Code** installed.
*   **Python 3.8+** installed.

## 1. Open the Project in VS Code

1.  Launch VS Code.
2.  Go to **File > Open Folder...**
3.  Select the root folder of this project (where `requirements.txt` is located).

## 2. Set Up a Python Virtual Environment

It is recommended to use a virtual environment to manage dependencies in isolation.

1.  Open the **Integrated Terminal** in VS Code (`Ctrl + ~` or **Terminal > New Terminal**).
2.  Run the following command to create a virtual environment named `venv`:

    ```bash
    python -m venv venv
    ```

    *(Note: On some systems, you might need to use `python3` instead of `python`.)*

3.  **Activate the Virtual Environment:**

    *   **Windows (Command Prompt):**
        ```cmd
        venv\Scripts\activate
        ```
    *   **Windows (PowerShell):**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    *   **macOS / Linux:**
        ```bash
        source venv/bin/activate
        ```

    Once activated, your terminal prompt should show `(venv)` at the beginning.

## 3. Install Dependencies

With the virtual environment activated, install the required Python packages:

```bash
pip install -r requirements.txt
```

This will install **FastAPI**, **Uvicorn**, **PyTorch**, **OpenCV**, and other necessary libraries.

## 4. Run the Backend Server

The backend is built with FastAPI. To start the server:

1.  Ensure you are still in the root directory and your virtual environment is activated.
2.  Run the following command:

    ```bash
    uvicorn backend.api.app:app --reload
    ```

3.  You should see output indicating the server has started, typically at `http://127.0.0.1:8000`.

## 5. Run the Frontend

The frontend is a simple HTML/JS application.

**Option A: Using "Live Server" Extension (Recommended)**

1.  Install the **Live Server** extension in VS Code (by Ritwick Dey).
2.  Open `frontend/index.html` in the editor.
3.  Right-click anywhere in the code and select **Open with Live Server**.
4.  This will automatically open your browser pointing to the frontend (usually `http://127.0.0.1:5500`).

**Option B: Direct File Open**

1.  Navigate to the `frontend` folder in your file explorer.
2.  Double-click `index.html` to open it in your web browser.

## 6. Testing the Application

1.  With both the Backend (terminal) and Frontend (browser) running:
2.  Click **Choose File** in the web interface.
3.  Select an image to analyze.
4.  Click **Upload & Predict**.
5.  The system will display:
    *   **Prediction:** Real or Fake.
    *   **Confidence:** The probability score.
    *   **Grad-CAM Heatmap:** A visual overlay showing which parts of the image influenced the decision.
