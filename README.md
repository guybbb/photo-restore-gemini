# Photo Restorer

This tool uses the Gemini API (Nano Banana / Gemini 2.0 Flash Exp) to restore and lightly colorize old photos.

## Setup

1.  **Install Dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install google-generativeai Pillow python-dotenv streamlit
    ```

## Usage

### Option 1: Web Interface (Recommended)

1.  **Run the app:**
    ```bash
    streamlit run app.py
    ```
2.  Open the link shown in your browser (usually `http://localhost:8501`).
3.  Enter your API Key in the sidebar.
4.  Upload an image and click "Restore".

### Option 2: Command Line

1.  **Run with path:**
    ```bash
    python3 restore.py /path/to/your/photos
    ```
    (If no path is provided, it defaults to the current directory).
2.  **Output:**
    Restored images will be saved in a `restored/` folder inside the path you provided.

## API Key
You can get a free key at [Google AI Studio](https://aistudio.google.com/).