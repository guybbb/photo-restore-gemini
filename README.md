# Photo Restorer

A Streamlit-based photo restoration tool powered by Google's Gemini API. Mainly built for **Nano Banana** (Gemini image generation models).

## What It Does

- **Restores old photos** - Repairs damage, scratches, folds, and tears using AI
- **Colorizes black & white photos** - Adds natural colors with optional era/style guidelines (e.g., "1950s Kodachrome")
- **Crop & rotate** - Built-in image editing before restoration
- **Multiple output resolutions** - Choose between 1K, 2K, or 4K output
- **Version management** - Generate multiple restoration versions and select your favorite
- **Super resolution export** - Optional 2x upscaling using FSRCNN for final export
- **Custom instructions** - Fine-tune restorations with specific prompts (change colors, adjust lighting, etc.)

## Setup

1.  **Install Dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install google-generativeai Pillow python-dotenv streamlit streamlit-image-cropper opencv-contrib-python
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

### Super Resolution Model

To use the Export "Super resolution (x2)" option, place the model file at:

```
models/FSRCNN_x2.pb
```

Note: the `cv2` module is provided by `opencv-contrib-python` (you do not install a separate `cv2` package).

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
