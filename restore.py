import os
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Configuration
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_NAME = "gemini-3-pro-image-preview"

def setup_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env")
        exit(1)
    return genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

def restore_image(client, image_path, output_dir):
    print(f"Processing {image_path.name}...")
    
    try:
        # Load image for the new SDK
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        prompt = (
            "Restore this old photo. "
            "1. Repair any damage, scratches, folds, or tears. "
            "2. Lightly colorize it to look natural but strictly maintain its original form and vintage character. "
            "3. Do not change the composition or faces. "
            "Output ONLY the restored image."
        )

        # Generate using the new SDK syntax
        # The prompt and image are passed as a list of parts
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/png")
            ],
            config=types.GenerateContentConfig(
                temperature=0.4,
            )
        )
        
        # In the new SDK, generated images are often returned as parts with inline_data
        image_data = None
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    image_data = part.inline_data.data
                    break
                if part.text:
                    # Sometimes it might return a URL or text if it fails to embed the image
                    pass

        if not image_data:
            print(f"  ❌ No image data returned for {image_path.name}")
            if response.text:
                print(f"  Response text: {response.text[:200]}...")
            return False

        output_path = output_dir / f"restored_{image_path.name}"
        with open(output_path, "wb") as f:
            f.write(image_data)
            
        print(f"  ✅ Saved to {output_path}")
        return True

    except Exception as e:
        print(f"  ❌ Error processing {image_path.name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Restore old photos using Gemini 3 Pro (Nano Banana Pro).")
    parser.add_argument("path", nargs="?", default=".", help="Path to the directory containing old photos.")
    parser.add_argument("--sample", action="store_true", help="Process only the first image.")
    args = parser.parse_args()

    input_dir = Path(args.path)
    output_dir = input_dir / "restored"
    output_dir.mkdir(exist_ok=True)

    client = setup_client()

    images = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        print("No images found.")
        return

    print(f"Found {len(images)} images. Using {MODEL_NAME}")

    for img_path in images:
        restore_image(client, img_path, output_dir)
        if args.sample:
            break
        time.sleep(10)

if __name__ == "__main__":
    main()