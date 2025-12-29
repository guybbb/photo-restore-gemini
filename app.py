import streamlit as st
import os
import io
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "gemini-3-pro-image-preview"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

st.set_page_config(page_title="Gemini Photo Restorer", page_icon="📷", layout="wide")

st.title("✨ Gemini Photo Restorer")
st.markdown(f"Using model: **{MODEL_NAME}**")

# Sidebar
with st.sidebar:
    st.header("Settings")
    api_key_env = os.getenv("GOOGLE_API_KEY")
    api_key = st.text_input("API Key", value=api_key_env if api_key_env else "", type="password")
    
    st.markdown("---")
    st.header("Photo Source")
    default_path = "/Users/guybalzam/Desktop/Mom Photos/ready"
    if "input_path" not in st.session_state:
        st.session_state.input_path = default_path
        
    input_path_str = st.text_input("Folder Path", value=st.session_state.input_path)
    
    # Refresh button to reload images
    if st.button("Scan Folder"):
        st.session_state.input_path = input_path_str
        st.rerun()

    # Get images
    input_path = Path(input_path_str)
    image_files = []
    if input_path.exists() and input_path.is_dir():
        image_files = sorted([
            f for f in input_path.iterdir() 
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])
    
    if not image_files:
        st.warning("No images found in path.")
    else:
        st.success(f"Found {len(image_files)} images.")
        
        # Selection logic
        if "selected_index" not in st.session_state:
            st.session_state.selected_index = 0
            
        # Navigation
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅️ Previous", use_container_width=True):
                st.session_state.selected_index = max(0, st.session_state.selected_index - 1)
        with col_next:
            if st.button("Next ➡️", use_container_width=True):
                st.session_state.selected_index = min(len(image_files) - 1, st.session_state.selected_index + 1)

        # Dropdown selection (syncs with index)
        selected_file_name = st.selectbox(
            "Select Photo", 
            options=[f.name for f in image_files],
            index=st.session_state.selected_index
        )
        
        # Update index if dropdown changed manually
        new_index = [f.name for f in image_files].index(selected_file_name)
        if new_index != st.session_state.selected_index:
            st.session_state.selected_index = new_index
            st.rerun()

def get_client(key):
    return genai.Client(api_key=key, http_options={'api_version': 'v1alpha'})

if api_key and image_files:
    client = get_client(api_key)
    
    # Get currently selected file
    current_file_path = image_files[st.session_state.selected_index]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Original: {current_file_path.name}")
        try:
            image = Image.open(current_file_path)
            st.image(image, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")

    with col2:
        st.subheader("Restored")
        
        # Unique key for button so it resets per image
        restore_btn_key = f"restore_{current_file_path.name}"
        
        # Check if already restored (check output folder)
        output_dir = input_path / "restored"
        output_dir.mkdir(exist_ok=True)
        restored_path = output_dir / f"restored_{current_file_path.name}"
        
        already_restored = restored_path.exists()
        
        if already_restored:
            try:
                restored_image = Image.open(restored_path)
                st.image(restored_image, use_container_width=True, caption="Loaded from disk")
                if st.button("Re-run Restoration 🔄", key=restore_btn_key):
                    already_restored = False # Force re-run logic below
            except:
                already_restored = False

        if not already_restored:
            if st.button("Restore Photo 🪄", key=restore_btn_key, type="primary"):
                with st.spinner("Restoring... This usually takes 10-20 seconds."):
                    try:
                        # Load image bytes
                        with open(current_file_path, "rb") as f:
                            image_bytes = f.read()
                        
                        # Determine MIME type
                        mime_type = Image.MIME.get(image.format, "image/png")
                            
                        prompt = (
                            "Restore this old photo. "
                            "1. Repair any damage, scratches, folds, or tears. "
                            "2. Lightly colorize it to look natural but strictly maintain its original form and character. "
                            "3. Do not change the composition or faces. "
                            "Output ONLY the restored image."
                        )
                        
                        response = client.models.generate_content(
                            model=MODEL_NAME,
                            contents=[
                                prompt,
                                types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                            ],
                        )
                        
                        # Extract image
                        restored_data = None
                        if response.candidates and response.candidates[0].content.parts:
                            for part in response.candidates[0].content.parts:
                                if part.inline_data:
                                    restored_data = part.inline_data.data
                                    break
                        
                        if restored_data:
                            # Save to disk
                            with open(restored_path, "wb") as f:
                                f.write(restored_data)
                            
                            st.image(restored_data, use_container_width=True)
                            st.success(f"✅ Saved to: {restored_path}")
                            
                            # Brief pause before reload to let user see success
                            st.button("Click to Refresh View")
                        else:
                            st.error("No image generated.")
                            if response.text:
                                st.warning(f"Response: {response.text}")

                    except Exception as e:
                        st.error(f"Error: {e}")

elif not api_key:
    st.warning("Please enter your API Key in the sidebar.")
elif not image_files:
    st.info("No images found. Please check the folder path in the sidebar.")
