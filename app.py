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
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

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

    st.markdown("---")
    st.header("Restoration Options")
    enable_colorization = st.checkbox("Colorize Photo", value=True)
    
    era_guidelines = ""
    if enable_colorization:
        era_guidelines = st.text_input(
            "Era/Style Guidelines (Optional)", 
            placeholder="e.g. 1950s Kodachrome, Victorian, 80s vibrant"
        )

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
    # 'timeout': None disables the read timeout, allowing long processing times
    return genai.Client(api_key=key, http_options={'api_version': 'v1alpha', 'timeout': None})

if api_key and image_files:
    client = get_client(api_key)
    
    # Get currently selected file
    current_file_path = image_files[st.session_state.selected_index]
    
    # Check for existing restored versions
    output_dir = input_path / "restored"
    output_dir.mkdir(exist_ok=True)
    
    existing_versions = []
    for f in output_dir.iterdir():
        if f.is_file() and f.name.startswith(f"restored_{current_file_path.name}"):
            existing_versions.append(f)
    existing_versions.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    col1, col2 = st.columns(2)
    
    with col1:
        # Header with Navigation
        h_col, prev_col, next_col = st.columns([3, 1, 1], vertical_alignment="bottom")
        
        with h_col:
            st.subheader(current_file_path.name)
        
        with prev_col:
            if st.button("⬅️ Prev", key="nav_prev", use_container_width=True):
                 st.session_state.selected_index = max(0, st.session_state.selected_index - 1)
                 st.rerun()
                 
        with next_col:
            if st.button("Next ➡️", key="nav_next", use_container_width=True):
                 st.session_state.selected_index = min(len(image_files) - 1, st.session_state.selected_index + 1)
                 st.rerun()

        try:
            image = Image.open(current_file_path)
            st.image(image, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")

    with col2:
        # Header and Selector side-by-side
        header_col, selector_col = st.columns([1, 1])
        with header_col:
            st.subheader("Restored")
        
        radio_key = f"ver_sel_{current_file_path.name}"
        selected_ver_path = None
        
        with selector_col:
            if existing_versions:
                version_names = [f.name for f in existing_versions]
                try:
                    current_idx = version_names.index(st.session_state.get(radio_key, version_names[0]))
                except Exception:
                    current_idx = 0

                st.selectbox(
                    "Select Version",
                    options=version_names,
                    index=current_idx,
                    key=radio_key,
                    label_visibility="collapsed"
                )
                
                # Get selected path for rendering below
                current_choice = st.session_state.get(radio_key, version_names[0])
                for f in existing_versions:
                    if f.name == current_choice:
                        selected_ver_path = f
                        break
        
        # Render Selected Image
        if selected_ver_path:
            try:
                ver_image = Image.open(selected_ver_path)
                st.image(ver_image, use_container_width=True)
                
                # # Download button
                # st.download_button(
                #     label="Download This Version",
                #     data=open(selected_ver_path, "rb").read(),
                #     file_name=selected_ver_path.name,
                #     mime="image/png",
                #     key=f"dl_btn_{selected_ver_path.name}"
                # )
            except Exception as e:
                st.error(f"Error loading {selected_ver_path.name}: {e}")
        else:
            st.info("No restored versions yet. Click the button below to generate one.")

        # st.markdown("---")
        
        # Button to create NEW version
        btn_key = f"create_new_{current_file_path.name}_{len(existing_versions)}"
        
        if st.button("Generate New Version 🪄", key=btn_key, type="primary"):
            with st.spinner("Restoring..."):
                try:
                    # Load image to check size
                    image_to_process = Image.open(current_file_path)
                    
                    # Resize if too large (helps with timeouts/errors)
                    max_dimension = 3072
                    if max(image_to_process.size) > max_dimension:
                        st.warning(f"Image is large ({image_to_process.size}). Resizing to max {max_dimension}px for processing...")
                        image_to_process.thumbnail((max_dimension, max_dimension))
                    
                    # Convert to bytes
                    buf = io.BytesIO()
                    fmt = image_to_process.format if image_to_process.format else "PNG"
                    image_to_process.save(buf, format=fmt)
                    image_bytes = buf.getvalue()
                    mime_type = Image.MIME.get(fmt, "image/png")
                        
                    # Build prompt
                    prompt = "Restore this old photo.\n1. Repair any damage, scratches, folds, or tears.\n"
                    if enable_colorization:
                        prompt += "2. Colorize the photo to look natural."
                        if era_guidelines:
                            prompt += f" Follow these specific era/style guidelines: {era_guidelines}."
                        else:
                            prompt += " Maintain its original form and vintage character."
                    else:
                        prompt += "2. Keep the original black and white (or sepia) tone. Do NOT colorize."

                    prompt += "\n3. Do not change the composition or faces.\nOutput ONLY the restored image."
                    
                    response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=[
                            prompt,
                            types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                        ],
                        config=types.GenerateContentConfig(temperature=0.4)
                    )
                    
                    # Extract image
                    restored_data = None
                    if response.candidates and response.candidates[0].content.parts:
                        for part in response.candidates[0].content.parts:
                            if part.inline_data:
                                restored_data = part.inline_data.data
                                break
                    
                    if restored_data:
                        version_count = len(existing_versions) + 1
                        new_filename = f"restored_{current_file_path.name}_v{version_count}.png"
                        save_path = output_dir / new_filename
                        with open(save_path, "wb") as f:
                            f.write(restored_data)
                        
                        st.success(f"✅ Generated new version: {save_path.name}")
                        try:
                            if response.text:
                                st.markdown("**Model Message:**")
                                st.info(response.text)
                        except Exception:
                            pass
                        st.rerun()
                    else:
                        st.error("No image generated.")
                        try:
                            if response.text:
                                st.markdown("### Model Text Response:")
                                st.info(response.text)
                        except Exception:
                            st.warning("Could not retrieve text response from the model.")

                except Exception as e:
                    st.error(f"Error: {e}")

elif not api_key:
    st.warning("Please enter your API Key in the sidebar.")
elif not image_files:
    st.info("No images found. Please check the folder path in the sidebar.")
