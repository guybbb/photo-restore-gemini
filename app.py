import streamlit as st
import os
import io
import shutil
from pathlib import Path
from datetime import datetime
from PIL import Image
from streamlit_cropper import st_cropper

import cv2
from cv2 import dnn_superres
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "gemini-3-pro-image-preview"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
SR_SCALE = 2
SR_MAX_DIM = 2000
SR_MODEL_PATH = Path(__file__).parent / "models" / "FSRCNN_x2.pb"

st.set_page_config(page_title="Gemini Photo Restorer", page_icon="📷", layout="wide")

st.title("✨ Gemini Photo Restorer")
st.markdown(f"Using model: **{MODEL_NAME}**")

def get_restored_versions(restored_dir, original_file):
    if not restored_dir.exists():
        return []
    versions = [
        f for f in restored_dir.iterdir()
        if f.is_file() and f.name.startswith(f"restored_{original_file.name}")
    ]
    versions.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return versions

def get_selected_version(versions):
    for v in versions:
        if "-selected" in v.stem:
            return v
    if len(versions) == 1:
        return versions[0]
    return None

def get_rotate_key(file_path):
    return f"rotate_deg_{file_path.name}"

def get_edited_path(file_path):
    return file_path.parent / f"{file_path.stem}-edited{file_path.suffix}"

def load_source_image(file_path):
    edited_path = get_edited_path(file_path)
    if edited_path.exists():
        return Image.open(edited_path)
    return Image.open(file_path)

def is_upscale_allowed(image_path):
    try:
        with Image.open(image_path) as image:
            max_dim = max(image.size)
        return max_dim < SR_MAX_DIM, max_dim
    except Exception:
        return False, None

@st.cache_resource
def load_superres_model(model_path_str):
    sr_impl = dnn_superres.DnnSuperResImpl_create()
    sr_impl.readModel(model_path_str)
    sr_impl.setModel("fsrcnn", SR_SCALE)
    return sr_impl

def ensure_superres_ready():
    if cv2 is None or dnn_superres is None:
        return None, "OpenCV not available. Install opencv-contrib-python to enable super resolution."
    if not SR_MODEL_PATH.exists():
        return None, f"Missing super resolution model: {SR_MODEL_PATH}"
    return load_superres_model(str(SR_MODEL_PATH)), None

def upscale_image_to_bytes(input_path, sr_impl):
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Image not found: {input_path}")
    upscaled = sr_impl.upsample(image)
    success, buffer = cv2.imencode(".png", upscaled)
    if not success:
        raise ValueError("Failed to encode upscaled image.")
    return buffer.tobytes()

def upscale_with_superres(input_path, output_path, sr_impl):
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Image not found: {input_path}")
    upscaled = sr_impl.upsample(image)
    cv2.imwrite(str(output_path), upscaled)

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

    output_resolution = st.selectbox(
        "Output Resolution",
        options=["1K", "2K", "4K"],
        index=0,  # Default to 2K
        help="Higher resolution = better quality but slower processing"
    )
    
    era_guidelines = ""
    if enable_colorization:
        era_guidelines = st.text_input(
            "Era/Style Guidelines (Optional)",
            placeholder="e.g. 1950s Kodachrome, Victorian, 80s vibrant"
        )

    st.markdown("---")
    st.header("Custom Instructions")
    custom_instructions = st.text_area(
        "Additional Tweaking (Optional)",
        placeholder="Be specific and descriptive. Examples:\n\n"
                    "Color changes:\n"
                    "• Change the dress to deep navy blue, keep the fabric texture\n"
                    "• Make the car a vibrant cherry red\n"
                    "• Change the wall to soft cream, preserve the lighting\n\n"
                    "Lighting & mood:\n"
                    "• Apply warm golden-hour lighting\n"
                    "• Enhance contrast while keeping soft shadows\n\n"
                    "Focus & depth:\n"
                    "• Slightly blur the background for depth",
        height=150,
        help="Add your own instructions to fine-tune the restoration. Be descriptive about what to change and what to preserve."
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

    st.markdown("---")
    st.header("Export")
    export_superres = st.checkbox("Super resolution (x2, under 2K only)", value=False)
    if st.button("Export"):
        if not image_files:
            st.warning("No images to export.")
        else:
            restored_dir = input_path / "restored"
            export_dir = input_path / "ready-for-edit"
            missing_selected = []
            export_jobs = []

            for original in image_files:
                versions = get_restored_versions(restored_dir, original)
                if versions:
                    selected = get_selected_version(versions)
                    if not selected:
                        missing_selected.append(original.name)
                        continue
                    export_jobs.append((selected, original.stem, selected.suffix))
                else:
                    edited_path = get_edited_path(original)
                    source_path = edited_path if edited_path.exists() else original
                    export_jobs.append((source_path, source_path.stem, source_path.suffix))

            if missing_selected:
                st.error(
                    "Missing selected version for: "
                    + ", ".join(missing_selected)
                )
            else:
                sr_impl = None
                if export_superres:
                    sr_impl, sr_error = ensure_superres_ready()
                    if sr_error:
                        st.error(sr_error)
                        st.stop()
                export_dir.mkdir(exist_ok=True)
                errors = []
                skipped_sr = []
                for src, dest_stem, dest_suffix in export_jobs:
                    try:
                        if export_superres:
                            allowed, max_dim = is_upscale_allowed(src)
                            if allowed:
                                dest = export_dir / f"{dest_stem}-sr{dest_suffix}"
                                upscale_with_superres(src, dest, sr_impl)
                            else:
                                dest = export_dir / f"{dest_stem}{dest_suffix}"
                                shutil.copy2(src, dest)
                                label = f"{src.name} ({max_dim}px)" if max_dim else src.name
                                skipped_sr.append(label)
                        else:
                            dest = export_dir / f"{dest_stem}{dest_suffix}"
                            shutil.copy2(src, dest)
                    except Exception as e:
                        errors.append(f"{src.name}: {e}")
                if errors:
                    st.error("Export errors: " + " | ".join(errors))
                else:
                    st.success(f"Exported {len(export_jobs)} photos to {export_dir}")
                    if skipped_sr:
                        st.warning(
                            "Skipped super resolution for >=2K: "
                            + ", ".join(skipped_sr)
                        )

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

    edit_toggle_key = f"edit_toggle_{current_file_path.name}"
    is_editing = st.checkbox("Crop/Rotate", key=edit_toggle_key)

    if is_editing:
        col1 = st.container()
    else:
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

        rotate_key = get_rotate_key(current_file_path)

        if is_editing:
            try:
                base_image = load_source_image(current_file_path)
                rotate_deg = st.slider(
                    "Rotate (degrees, clockwise)",
                    min_value=0,
                    max_value=359,
                    value=st.session_state.get(rotate_key, 0),
                    key=rotate_key
                )
                if rotate_deg:
                    base_image = base_image.rotate(-rotate_deg, expand=True)

                cropped_image = st_cropper(
                    base_image,
                    realtime_update=True,
                    box_color="#ff4b4b",
                    return_type="image"
                )

                apply_col, reset_col = st.columns([1, 1])
                with apply_col:
                    if st.button("Use Edited Image", key=f"apply_edit_{current_file_path.name}"):
                        edited_path = get_edited_path(current_file_path)
                        cropped_image.save(edited_path)
                        st.success(f"Edited image saved: {edited_path.name}")
                with reset_col:
                    if st.button("Reset Edits", key=f"reset_edit_{current_file_path.name}"):
                        edited_path = get_edited_path(current_file_path)
                        if edited_path.exists():
                            edited_path.unlink()
                        st.session_state.pop(rotate_key, None)
                        st.rerun()
            except Exception as e:
                st.error(f"Error editing image: {e}")
        else:
            try:
                image = load_source_image(current_file_path)
                st.image(image, width=600)
            except Exception as e:
                st.error(f"Error loading image: {e}")

    if not is_editing:
        with col2:
            # Session state for version index
            ver_idx_key = f"ver_idx_{current_file_path.name}"
            pending_key = f"pending_ver_{current_file_path.name}"
            generating_key = f"generating_{current_file_path.name}"

            # Check if there's a pending selection (from new generation)
            # Set to index 0 since versions are sorted by mtime (newest first)
            if pending_key in st.session_state:
                st.session_state.pop(pending_key)
                st.session_state[ver_idx_key] = 0

            if ver_idx_key not in st.session_state:
                # Default to selected version if one exists
                selected_idx = 0
                if existing_versions:
                    for i, v in enumerate(existing_versions):
                        if "-selected" in v.stem:
                            selected_idx = i
                            break
                st.session_state[ver_idx_key] = selected_idx

            # Ensure index is valid
            if existing_versions:
                st.session_state[ver_idx_key] = min(st.session_state[ver_idx_key], len(existing_versions) - 1)

            selected_ver_path = existing_versions[st.session_state[ver_idx_key]] if existing_versions else None

            # Header with Generate button and Select button
            if selected_ver_path:
                is_selected = "-selected" in selected_ver_path.stem
                header_col, gen_col, select_col = st.columns([2, 1.5, 1], vertical_alignment="bottom")
                with header_col:
                    title = "Restored ⭐" if is_selected else "Restored"
                    st.subheader(title)
                with gen_col:
                    btn_key = f"create_new_{current_file_path.name}_{len(existing_versions)}"
                    generate_clicked = st.button("Generate 🪄", key=btn_key, type="primary", use_container_width=True)
                with select_col:
                    if not is_selected and len(existing_versions) > 1:
                        if st.button("Select", key=f"select_btn_{selected_ver_path.name}", use_container_width=True):
                            new_name = selected_ver_path.stem + "-selected" + selected_ver_path.suffix
                            new_path = selected_ver_path.parent / new_name
                            selected_ver_path.rename(new_path)
                            st.rerun()
            else:
                header_col, gen_col = st.columns([2.5, 1.5], vertical_alignment="bottom")
                with header_col:
                    st.subheader("Restored")
                with gen_col:
                    btn_key = f"create_new_{current_file_path.name}_{len(existing_versions)}"
                    generate_clicked = st.button("Generate 🪄", key=btn_key, type="primary", use_container_width=True)

            # Status placeholder (fixed height to prevent layout shift)
            status_placeholder = st.empty()

            # Render Selected Image
            if selected_ver_path:
                try:
                    ver_image = Image.open(selected_ver_path)
                    st.image(ver_image, width=600)

                    # Version navigation and filename
                    if len(existing_versions) > 1:
                        prev_col, info_col, next_col = st.columns([1, 3, 1])
                        with prev_col:
                            if st.button("⬅️", key="ver_prev", use_container_width=True):
                                st.session_state[ver_idx_key] = max(0, st.session_state[ver_idx_key] - 1)
                                st.rerun()
                        with info_col:
                            st.caption(f"{selected_ver_path.name} ({st.session_state[ver_idx_key] + 1}/{len(existing_versions)})")
                        with next_col:
                            if st.button("➡️", key="ver_next", use_container_width=True):
                                st.session_state[ver_idx_key] = min(len(existing_versions) - 1, st.session_state[ver_idx_key] + 1)
                                st.rerun()
                    else:
                        st.caption(selected_ver_path.name)
                except Exception as e:
                    st.error(f"Error loading {selected_ver_path.name}: {e}")
            else:
                st.info("No restored versions yet. Click Generate to create one.")

            upscale_source_path = selected_ver_path
            if not upscale_source_path:
                edited_path = get_edited_path(current_file_path)
                upscale_source_path = edited_path if edited_path.exists() else current_file_path

            if st.button("Prepare Upscaled Download (x2, under 2K only)", key=f"prep_upscale_{upscale_source_path.name}"):
                sr_impl, sr_error = ensure_superres_ready()
                if sr_error:
                    st.error(sr_error)
                else:
                    allowed, max_dim = is_upscale_allowed(upscale_source_path)
                    if not allowed:
                        size_label = f"{max_dim}px" if max_dim else "unknown size"
                        st.warning(f"Skipping super resolution (>=2K): {size_label}")
                    else:
                        try:
                            upscale_bytes = upscale_image_to_bytes(upscale_source_path, sr_impl)
                            st.session_state[f"upscale_bytes_{upscale_source_path.name}"] = upscale_bytes
                            st.success("Upscaled image ready.")
                        except Exception as e:
                            st.error(f"Upscale failed: {e}")

            download_key = f"upscale_bytes_{upscale_source_path.name}"
            if download_key in st.session_state:
                download_name = f"{Path(upscale_source_path.name).stem}-sr.png"
                st.download_button(
                    "Download Upscaled Image",
                    data=st.session_state[download_key],
                    file_name=download_name,
                    mime="image/png"
                )

            # Handle generation (button was clicked above)
            if generate_clicked:
                with status_placeholder:
                    with st.spinner("Restoring..."):
                        try:
                            # Load image to check size
                            image_to_process = load_source_image(current_file_path)
                            
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
                                prompt += "2. Keep the original color and tone"

                            prompt += "\n3. Keep the composition or faces."
                            prompt += "\n4. Keep the lightning and mood as original. Enhance contrast while keeping soft shadows."

                            if custom_instructions:
                                prompt += f"\n5. Additional instructions: {custom_instructions}"

                            prompt += "\nOutput ONLY the restored image."
                            
                            response = client.models.generate_content(
                                model=MODEL_NAME,
                                contents=[
                                    prompt,
                                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                                ],
                                config=types.GenerateContentConfig(
                                    response_modalities=['IMAGE'],
                                    image_config=types.ImageConfig(
                                        image_size=output_resolution
                                    ),
                                    temperature=0.4
                                )
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
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                new_filename = f"restored_{current_file_path.name}_v{version_count}_{timestamp}.png"
                                save_path = output_dir / new_filename
                                with open(save_path, "wb") as f:
                                    f.write(restored_data)

                                # Focus on the new version after rerun (index 0 since sorted by mtime desc)
                                st.session_state[pending_key] = True

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
