"""
FoodSaver Vision - College Canteen Food Waste Estimator
=======================================================
A beginner-friendly Streamlit app that uses CLASSICAL image processing
(no AI model) to estimate how much of a plate region differs in colour
from a chosen "clean plate" reference patch.

IMPORTANT: This does NOT measure food weight, calories, or exact food waste.
It only estimates the visible area (in pixels) whose colour differs from
the reference colour.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

# ----------------------------------------------------------------------------
# Settings (constants)
# ----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"

MIN_SIDE = 100                 # images smaller than this (in pixels) are rejected
MAX_SIDE = 1280                # bigger images are shrunk to this for speed
MAX_ORIGINAL_PIXELS = 50_000_000  # reject absurdly large files (50 megapixels)
MIN_PATCH_PIXELS = 25          # reference patch must contain at least this many pixels

# Colours used for drawing (the image is in RGB order, so these are RGB)
COLOR_ELLIPSE = (0, 200, 0)    # green outline = plate region
COLOR_PATCH = (0, 90, 255)     # blue box = reference patch
COLOR_DETECTED = (255, 0, 0)   # red = detected pixels


class ImageProblem(Exception):
    """Raised when the uploaded image cannot be used. The message is shown to the user."""


# ----------------------------------------------------------------------------
# Step 1: Loading the image safely
# ----------------------------------------------------------------------------
def load_image(file_bytes: bytes):
    """
    Read uploaded bytes into an RGB NumPy array.

    Returns:
        rgb          - the image as a NumPy array (height x width x 3, uint8)
        original_size - (width, height) of the uploaded photo before any resizing
    Raises:
        ImageProblem - with a friendly message if the file is not a usable image.
    """
    try:
        pil_image = Image.open(io.BytesIO(file_bytes))

        # Check size BEFORE fully decoding, so huge files don't freeze the app.
        width, height = pil_image.size
        if width * height > MAX_ORIGINAL_PIXELS:
            raise ImageProblem(
                f"The image is too large ({width} x {height} pixels). "
                "Please upload a smaller photo (under 50 megapixels)."
            )

        # Phone photos often store rotation in EXIF data; this fixes the orientation.
        pil_image = ImageOps.exif_transpose(pil_image)

        # Convert to plain RGB (handles PNG transparency, greyscale, etc.)
        pil_image = pil_image.convert("RGB")

    except ImageProblem:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            Image.DecompressionBombError) as error:
        raise ImageProblem(
            "This file could not be read as an image. It may be corrupted or not a "
            "real JPG/PNG file. Please try another photo."
        ) from error

    width, height = pil_image.size
    if min(width, height) < MIN_SIDE:
        raise ImageProblem(
            f"The image is too small ({width} x {height} pixels). "
            f"Both sides must be at least {MIN_SIDE} pixels."
        )

    rgb = np.array(pil_image, dtype=np.uint8)

    # Shrink very large images so the app stays fast.
    longest_side = max(width, height)
    if longest_side > MAX_SIDE:
        scale = MAX_SIDE / longest_side
        new_size = (max(int(width * scale), 1), max(int(height * scale), 1))
        rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)

    return rgb, (width, height)


# ----------------------------------------------------------------------------
# Step 2: Colour conversion (RGB -> CIELAB)
# ----------------------------------------------------------------------------
def rgb_to_lab_float(rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to CIELAB using OpenCV, and rescale to the usual
    CIELAB numbers so that distances are meaningful.

    OpenCV stores 8-bit Lab like this:  L = 0..255,  a = 0..255 (128 = neutral),
    b = 0..255 (128 = neutral).
    We convert to the standard ranges:  L = 0..100, a and b about -128..127.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    return lab


# ----------------------------------------------------------------------------
# Step 3: Plate region (ellipse) and reference patch (rectangle)
# ----------------------------------------------------------------------------
def make_ellipse_geometry(width, height, center_x_pct, center_y_pct, radius_x_pct, radius_y_pct):
    """
    Turn slider percentages into pixel values for the ellipse.

    Returns ((center_x, center_y), (radius_x, radius_y)) in pixels.
    """
    center = (int(width * center_x_pct / 100.0), int(height * center_y_pct / 100.0))
    axes = (max(int(width * radius_x_pct / 100.0), 1),
            max(int(height * radius_y_pct / 100.0), 1))
    return center, axes


def make_plate_mask(width, height, center, axes) -> np.ndarray:
    """Create a mask: 255 inside the ellipse (plate), 0 outside."""
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, thickness=-1)
    return mask


def make_patch_box(width, height, center_x_pct, center_y_pct, size_pct):
    """
    Turn slider percentages into a square reference patch (x0, y0, x1, y1).
    The box is clamped so it never goes outside the image.
    """
    center_x = int(width * center_x_pct / 100.0)
    center_y = int(height * center_y_pct / 100.0)
    half = max(int(min(width, height) * size_pct / 100.0 / 2), 1)

    x0 = int(np.clip(center_x - half, 0, width - 1))
    y0 = int(np.clip(center_y - half, 0, height - 1))
    x1 = int(np.clip(center_x + half, x0 + 1, width))
    y1 = int(np.clip(center_y + half, y0 + 1, height))
    return x0, y0, x1, y1


def patch_is_inside_ellipse(patch_box, center, axes) -> bool:
    """Check whether the middle of the reference patch lies inside the plate ellipse."""
    x0, y0, x1, y1 = patch_box
    px = (x0 + x1) / 2.0
    py = (y0 + y1) / 2.0
    value = ((px - center[0]) / axes[0]) ** 2 + ((py - center[1]) / axes[1]) ** 2
    return value <= 1.0


# ----------------------------------------------------------------------------
# Step 4: Reference colour and colour distance
# ----------------------------------------------------------------------------
def compute_reference_color(lab: np.ndarray, patch_box):
    """
    Calculate the MEDIAN Lab colour of the patch (median ignores small stray spots).

    Returns:
        reference  - array of 3 numbers [L, a, b]
        spread     - median distance of patch pixels from the reference
                     (a small number means the patch is a nice uniform colour)
        pixel_count - number of pixels in the patch
    """
    x0, y0, x1, y1 = patch_box
    patch_pixels = lab[y0:y1, x0:x1].reshape(-1, 3)
    reference = np.median(patch_pixels, axis=0)
    distances = np.sqrt(np.sum((patch_pixels - reference) ** 2, axis=1))
    return reference, float(np.median(distances)), int(patch_pixels.shape[0])


def compute_color_distance(lab: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    For EVERY pixel, calculate the Euclidean distance in Lab space to the
    reference colour (this is the "CIE76 Delta E" colour difference):

        distance = sqrt( (L - L_ref)^2 + (a - a_ref)^2 + (b - b_ref)^2 )

    Small distance = similar to clean plate. Large distance = different colour.
    """
    difference = lab - reference.reshape(1, 1, 3)
    return np.sqrt(np.sum(difference ** 2, axis=2))


# ----------------------------------------------------------------------------
# Step 5: Detection mask (threshold + plate restriction + noise cleanup)
# ----------------------------------------------------------------------------
def build_detection_mask(distance, plate_mask, threshold, use_cleanup, kernel_size) -> np.ndarray:
    """
    Make the binary mask of "pixels that differ from the clean-plate colour".

    1. Threshold: keep pixels whose colour distance is above the threshold.
    2. Plate restriction: keep only pixels inside the ellipse.
    3. Optional cleanup:
         - Opening  (erode then dilate) removes tiny isolated specks.
         - Closing  (dilate then erode) fills tiny holes inside detected areas.
    Result: 255 = detected, 0 = not detected.
    """
    mask = ((distance > threshold) & (plate_mask > 0)).astype(np.uint8) * 255

    if use_cleanup and kernel_size >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Closing can slightly grow the mask past the ellipse edge, so clip it again.
        mask = cv2.bitwise_and(mask, plate_mask)

    return mask


# ----------------------------------------------------------------------------
# Step 6: Drawing the overlay images
# ----------------------------------------------------------------------------
def draw_guides(rgb, center, axes, patch_box) -> np.ndarray:
    """Draw the green plate ellipse and the blue reference patch box on a copy of the image."""
    guide_image = rgb.copy()
    thickness = max(2, min(rgb.shape[:2]) // 300)
    cv2.ellipse(guide_image, center, axes, 0, 0, 360, COLOR_ELLIPSE, thickness)
    x0, y0, x1, y1 = patch_box
    cv2.rectangle(guide_image, (x0, y0), (x1 - 1, y1 - 1), COLOR_PATCH, thickness)
    return guide_image


def make_overlay(rgb, mask, center, axes, patch_box, alpha=0.55) -> np.ndarray:
    """Colour the detected pixels red (semi-transparent) and add the guide shapes."""
    overlay = rgb.copy()
    selected = mask > 0
    red = np.array(COLOR_DETECTED, dtype=np.float32)
    blended = rgb[selected].astype(np.float32) * (1.0 - alpha) + red * alpha
    overlay[selected] = blended.astype(np.uint8)
    return draw_guides(overlay, center, axes, patch_box)


def image_to_png_bytes(rgb: np.ndarray) -> bytes:
    """Convert an RGB array into PNG bytes (used for the download button)."""
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


# ----------------------------------------------------------------------------
# The Streamlit user interface
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="FoodSaver Vision", page_icon="🍽️", layout="wide")

    st.title("🍽️ FoodSaver Vision")
    st.subheader("College Canteen Food Waste Estimator (classical colour segmentation demo)")
    st.caption(
        "This is a simple image-processing demonstration, NOT an AI food-recognition model. "
        "It does not measure food weight, calories, or exact food waste. It only estimates "
        "the visible area of pixels whose colour differs from a clean-plate reference colour."
    )

    # ----- Sidebar controls -----
    st.sidebar.header("Controls")

    st.sidebar.subheader("1. Plate region (ellipse)")
    ellipse_cx = st.sidebar.slider("Centre X (%)", 0, 100, 50)
    ellipse_cy = st.sidebar.slider("Centre Y (%)", 0, 100, 50)
    ellipse_rx = st.sidebar.slider("Half-width (% of image width)", 5, 60, 45)
    ellipse_ry = st.sidebar.slider("Half-height (% of image height)", 5, 60, 45)

    st.sidebar.subheader("2. Clean plate reference patch")
    patch_cx = st.sidebar.slider("Patch centre X (%)", 0, 100, 50)
    patch_cy = st.sidebar.slider("Patch centre Y (%)", 0, 100, 50)
    patch_size = st.sidebar.slider("Patch size (% of shorter image side)", 2, 20, 6)

    st.sidebar.subheader("3. Detection settings")
    threshold = st.sidebar.slider(
        "Colour distance threshold", 5, 80, 20,
        help="Lower = more sensitive (detects smaller colour differences). "
             "Higher = stricter.")
    use_cleanup = st.sidebar.checkbox("Reduce noise (morphological opening + closing)", value=True)
    kernel_size = st.sidebar.slider("Cleanup kernel size (pixels)", 3, 15, 5, step=2,
                                    disabled=not use_cleanup)

    # ----- Upload -----
    uploaded = st.file_uploader("Upload a photograph of a plate (JPG, JPEG or PNG)",
                                type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload a photo to begin. Best results: photo taken from above, "
                "plain light-coloured plate, even lighting, no strong shadows or reflections.")
        return

    # ----- Load and validate image -----
    try:
        rgb, original_size = load_image(uploaded.getvalue())
    except ImageProblem as problem:
        st.error(str(problem))
        return
    except Exception:  # last safety net so the app never crashes on a strange file
        st.error("Something went wrong while opening the image. Please try a different photo.")
        return

    height, width = rgb.shape[:2]

    st.markdown("### Uploaded image")
    st.write(f"**Uploaded size:** {original_size[0]} x {original_size[1]} pixels")
    if (width, height) != original_size:
        st.write(f"**Processed size (shrunk for speed):** {width} x {height} pixels")
    st.image(rgb, caption="Original image")

    # ----- Set up geometry -----
    center, axes = make_ellipse_geometry(width, height, ellipse_cx, ellipse_cy,
                                         ellipse_rx, ellipse_ry)
    patch_box = make_patch_box(width, height, patch_cx, patch_cy, patch_size)

    st.markdown("### Setup preview")
    st.write("Move the sliders in the sidebar until the **green ellipse** fits the plate "
             "and the **blue square** sits on a clean part of the plate.")
    st.image(draw_guides(rgb, center, axes, patch_box),
             caption="Green = plate region, Blue = clean plate reference patch")

    # ----- Check the reference patch -----
    lab = rgb_to_lab_float(rgb)
    reference, spread, patch_pixel_count = compute_reference_color(lab, patch_box)

    if patch_pixel_count < MIN_PATCH_PIXELS:
        st.error(f"The reference patch is too small ({patch_pixel_count} pixels). "
                 "Increase the patch size.")
        return
    if not patch_is_inside_ellipse(patch_box, center, axes):
        st.warning("The reference patch is outside the plate ellipse. "
                   "Move it onto the plate.")
    if spread > 8:
        st.warning("The reference patch does not look like one uniform colour "
                   "(it may contain food, a shadow or a pattern). "
                   "Try moving it to a cleaner area.")

    # ----- Run the detection -----
    plate_mask = make_plate_mask(width, height, center, axes)
    plate_pixels = int(np.count_nonzero(plate_mask))
    if plate_pixels == 0:
        st.error("The plate region is empty. Adjust the ellipse sliders.")
        return

    distance = compute_color_distance(lab, reference)
    detection_mask = build_detection_mask(distance, plate_mask, threshold,
                                          use_cleanup, kernel_size)
    detected_pixels = int(np.count_nonzero(detection_mask))
    percentage = 100.0 * detected_pixels / plate_pixels
    overlay = make_overlay(rgb, detection_mask, center, axes, patch_box)

    # ----- Show results -----
    st.markdown("### Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(rgb, caption="Original image")
    with col2:
        st.image(detection_mask, caption="Binary detection mask (white = detected)")
    with col3:
        st.image(overlay, caption="Overlay (red = pixels differing from clean plate)")

    m1, m2, m3 = st.columns(3)
    m1.metric("Detected pixels", f"{detected_pixels:,}")
    m2.metric("Pixels in plate region", f"{plate_pixels:,}")
    m3.metric("Estimated visible food area (%)", f"{percentage:.2f}%")

    st.caption(
        f"Reference colour (Lab): L={reference[0]:.1f}, a={reference[1]:.1f}, "
        f"b={reference[2]:.1f}.  Pixels counted are from the processed image."
    )
    st.warning(
        "Limitations: shadows, reflections, patterned plates, and food with a colour "
        "similar to the plate can give wrong results. This value is a visible-area "
        "estimate only, not food weight, calories, or exact food waste."
    )

    # ----- Download / save -----
    png_bytes = image_to_png_bytes(overlay)
    st.download_button("⬇️ Download overlay image (PNG)", data=png_bytes,
                       file_name="foodsaver_overlay.png", mime="image/png")

    if st.button("Save overlay into the outputs/ folder"):
        try:
            OUTPUT_DIR.mkdir(exist_ok=True)
            file_name = f"overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            (OUTPUT_DIR / file_name).write_bytes(png_bytes)
            st.success(f"Saved: outputs/{file_name}")
        except OSError as error:
            st.error(f"Could not save the file: {error}")


if __name__ == "__main__":
    main()