"""Live demo: upload a photo of a document and get a flat, clean scan.

Run locally:  streamlit run app.py
"""
import cv2
import numpy as np
import streamlit as st

from scan import scan_document

st.set_page_config(page_title="OpenCV Document Scanner", page_icon="📄", layout="wide")


def process(file_bytes, mode, hdr):
    """Decode the uploaded bytes, run the scanner. Returns (original, result) or (original, None)."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    return img, scan_document(img, mode=mode, hdr=hdr)


def to_png_bytes(img):
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


st.title("📄 OpenCV Document Scanner")
st.write(
    "Upload a photo of a paper on a darker surface with all four corners visible. "
    "The app finds the page, corrects the perspective and returns a clean scan. "
    "Built with Python, OpenCV and NumPy."
)

with st.sidebar:
    st.header("Options")
    mode_label = st.radio("Output style", ["Black & white", "Grayscale", "Colour"])
    hdr = st.checkbox("HDR contrast boost", help="Helps when the photo has shadows or uneven light.")
    st.caption("Source code: github.com/ranjithcodein/opencv-document-scanner")

mode = {"Black & white": "bw", "Grayscale": "gray", "Colour": "color"}[mode_label]

uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    original, result = process(uploaded.getvalue(), mode, hdr)
    if original is None:
        st.error("Could not read that file. Please upload a JPG or PNG photo.")
    else:
        left, right = st.columns(2)
        left.subheader("Original")
        left.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), use_container_width=True)
        right.subheader("Scan")
        if result is None:
            right.warning(
                "Could not find a 4-corner document outline. Try a darker background, "
                "keep the whole page in the frame, or turn on the HDR option."
            )
        else:
            shown = result if result.ndim == 3 else cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            if result.ndim == 3:
                shown = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            right.image(shown, use_container_width=True)
            right.download_button("Download scan (PNG)", to_png_bytes(result), "scan.png", "image/png")
else:
    st.info("Waiting for a photo. No sample handy? Try one from the `samples` folder in the repository.")