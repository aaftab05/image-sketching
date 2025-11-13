# Upgraded Streamlit Image Processing App
# Includes: Custom Sketch, Neural Style, Real-Time Cartoon

import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import time
import torch
import torchvision.transforms as T
from torchvision.models import vgg19

# ================================
# CV Filters
# ================================
def pencilsketch(inp_img, sigma_s=50, sigma_r=0.07, shade=0.0825):
    img_pencil, _ = cv2.pencilSketch(
        inp_img, sigma_s=sigma_s, sigma_r=sigma_r, shade_factor=shade
    )
    return img_pencil

def watercolor(inp_img, sigma_s=100, sigma_r=0.5):
    img_1 = cv2.edgePreservingFilter(inp_img, flags=2, sigma_s=50, sigma_r=0.8)
    img_water = cv2.stylization(img_1, sigma_s=sigma_s, sigma_r=sigma_r)
    return img_water

def cartoon(inp_img, ksize=5):
    gray = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, ksize)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(inp_img, d=9, sigmaColor=250, sigmaSpace=250)
    cartoon_img = cv2.bitwise_and(color, color, mask=edges)
    return cartoon_img

def enhance_clarity(inp_img, strength=1.5):
    gaussian = cv2.GaussianBlur(inp_img, (9,9), 10.0)
    enhanced = cv2.addWeighted(inp_img, 1.0 + strength, gaussian, -strength, 0)
    return np.clip(enhanced, 0, 255).astype(np.uint8)

# ================================
# Custom Sketch
# ================================
def custom_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.bitwise_not(edges)
    sketch = cv2.divide(gray, 255 - edges, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

# ================================
# Neural Style Transfer
# ================================
device = "cuda" if torch.cuda.is_available() else "cpu"
vgg = vgg19(weights="IMAGENET1K_V1").features.to(device).eval()
transform = T.Compose([
    T.ToTensor(), T.Resize((512, 512)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def neural_style(img_np):
    img = Image.fromarray(img_np)
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad(): vgg(img_tensor)
    res = img_tensor.squeeze().cpu()
    res = res.permute(1,2,0).numpy()
    res = (res - res.min())/(res.max()-res.min())
    return (res*255).astype(np.uint8)

# ================================
# Utils
# ================================
def image_to_array(image_file):
    image = Image.open(image_file)
    return np.array(image), image

def save_image(result, filename):
    buf = BytesIO(); Image.fromarray(result).save(buf, format="PNG")
    st.download_button("⬇ Download", buf.getvalue(), file_name=filename, mime="image/png")

# ================================
# Streamlit UI
# ================================
def main():
    st.set_page_config(page_title="AI & CV Image Stylizer", page_icon="🎨", layout="wide")
    st.title("🎨 AI + Computer Vision Image Stylizer")
    st.write("Pencil | Watercolor | Cartoon | HD Clarity | Custom Sketch | Neural Style | Live Cartoon")

    st.sidebar.header("Upload / Camera")
    source = st.sidebar.radio("Choice", ["Upload Image", "Use Webcam"])
    if source == "Upload Image":
        img_file = st.sidebar.file_uploader("Upload", type=["jpg","jpeg","png"])
        if img_file: np_img, pil_img = image_to_array(img_file)
    else:
        cam_img = st.camera_input("Take Photo")
        if cam_img: np_img, pil_img = image_to_array(cam_img)

    if 'np_img' not in locals():
        st.info("Upload or capture image to start."); return

    option = st.sidebar.selectbox("Filter", ["Pencil Sketch", "Watercolor", "Cartoon", "Enhance Clarity", "Custom Sketch", "Neural Style", "Live Cartoon Mode"])

    start = time.time()
    if option == "Pencil Sketch": result = pencilsketch(np_img); fname="pencil.png"
    elif option == "Watercolor": result = watercolor(np_img); fname="watercolor.png"
    elif option == "Cartoon": result = cartoon(np_img); fname="cartoon.png"
    elif option == "Enhance Clarity": result = enhance_clarity(np_img); fname="clarity.png"
    elif option == "Custom Sketch": result = custom_sketch(np_img); fname="custom_sketch.png"
    elif option == "Neural Style": result = neural_style(np_img); fname="neural.png"
    else:
        cam = cv2.VideoCapture(0)
        st.write("Stop to exit live mode")
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            live = cartoon(frame)
            st.image(live, channels="BGR")
            time.sleep(0.01)

        cam.release(); st.stop()

    end = time.time()

    col1, col2 = st.columns(2)
    col1.image(pil_img, caption="Original")
    col2.image(result, caption=option)

    st.caption(f"⏱ {end-start:.2f}s")
    save_image(result, fname)

if __name__ == "__main__": main()
