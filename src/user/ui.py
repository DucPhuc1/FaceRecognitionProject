# app.py

import os
import sys
import time

import cv2
import numpy as np
import streamlit as st
import torch

from src.server.server import full_pipeline_face_detection


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def main():
    # PHẢI gọi set_page_config ngay đầu
    st.set_page_config(page_title="Skin‐Based Face Detection", layout="wide")

    # Thêm src vào path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Import model & frame-processor
    from src.server.server import load_model, process_frame, get_transform_for_inference

    # Load model & transform 1 lần
    @st.cache_resource
    def init(ckpt_path):
        model = load_model(ckpt_path)
        transform = get_transform_for_inference(target_size=(256,256))
        return model, transform

    MODEL_CKPT = os.getenv("UNET_CHECKPOINT", "src/snapshot/best_unet_model.pth")
    model, transform = init(MODEL_CKPT)

    # --- Streamlit UI ---
    st.title("Face Detection Based on Skin Color")

    # Start / Stop buttons
    col_run, col_stop = st.columns(2)
    if 'running' not in st.session_state:
        st.session_state.running = False

    if col_run.button("▶️ Start Video"):
        st.session_state.running = True
    if col_stop.button("⏹ Stop Video"):
        st.session_state.running = False

    # 4 placeholders side-by-side
    col_orig, col_filt, col_cont, col_box = st.columns(4)
    orig_ph = col_orig.empty()
    filt_ph = col_filt.empty()
    cont_ph = col_cont.empty()
    box_ph  = col_box.empty()
    big_ph = st.empty()

    # Nếu đang chạy: mở webcam và loop
    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Không thể mở webcam.")
        else:
            try:
                while st.session_state.running:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Không nhận được frame từ webcam.")
                        break

                    # --- QUAN TRỌNG: unpack đúng 5 giá trị ---
                    orig_rgb, mask, filt_raw, filt_ct, bbox = process_frame(
                        frame, model, device, transform, threshold=0.4
                    )

                    # 1) Original
                    orig_ph.image(orig_rgb, channels="RGB", caption="Original", use_container_width=True)

                    # 2) Filtered onl
                    filt_ph.image(filt_raw, channels="RGB", caption="Filtered Skin",use_container_width=True)

                    # 3) Filtered + Contour
                    cont_ph.image(filt_ct, caption="Filtered + Contour", use_container_width=True)

                    # 4) BBox + Contour
                    box_ph.image(bbox, caption="BBox + Contour", use_container_width=True)

                    face_img, face_count = full_pipeline_face_detection(
                        frame, model, device, transform, method='mtcnn', threshold=0.4
                    )

                    big_ph.image(
                        face_img,
                        channels="RGB",
                        caption=f"Detected faces: {face_count}",
                        use_container_width=True
                    )

                    time.sleep(0.01)
            finally:
                cap.release()
    else:
        st.info("Video đang tạm dừng. Nhấn ▶️ Start Video để bắt đầu.")

if __name__ == "__main__":
    main()
