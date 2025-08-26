import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.model.unet import UNet
from dataset.data_preprocessing import get_transforms

# Thiết lập device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path):
    print(">>> Loading model from disk")

    model = UNet(in_channels=3, out_channels=1, init_features=32, dropout_prob=0.3).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

def get_transform_for_inference(target_size=(256, 256)):
    return get_transforms(target_size=target_size, is_train=False, for_visualization=False)

def denormalize(image_tensor):
    """
    Chuyển đổi ảnh đã được normalize về dạng ban đầu để hiển thị đúng màu sắc.
    Giả sử ảnh đã được normalize theo mean=[0.485,0.456,0.406] và std=[0.229,0.224,0.225].
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
    image_np = image_np * std + mean  # Denormalize
    image_np = np.clip(image_np, 0, 1)
    return image_np

def get_skin_mask(image, model, device, transform, threshold=0.4):
    """
    Lấy skin mask từ ảnh dựa trên model segmentation.

    Args:
        image_path (str): Đường dẫn đến ảnh gốc.
        model: Model segmentation dùng để dự đoán mask.
        device: Thiết bị thực hiện infer (CPU hoặc GPU).
        transform: Hàm chuyển đổi ảnh (ví dụ: từ Albumentations) trả về dict với key 'image'.
        threshold (float): Ngưỡng phân loại để chuyển mask dự đoán thành nhị phân.

    Returns:
        pred_mask_np (numpy.ndarray): Mảng mask nhị phân có giá trị 0 hoặc 1, dạng numpy array.
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. Áp dụng transform để chuẩn bị đầu vào cho model
    augmented = transform(image=image)
    image_tensor = augmented['image']  # expected shape: [C, H, W]
    print("DEBUG: image_tensor shape:", image_tensor.shape)
    print("DEBUG: image_tensor dtype:", image_tensor.dtype)

    # 3. Thêm batch dimension và đưa vào device
    input_tensor = image_tensor.unsqueeze(0).to(device)

    # 4. Dự đoán mask bằng model và đưa về CPU
    model.eval()
    with torch.no_grad():
        pred_mask = model(input_tensor).cpu()

    # 5. Loại bỏ các dimension có kích thước 1 (batch, channel nếu có)
    pred_mask = pred_mask.squeeze()  # Nếu shape ban đầu là (1, 256, 256) hoặc (1,1,256,256) thì sẽ trở về (256,256)
    print("DEBUG: pred_mask shape sau squeeze:", pred_mask.shape)
    print("DEBUG: pred_mask dtype:", pred_mask.dtype)

    # 6. Áp dụng threshold để biến mask thành nhị phân (0 và 1)
    # Kết quả sau khi .numpy() đã là NumPy array
    pred_mask_np = (pred_mask > threshold).float().numpy()
    print("DEBUG: Unique values sau threshold (NumPy):", np.unique(pred_mask_np))
    pred_mask_np = (pred_mask_np * 255).astype(np.uint8)
    return pred_mask_np

def draw_contours_on_mask(mask):
    """
    mask: Ảnh nhị phân [H, W] có giá trị 0 hoặc 255
    return: Ảnh mask 3 kênh có vẽ contour (màu xanh dương)
    """
    # Đảm bảo mask có kiểu uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # Áp dụng threshold (nếu cần, đảm bảo chỉ có 0 hoặc 255)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # Chuyển mask sang ảnh 3 kênh
    mask_color = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
    # Tìm contours
    contours, _ = cv2.findContours(mask_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Vẽ contour lên ảnh (màu xanh dương: BGR (255, 0, 0))
    cv2.drawContours(mask_color, contours, -1, (255, 0, 0), 2)
    # Chuyển đổi sang RGB để hiển thị đúng với matplotlib
    mask_color_rgb = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)
    return mask_color_rgb

def draw_bbox_on_image(image, mask):
    """
    image: Ảnh RGB, float32 [0,1]
    mask: Ảnh nhị phân [0 hoặc 255]
    return: Ảnh RGB với bbox và contour (vẫn đúng màu)
    """
    image_disp = (image * 255).astype(np.uint8).copy()

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image_disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(image_disp, [cnt], -1, (0, 0, 255), 2)

    image_with_box = image_disp.astype(np.float32) / 255.0
    return image_with_box

def process_frame(frame, model, device, transform, threshold=0.4):
    """
    Input:
      - frame: uint8 BGR [H,W,3] từ camera
      - transform: Albumentations transform -> {'image': Tensor}
    Output (5-tuple):
      - img_disp_rgb: uint8 RGB [256,256,3]
      - mask_bin: uint8 0/255 [256,256]
      - filtered_raw_bgr: uint8 BGR [256,256,3]
      - filtered_with_contour_bgr: uint8 BGR [256,256,3]
      - bbox_img_bgr: uint8 BGR [256,256,3]
    """
    print(">>> Running inference for this frame")
    # 1. Chuyển BGR->RGB và resize

    skin_mask = get_skin_mask(frame, model, device, transform, threshold=threshold)

    orig_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image_resized = cv2.resize(orig_image, (256, 256), interpolation=cv2.INTER_LINEAR)
    image_resized = image_resized.astype(np.float32) / 255.0  # chuyển về float [0,1]

    mask_expanded = (skin_mask.astype(np.float32) / 255.0)[..., np.newaxis]
    filtered_image = image_resized * mask_expanded

    contour_img = draw_contours_on_mask(skin_mask)
    image_with_bbox = draw_bbox_on_image(image_resized, skin_mask)

    return (
        image_resized,
        skin_mask,
        filtered_image,
        contour_img,
        image_with_bbox,
    )

def detect_face_mtcnn_from_skin(frame, skin_mask, min_face_size=20):
    orig_image = frame.copy()
    orig_image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    h, w = orig_image_rgb.shape[:2]
    if skin_mask.shape != (h, w):
        skin_mask = cv2.resize(skin_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Áp mặt nạ da lên ảnh RGB (lọc vùng không phải da)
    filtered_image = orig_image_rgb.copy()
    filtered_image[skin_mask == 0] = [255, 255, 255]  # hoặc [0, 0, 0] nếu thích nền đen

    filtered_image_pil = Image.fromarray(filtered_image)

    # Tạo MTCNN detector
    mtcnn = MTCNN(keep_all=True, min_face_size=min_face_size)

    # Phát hiện khuôn mặt và landmark
    boxes, probs, landmarks = mtcnn.detect(filtered_image_pil, landmarks=True)

    result_image = orig_image_rgb.copy()

    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            prob = probs[i]
            print(f"[{i}] Box: {box}, Confidence: {prob:.4f}")

            # Vẽ bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{prob:.2f}"
            cv2.putText(result_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Vẽ landmark (mắt trái, mắt phải, mũi, miệng trái, miệng phải)
            for (x, y) in landmarks[i]:
                cv2.circle(result_image, (int(x), int(y)), 2, (0, 0, 255), -1)
    else:
        print("Không phát hiện được khuôn mặt bằng MTCNN.")

    return result_image, len(boxes) if boxes is not None else 0

def full_pipeline_face_detection(frame, model, device, transform, method='mtcnn', threshold=0.4):
    """
    Quy trình đầy đủ để phát hiện khuôn mặt:
    1. Dự đoán vùng da
    2. Áp dụng bộ phát hiện khuôn mặt
    3. Hiển thị kết quả
    """
    # 1. Dự đoán vùng da
    skin_mask = get_skin_mask(frame, model, device, transform, threshold=threshold)

    # 2. Áp dụng bộ phát hiện khuôn mặt
    if method == 'mtcnn':
        result_image, count = detect_face_mtcnn_from_skin(frame, skin_mask)
        print("Số lượng khuôn mặt phát hiện được: ", count)
    else:
        raise ValueError("Phương pháp không được hỗ trợ. Sử dụng 'mtcnn'.")
    return result_image, count
