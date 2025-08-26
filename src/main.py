import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings

warnings.filterwarnings("ignore", message=".*iCCP: known incorrect sRGB profile.*")

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as T

# Giả sử bạn đã có mô hình UNet được định nghĩa trong model/unet.py
from model.unet import UNet
# Giả sử hàm get_transforms đã được định nghĩa trong dataset/data_preprocessing.py
from dataset.data_preprocessing import get_transforms
from mtcnn import MTCNN

# ---------------------------
# 1. Hàm dự đoán skin mask từ ảnh sử dụng UNet (kích thước mask: 256x256)
# ---------------------------
def get_skin_mask(image_path, model, device, transform):
    """
    Lấy skin mask (nhị phân với giá trị 0 hoặc 255) từ ảnh dùng mô hình UNet.
    Args:
        image_path (str): Đường dẫn ảnh cần test.
        model (torch.nn.Module): Mô hình UNet đã huấn luyện.
        device (torch.device): CPU hoặc GPU.
        transform (callable): Transform để tiền xử lý ảnh (ví dụ: từ Albumentations).
    Returns:
        skin_mask (np.ndarray): Ảnh mask nhị phân kích thước 256x256.
    """
    # Đọc ảnh và chuyển về RGB
    img = Image.open(image_path).convert("RGB")
    # Áp dụng transform
    if transform is None:
        default_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])
        img_tensor = default_transform(img)
    else:
        augmented = transform(image=np.array(img))
        img_tensor = augmented['image']
        if not isinstance(img_tensor, torch.Tensor):
            img_tensor = torch.tensor(img_tensor).permute(2, 0, 1).float() / 255.0

    # Thêm chiều batch: [1, C, H, W]
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Dự đoán skin mask
    model.eval()
    with torch.no_grad():
        pred = model(img_tensor)
    pred_np = pred.squeeze().cpu().numpy()

    # Thử nghiệm giảm ngưỡng từ 0.5 xuống 0.4 nếu cần
    skin_mask = (pred_np > 0.4).astype(np.uint8) * 255
    return skin_mask


# ---------------------------
# 2. Hàm trích xuất ROI (bounding box) từ skin mask
# ---------------------------
def extract_skin_ROIs(skin_mask, min_area=200, aspect_ratio_range=(0.3, 3.0)):
    """
    Từ skin mask nhị phân, trích xuất các bounding box của các vùng liên thông.
    Args:
        skin_mask (np.ndarray): Ảnh mask nhị phân kích thước 256x256.
        min_area (int): Diện tích nhỏ nhất của contour.
        aspect_ratio_range (tuple): Khoảng cho tỉ lệ width/height.
    Returns:
        List[tuple]: Danh sách các bounding box dưới dạng (x, y, w, h) trên ảnh 256x256.
    """
    # Tìm contours
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Found total contours:", len(contours))
    rois = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h != 0 else 9999
        print(f"Contour {i}: area={area:.2f}, bbox=({x}, {y}, {w}, {h}), aspect={aspect_ratio:.2f}")
        if area < min_area:
            continue
        if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
            continue
        rois.append((x, y, w, h))
    print("Extracted ROI:", rois)
    return rois


# ---------------------------
# 3. Hàm kiểm tra khuôn mặt trong mỗi ROI sử dụng MTCNN pretrained
# ---------------------------
def detect_face_in_ROI(image, roi):
    """
    Cắt ROI từ ảnh và dùng MTCNN (embedding) để kiểm tra xem có khuôn mặt không.
    Args:
        image (np.ndarray): Ảnh gốc (RGB).
        roi (tuple): Bounding box (x, y, w, h) trên ảnh gốc.
    Returns:
        (bool, list): True nếu phát hiện khuôn mặt trong ROI, kèm theo thông tin dự đoán.
    """
    detector = MTCNN()
    x, y, w, h = roi
    roi_img = image[y:y + h, x:x + w]
    results = detector.detect_faces(roi_img)
    return (len(results) > 0, results)


# ---------------------------
# 4. Hàm tích hợp pipeline: từ skin segmentation -> ROI -> Face detection
# ---------------------------
def detect_faces_from_skin_segmentation(image_path, model, device, transform):
    """
    Quy trình: từ skin segmentation qua UNet (256x256) -> chuyển ROI sang ảnh gốc -> kiểm tra khuôn mặt bằng MTCNN -> vẽ bounding box.
    Args:
        image_path (str): Đường dẫn ảnh test.
        model (torch.nn.Module): Mô hình UNet đã huấn luyện.
        device (torch.device): CPU hoặc GPU.
        transform: Transform tiền xử lý ảnh.
    """
    # 1. Dự đoán skin mask (256x256)
    skin_mask = get_skin_mask(image_path, model, device, transform)
    plt.figure(figsize=(6, 6))
    plt.imshow(skin_mask, cmap='gray')
    plt.title("Skin Mask (256x256)")
    plt.axis("off")
    plt.show()
    print("Unique values in skin mask:", np.unique(skin_mask))

    # 2. Trích xuất ROI trên ảnh 256x256
    rois_resized = extract_skin_ROIs(skin_mask, min_area=200, aspect_ratio_range=(0.3, 3.0))
    print("Number of ROI extracted (256x256):", len(rois_resized))

    # 3. Đọc ảnh gốc và lấy kích thước ban đầu
    img_bgr = cv2.imread(image_path)
    orig_h, orig_w = img_bgr.shape[:2]

    # 4. Tính tỉ lệ chuyển đổi tọa độ từ ảnh 256x256 sang ảnh gốc
    scale_x = orig_w / 256.0
    scale_y = orig_h / 256.0
    adjusted_rois = []
    for (x, y, w, h) in rois_resized:
        x_adj = int(x * scale_x)
        y_adj = int(y * scale_y)
        w_adj = int(w * scale_x)
        h_adj = int(h * scale_y)
        adjusted_rois.append((x_adj, y_adj, w_adj, h_adj))
    print("Adjusted ROI in original image:", adjusted_rois)

    # 5. Chuyển đổi ảnh gốc từ BGR sang RGB (cho MTCNN)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 6. Kiểm tra từng ROI bằng MTCNN
    final_faces = []
    for roi in adjusted_rois:
        found, faces = detect_face_in_ROI(img_rgb, roi)
        print("ROI:", roi, "=> Face detected:", found, "with", len(faces), "face(s)")
        # Nếu có khuôn mặt được phát hiện, lưu ROI vào danh sách final_faces
        if found:
            final_faces.append(roi)

    print("Final ROIs with detected faces:", final_faces)
    # 7. Vẽ bounding box lên ảnh gốc
    for (x, y, w, h) in final_faces:
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title("Detected Faces on Original Image")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Thiết lập thiết bị chạy (GPU hoặc CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Khởi tạo mô hình UNet (giả sử đã được định nghĩa trong model/unet.py)
    model = UNet(in_channels=3, out_channels=1).to(device)
    # Tải trọng số đã huấn luyện (đảm bảo file tồn tại và đúng đường dẫn)
    model.load_state_dict(torch.load("../unet_skin_segmentation.pth", map_location=device))

    transform = get_transforms(target_size=(256, 256))

    # Đường dẫn tới ảnh cần test (đảm bảo ảnh có chứa khuôn mặt với vùng da rõ ràng)
    test_image_path = '../dataset/test/test.png'

    # Chạy pipeline phát hiện khuôn mặt từ skin segmentation
    detect_faces_from_skin_segmentation(test_image_path, model, device, transform)
