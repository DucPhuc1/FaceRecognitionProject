import matplotlib.pyplot as plt
import torch

import warnings
warnings.filterwarnings("ignore", message=".*iCCP: known incorrect sRGB profile.*")

import numpy as np

def show_batch(images, masks):
    """
    Hiển thị một batch ảnh và mask.
    """
    batch_size = images.size(0)
    images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
    masks_np = masks.cpu().numpy().squeeze(1)

    fig, axs = plt.subplots(batch_size, 2, figsize=(8, 2 * batch_size))
    for i in range(batch_size):
        axs[i, 0].imshow(images_np[i])
        axs[i, 0].set_title("Image")
        axs[i, 0].axis("off")
        axs[i, 1].imshow(masks_np[i], cmap='gray')
        axs[i, 1].set_title("Mask")
        axs[i, 1].axis("off")
    plt.tight_layout()
    plt.show()


def visualize_prediction(model, test_loader, device, sample_idx=0, apply_denormalization=False):
    """
    Hiển thị Test Image, Ground Truth Mask, Predicted Mask và Filtered Image.

    Parameters:
      - model: mô hình đã huấn luyện (UNet)
      - test_loader: DataLoader của tập test (lấy batch đầu tiên)
      - device: thiết bị chạy (cpu hoặc cuda)
      - sample_idx: chỉ số mẫu trong batch (phải nằm trong khoảng [0, batch_size-1])
      - apply_denormalization: nếu True, thực hiện denormalize (cho dữ liệu đã normalize)
    """
    model.eval()
    with torch.no_grad():
        for images, masks in test_loader:
            # Ép kiểu ảnh thành float nếu chưa, chuyển về device
            images = images.float().to(device)
            break  # Lấy batch đầu tiên

        # Kiểm tra sample_idx
        if sample_idx < 0 or sample_idx >= images.size(0):
            raise IndexError(f"sample_idx {sample_idx} không hợp lệ với batch size {images.size(0)}")

        orig_image = images[sample_idx].cpu()  # [C, H, W]
        gt_mask = masks[sample_idx].cpu()      # [1, H, W]

        # Đưa ảnh vào mô hình
        input_image = orig_image.unsqueeze(0).float().to(device)
        pred_mask = model(input_image).squeeze(0).cpu()  # [1, H, W]

    # Chuyển đổi ảnh từ tensor sang numpy [H, W, C]
    image_np = orig_image.permute(1, 2, 0).numpy()
    if apply_denormalization:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean  # Denormalize
    image_np = np.clip(image_np, 0, 1)

    # Chuyển đổi mask sang numpy
    gt_mask_np = gt_mask.squeeze(0).numpy()
    pred_mask_np = pred_mask.squeeze(0).numpy()  # [H, W]

    # Lấy phần vùng ảnh dựa trên predicted mask
    filtered_image = image_np * pred_mask_np[..., None]

    # Hiển thị kết quả
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(image_np)
    axs[0].set_title("Test Image")
    axs[0].axis("off")

    axs[1].imshow(gt_mask_np, cmap="gray")
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis("off")

    axs[2].imshow(pred_mask_np, cmap="gray")
    axs[2].set_title("Predicted Mask")
    axs[2].axis("off")

    axs[3].imshow(filtered_image)
    axs[3].set_title("Filtered Image")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()