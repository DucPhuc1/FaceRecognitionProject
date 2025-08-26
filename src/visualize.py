import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import torch
from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Project root directory: {project_root}")

src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
print(f"src directory: {src_path}")

# Import các hàm tiền xử lý và dataset từ file data_preprocessing.py
from dataset.data_preprocessing import get_dataset, get_transforms
# Import mô hình UNet cải tiến
from model.unet import UNet

from src.utils.visualization import visualize_prediction


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize prediction for chosen dataset")
    parser.add_argument("--dataset_mode", type=str, required=True,
                        choices=["pratheepan", "celeba"],
                        help="Chọn dataset: 'pratheepan' hoặc 'celeba'")
    parser.add_argument("--target_size", type=int, nargs=2, default=[256, 256],
                        help="Kích thước ảnh sau khi resize, ví dụ: 256 256")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size cho DataLoader")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Chỉ số mẫu trong batch để visualize")
    parser.add_argument("--is_train", action="store_true",
                        help="Áp dụng augment cho training (thường không cần cho visualize)")
    # Các tham số cho Pratheepan Dataset
    parser.add_argument("--face_img_folder", type=str, default="dataset/Face_Dataset/Pratheepan_Dataset/FacePhoto",
                        help="Thư mục chứa ảnh FacePhoto (Pratheepan)")
    parser.add_argument("--face_mask_folder", type=str, default="dataset/Face_Dataset/Ground_Truth/GroundT_FacePhoto",
                        help="Thư mục chứa mask FacePhoto (Pratheepan)")
    parser.add_argument("--family_img_folder", type=str, default="dataset/Face_Dataset/Pratheepan_Dataset/FamilyPhoto",
                        help="Thư mục chứa ảnh FamilyPhoto (Pratheepan)")
    parser.add_argument("--family_mask_folder", type=str,
                        default="dataset/Face_Dataset/Ground_Truth/GroundT_FamilyPhoto",
                        help="Thư mục chứa mask FamilyPhoto (Pratheepan)")
    # Các tham số cho CelebAMask HQ Dataset
    parser.add_argument("--imgs_folder", type=str, default="dataset/preprocessed/imgs",
                        help="Thư mục chứa ảnh (CelebAMask HQ)")
    parser.add_argument("--masks_folder", type=str, default="dataset/preprocessed/masks",
                        help="Thư mục chứa mask (CelebAMask HQ)")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Sử dụng transform dành cho hiển thị: for_visualization=True (không augment, chỉ resize)
    transform = get_transforms(target_size=tuple(args.target_size), is_train=True, for_visualization=False)

    # Khởi tạo dataset dựa trên lựa chọn của người dùng
    if args.dataset_mode.lower() == "pratheepan":
        dataset = get_dataset(
            dataset_mode="pratheepan",
            transform=transform,
            face_img_folder=args.face_img_folder,
            face_mask_folder=args.face_mask_folder,
            family_img_folder=args.family_img_folder,
            family_mask_folder=args.family_mask_folder
        )
    elif args.dataset_mode.lower() == "celeba":
        dataset = get_dataset(
            dataset_mode="celeba",
            transform=transform,
            imgs_folder=args.imgs_folder,
            masks_folder=args.masks_folder
        )
    else:
        raise ValueError("dataset_mode không hợp lệ!")

    print(f"Tổng số samples trong dataset: {len(dataset)}")

    # Tạo DataLoader (nếu cần cho các bước khác, nhưng visualize_prediction dùng trực tiếp dataset)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Khởi tạo mô hình U-Net và tải trọng số đã huấn luyện từ checkpoint trong folder snapshot
    model = UNet(in_channels=3, out_channels=1, init_features=32, dropout_prob=0.3).to(device)

    checkpoint_path = "src/snapshot/best_unet_model.pth"
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    else:
        print(f"Model checkpoint not found at: {checkpoint_path}")
        exit()

    # Chọn mẫu cần visualize (index từ 0 đến len(dataset)-1)
    sample_idx = args.sample_idx
    print("Sample index:", sample_idx)

    # Gọi hàm visualize_prediction
    visualize_prediction(model, test_loader, device, sample_idx=sample_idx, apply_denormalization=True)

if __name__ == "__main__":
    main()
