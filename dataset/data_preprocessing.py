import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message=".*iCCP: known incorrect sRGB profile.*")


# Hàm đọc ảnh gốc (RGB)
def read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Hàm đọc mask (đen trắng)
def read_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Không tìm thấy mask: {mask_path}")
    # Áp dụng threshold để mask có giá trị nhị phân 0 hoặc 255
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return mask

# Hàm chuyển đổi ảnh và mask thành một tensor
def get_transforms(target_size=(256, 256), is_train=True, for_visualization=False):
    if for_visualization:
        # chỉ resize và chuyển sang tensor, ko normalize hoặc augment ngẫu nhiên
        transform = A.Compose([
            A.Resize(*target_size),
            ToTensorV2()
        ])
    elif is_train:
        transform = A.Compose([
            A.Resize(*target_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(*target_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    return transform

def process_sample(img_path, mask_path, transform):
    image = read_image(img_path)  # RGB ảnh
    mask = read_mask(mask_path)  # Grayscale mask (0/255)

    # print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")

    # Nếu mask chưa cùng size với image -> resize mask
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply transform
    augmented = transform(image=image, mask=mask)

    image_tensor = augmented['image']
    mask_tensor = augmented['mask']

    # Convert mask to binary 0.0 or 1.0
    mask_tensor = (mask_tensor > 0).float().unsqueeze(0)

    return image_tensor, mask_tensor


class PratheepanDataset(Dataset):
    def __init__(self, face_img_folder, face_mask_folder,
                 family_img_folder, family_mask_folder,
                 transform=None):
        self.transform = transform if transform is not None else get_transforms()
        self.samples = []

        # Load mẫu từ FacePhoto
        face_images = sorted([f for f in os.listdir(face_img_folder) if f.lower().endswith(('.jpg', '.png'))])
        for img_file in face_images:
            img_path = os.path.join(face_img_folder, img_file)

            # Thay phần mở rộng của file ảnh thành .png để tìm mask
            mask_file = os.path.splitext(img_file)[0] + '.png'
            mask_path = os.path.join(face_mask_folder, mask_file)

            if os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))
            else:
                print(f"Không tìm thấy mask cho {img_path}")
                print(f"Đường dẫn mask: {mask_path}")
                print(f"Đường dẫn ảnh: {img_path}")

        # Load mẫu từ FamilyPhoto
        family_images = sorted([f for f in os.listdir(family_img_folder) if f.lower().endswith(('.jpg', '.png'))])
        for img_file in family_images:
            img_path = os.path.join(family_img_folder, img_file)
            mask_file = os.path.splitext(img_file)[0] + '.png'
            mask_path = os.path.join(family_mask_folder, mask_file)

            if os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))
            else:
                print(f"Không tìm thấy mask cho {img_path}")
                print(f"Đường dẫn mask: {mask_path}")
                print(f"Đường dẫn ảnh: {img_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        return process_sample(img_path, mask_path, self.transform)

class CelebAMaskHQPreprocessedDataset(Dataset):
    def __init__(self, imgs_folder, masks_folder, transform=None):
        self.imgs_folder = imgs_folder
        self.masks_folder = masks_folder

        self.transform = transform if transform is not None else get_transforms()

        self.img_files = sorted([f for f in os.listdir(self.imgs_folder) if f.lower().endswith(('.jpg', '.png'))], key=lambda x: int(os.path.splitext(x)[0]))
        self.mask_files = sorted([f for f in os.listdir(self.masks_folder) if f.lower().endswith(('.jpg', '.png'))], key=lambda x: int(os.path.splitext(x)[0]))

        if len(self.img_files) != len(self.mask_files):
            raise ValueError("Số lượng ảnh và mask không khớp!")

        self.samples = [
            (os.path.join(self.imgs_folder, img_file),
             os.path.join(self.masks_folder, os.path.splitext(img_file)[0] + '.png'))
            for img_file in self.img_files
        ]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        return process_sample(img_path, mask_path, self.transform)

def get_dataset(dataset_mode, transform = None, **kwargs):
    dataset_mode = dataset_mode.lower()
    if dataset_mode == "pratheepan":
        return PratheepanDataset(
            face_img_folder=kwargs.get("face_img_folder"),
            face_mask_folder=kwargs.get("face_mask_folder"),
            family_img_folder=kwargs.get("family_img_folder"),
            family_mask_folder=kwargs.get("family_mask_folder"),
            transform=transform
        )
    elif dataset_mode == "celeba":
        return CelebAMaskHQPreprocessedDataset(
            imgs_folder=kwargs.get("imgs_folder"),
            masks_folder=kwargs.get("masks_folder"),
            transform=transform
        )
    else:
        raise ValueError("dataset_mode phải là 'pratheepan' hoặc 'celeba'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chọn bộ dataset và hiển thị dataset cho huấn luyện/tiền xử lý.")
    parser.add_argument("--dataset_mode", type=str, required=True,
                        choices=["pratheepan", "celeba"],
                        help="Chọn dataset: 'pratheepan' hoặc 'celeba'")
    # Các đường dẫn cho Pratheepan Dataset
    parser.add_argument("--face_img_folder", type=str, default="../dataset/Face_Dataset/Pratheepan_Dataset/FacePhoto",
                        help="Thư mục chứa ảnh FacePhoto (Pratheepan)")
    parser.add_argument("--face_mask_folder", type=str, default="../dataset/Face_Dataset/Ground_Truth/GroundT_FacePhoto",
                        help="Thư mục chứa mask FacePhoto (Pratheepan)")
    parser.add_argument("--family_img_folder", type=str, default="../dataset/Face_Dataset/Pratheepan_Dataset/FamilyPhoto",
                        help="Thư mục chứa ảnh FamilyPhoto (Pratheepan)")
    parser.add_argument("--family_mask_folder", type=str, default="../dataset/Face_Dataset/Ground_Truth/GroundT_FamilyPhoto",
                        help="Thư mục chứa mask FamilyPhoto (Pratheepan)")

    # Các đường dẫn cho CelebAMask-HQ Preprocessed Dataset
    parser.add_argument("--imgs_folder", type=str, default="../dataset/preprocessed/imgs",
                        help="Thư mục chứa ảnh gốc (CelebAMask-HQ preprocessed)")
    parser.add_argument("--masks_folder", type=str, default="../dataset/preprocessed/masks",
                        help="Thư mục chứa mask (CelebAMask-HQ preprocessed)")

    parser.add_argument("--target_size", type=int, nargs=2, default=[256, 256],
                        help="Kích thước ảnh sau khi resize, ví dụ: 256 256")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--display_indices", type=int, nargs="+", default=[0],
                        help="Chỉ số các ảnh trong batch để hiển thị (vd: 0 1 2)")
    parser.add_argument("--is_train", action="store_true",
                        help="Áp dụng augment cho training")
    parser.add_argument("--for_visualization", action="store_true",
                        help="Sử dụng transform cho hiển thị ảnh gốc (không Normalize & augment ngẫu nhiên)")

    args = parser.parse_args()

    # Thiết lập transform cho huấn luyện/validation
    transform = get_transforms(
        target_size=tuple(args.target_size),
        is_train=args.is_train,
        for_visualization=False  # cho quá trình training/validation, chúng ta dùng normalize nếu không set for_visualization
    )
    # Thiết lập transform riêng cho hiển thị ảnh gốc, không normalize và augment
    display_transform = get_transforms(
        target_size=tuple(args.target_size),
        for_visualization=True
    )

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

    print(f"Tổng số samples: {len(dataset)}")

    # Tạo DataLoader (không shuffle để hiển thị theo thứ tự file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print("Hiển thị một batch ảnh theo thứ tự (theo tên file):")
    for batch_idx, (images, masks) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}: images {images.shape}, masks {masks.shape}")
        for disp_idx in args.display_indices:
            if disp_idx >= images.size(0):
                print(f"⚠️ Chỉ số {disp_idx} vượt quá batch size {images.size(0)}. Bỏ qua.")
                continue

            sample_idx = batch_idx * args.batch_size + disp_idx
            sample_img_path, sample_mask_path = dataset.samples[sample_idx]

            # Đọc lại ảnh gốc và mask để hiển thị (không Normalize, chỉ resize)
            orig_image = read_image(sample_img_path)
            orig_mask = read_mask(sample_mask_path)
            if orig_image.shape[:2] != orig_mask.shape[:2]:
                orig_mask = cv2.resize(orig_mask, (orig_image.shape[1], orig_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            disp_aug = display_transform(image=orig_image, mask=orig_mask)
            img_disp = disp_aug['image'].permute(1, 2, 0).cpu().numpy()
            # img_disp ở đây đã có giá trị [0,1] sau ToTensorV2() do không Normalize.
            mask_disp = disp_aug['mask'].squeeze().cpu().numpy()

            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(img_disp)
            plt.title(f"Ảnh gốc: {os.path.basename(sample_img_path)}")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(mask_disp, cmap="gray")
            plt.title(f"Mask: {os.path.basename(sample_mask_path)}")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
        break