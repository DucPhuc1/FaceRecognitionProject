import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import logging
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm  # Sử dụng tqdm cho progress bar
from tensorboardX import SummaryWriter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# print(f"Project root directory: {project_root}")

src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
# print(f"src directory: {src_path}")

# Import dataset và các hàm tiền xử lý
from dataset.data_preprocessing import get_dataset, get_transforms
# Import mô hình UNet cải tiến
from model.unet import UNet

# Thiết lập thiết bị GPU (nếu có)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Đang sử dụng device:", device)


# Định nghĩa hàm Dice Loss
def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    loss = 1 - dice
    return loss.mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình U-Net cho segmentation vùng da mặt")
    parser.add_argument("--dataset_mode", type=str, required=True,
                        choices=["pratheepan", "celeba"],
                        help="Chọn dataset: 'pratheepan' hoặc 'celeba'")
    parser.add_argument("--target_size", type=int, nargs=2, default=[256, 256],
                        help="Kích thước ảnh sau khi resize, ví dụ: 256 256")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Số epoch huấn luyện")
    # Tỷ lệ chia dữ liệu: train, validation, test
    parser.add_argument("--train_split", type=float, default=0.70, help="Tỷ lệ dữ liệu train")
    parser.add_argument("--val_split", type=float, default=0.15, help="Tỷ lệ dữ liệu validation")
    parser.add_argument("--test_split", type=float, default=0.15, help="Tỷ lệ dữ liệu test")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay để chống overfitting")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")

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

    parser.add_argument("--is_train", action="store_true",
                        help="Áp dụng augment cho training")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Thiết lập logging và SummaryWriter để ghi log
    snapshot_path = "src/snapshot"
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=os.path.join(snapshot_path, "train_log.txt"),
                        level=logging.INFO,
                        format='[%(asctime)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("Starting training with args: " + str(args))

    # Thiết lập transform cho dữ liệu
    transform = get_transforms(target_size=tuple(args.target_size), is_train=args.is_train, for_visualization=False)

    # Khởi tạo dataset
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

    logging.info(f"Tổng số samples: {len(dataset)}")

    # Tách dataset thành train, validation và test
    total_size = len(dataset)
    train_size = int(args.train_split * total_size)
    val_size = int(args.val_split * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    logging.info(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Khởi tạo mô hình U-Net và đưa vào GPU
    model = UNet(in_channels=3, out_channels=1, init_features=32, dropout_prob=0.3).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    bce_loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    iter_num = 0
    max_iterations = args.num_epochs * len(train_loader)

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0

        # Tạo progress bar cho quá trình training của epoch
        epoch_iterator = tqdm(train_loader, desc=f"Train Epoch [{epoch + 1}/{args.num_epochs}]", ncols=80)
        for images, masks in epoch_iterator:
            images = images.to(device)
            masks = masks.to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss_bce = bce_loss_fn(outputs, masks)
            loss_dice = dice_loss(outputs, masks)
            loss = 0.5 * loss_bce + 0.5 * loss_dice
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            iter_num += 1
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('train/total_loss', loss.item(), iter_num)
            writer.add_scalar('train/loss_ce', loss_bce.item(), iter_num)
            epoch_iterator.set_postfix(loss=loss.item())

        train_loss = running_loss / train_size
        train_losses.append(train_loss)

        # Validation với progress bar
        model.eval()
        val_running_loss = 0.0
        val_iterator = tqdm(val_loader, desc=f"Validation Epoch [{epoch + 1}/{args.num_epochs}]", ncols=80)
        with torch.no_grad():
            for images, masks in val_iterator:
                images = images.to(device)
                masks = masks.to(device)
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)
                outputs = model(images)
                loss_bce = bce_loss_fn(outputs, masks)
                loss_dice = dice_loss(outputs, masks)
                loss = 0.5 * loss_bce + 0.5 * loss_dice
                val_running_loss += loss.item() * images.size(0)
                val_iterator.set_postfix(loss=loss.item())
        val_loss = val_running_loss / val_size
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        logging.info(f"Epoch [{epoch + 1}/{args.num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        # print(f"Epoch [{epoch + 1}/{args.num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(snapshot_path, "best_new_unet_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logging.info("Early stopping triggered.")
                print("Early stopping triggered.")
                break

    writer.close()

    # Vẽ đồ thị loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.legend()
    plt.show()

    logging.info("Huấn luyện hoàn thành. Mô hình tốt nhất được lưu tại: best_new_unet_model.pth")
    print("Huấn luyện hoàn thành. Mô hình tốt nhất được lưu tại: best_new_unet_model.pth")


if __name__ == "__main__":
    main()
