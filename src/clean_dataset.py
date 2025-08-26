from PIL import Image
import os

def clean_png_metadata(file_path):
    try:
        img = Image.open(file_path)
        # Lưu lại ảnh mà không truyền thông tin pnginfo (không giữ lại các chunks không cần thiết)
        img.save(file_path, format="PNG", pnginfo=None)
        print(f"Cleaned: {file_path}")
    except Exception as e:
        print(f"Error cleaning {file_path}: {e}")

# Duyệt qua các file png trong folder và làm sạch metadata
def clean_folder(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                clean_png_metadata(file_path)

# Ví dụ sử dụng:
folder_paths = [
    '../dataset/Face_Dataset/Pratheepan_Dataset/FacePhoto',
    '../dataset/Face_Dataset/Ground_Truth/GroundT_FacePhoto',
    '../dataset/Face_Dataset/Pratheepan_Dataset/FamilyPhoto',
    '../dataset/Face_Dataset/Ground_Truth/GroundT_FamilyPhoto'
]

for folder in folder_paths:
    clean_folder(folder)
