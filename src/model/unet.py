import torch
import torch.nn as nn
import warnings

# Bỏ qua cảnh báo liên quan đến profile sRGB không chính xác
warnings.filterwarnings("ignore", message=".*iCCP: known incorrect sRGB profile.*")


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32, dropout_prob=0.3):
        """
        Khởi tạo mô hình U-Net với các cải tiến chống overfitting:
          - Sử dụng Dropout2d trong các block để giảm overfitting.
          - BatchNorm để ổn định huấn luyện.

        Parameters:
            in_channels: Số kênh đầu vào (mặc định 3 cho ảnh RGB).
            out_channels: Số kênh đầu ra (1 cho bài toán nhị phân segmentation).
            init_features: Số lượng feature ban đầu, sau đó nhân dần theo encoder.
            dropout_prob: Xác suất dropout được áp dụng trong mỗi block.
        """
        super(UNet, self).__init__()
        features = init_features

        # Encoder: các block tuần tự giảm kích thước không gian và tăng số kênh.
        self.encoder1 = UNet._block(in_channels, features, name="enc1", dropout_prob=dropout_prob)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Giảm kích thước ảnh một nửa
        self.encoder2 = UNet._block(features, features * 2, name="enc2", dropout_prob=dropout_prob)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3", dropout_prob=dropout_prob)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4", dropout_prob=dropout_prob)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck: tầng giữa encoder và decoder, có số kênh cao nhất.
        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck", dropout_prob=dropout_prob)

        # Decoder: sử dụng ConvTranspose2d để tăng kích thước không gian, kết hợp (skip connections) với các encoder tương ứng.
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        # Decoder4 nhận vào 2 nguồn: output từ upconv4 và encoder4 (do skip connection)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4", dropout_prob=dropout_prob)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3", dropout_prob=dropout_prob)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2", dropout_prob=dropout_prob)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1", dropout_prob=dropout_prob)

        # Lớp cuối cùng để chuyển đổi số kênh thành out_channels, sử dụng kernel 1x1.
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)  # Kích thước: [B, features, H, W]
        enc2 = self.encoder2(self.pool1(enc1))  # [B, features*2, H/2, W/2]
        enc3 = self.encoder3(self.pool2(enc2))  # [B, features*4, H/4, W/4]
        enc4 = self.encoder4(self.pool3(enc3))  # [B, features*8, H/8, W/8]

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))  # [B, features*16, H/16, W/16]

        # Decoder path: mỗi bước dùng ConvTranspose2d tăng kích thước,
        # sau đó ghép (concat) với output từ encoder tương ứng và chuyển qua block.
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)  # Ghép với encoder4
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)  # Ghép với encoder3
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)  # Ghép với encoder2
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # Ghép với encoder1
        dec1 = self.decoder1(dec1)

        # Lớp cuối cùng chuyển đổi số kênh về out_channels và sử dụng hàm sigmoid để tạo output dạng xác suất
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name, dropout_prob=0.3):
        """
        Xây dựng một block gồm:
          - 2 lớp convolution (kernel size=3, padding=1) để duy trì kích thước không gian.
          - Batch normalization sau mỗi convolution để ổn định quá trình huấn luyện.
          - ReLU activation để tạo phi tuyến.
          - Dropout2d sau lớp ReLU đầu tiên nhằm giảm overfitting bằng cách ngẫu nhiên tắt các feature map.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),  # Thêm dropout để giảm overfitting
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
