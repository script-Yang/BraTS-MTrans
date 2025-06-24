# import torch
# import torch.nn as nn

# # 确保 CUDA 可用
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# # 创建一个简单的输入张量 [batch_size, channels, height, width]
# x = torch.randn(1, 3, 64, 64).to(device)

# # 定义一个简单的卷积层
# conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1).to(device)

# # 前向传播
# try:
#     out = conv(x)
#     print("✅ Forward pass successful! Output shape:", out.shape)
# except RuntimeError as e:
#     print("❌ CUDA RuntimeError:", e)


# # pip freeze > requirements_recon.txt


import torch
print(torch.cuda.is_available())                      # True or False
print(torch.cuda.get_device_name(0))                  # GPU 名称
print(torch.version.cuda)                             # 编译时的 CUDA 版本（如 '11.3'）
print(torch.backends.cudnn.version())                 # CUDNN 版本
