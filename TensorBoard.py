from torch.utils.tensorboard import SummaryWriter
import tensorflow
from PIL import Image
import cv2
import numpy as np

# 创建一个名为 "logs" 的 TensorBoard 日志写入器
writer = SummaryWriter("logs")

image_path="Pics/pic (2).jpg"
img=Image.open(image_path)
print(type(img))
img_array=np.array(img)
print(type(img_array))
img_PIL=Image.open(image_path)
img_array=np.array(img_PIL)

writer.add_image("test",img_array,2,dataformats="HWC")

# 循环从 0 到 99，将数据写入 TensorBoard
for i in range(100):
    # 使用 add_scalar 方法将名为 "y=x" 的标量数据写入 TensorBoard
    writer.add_scalar('y=x', i, i)

# 关闭 TensorBoard 写入器
writer.close()

#打开指令：tensorboard --logdir=logs