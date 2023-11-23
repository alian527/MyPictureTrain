from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import cv2



Img_path=("Pics/pic1.png")
img=Image.open (Img_path)
writer=SummaryWriter("logs")

cv_img=cv2.imread(Img_path)

tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)

writer.add_image("Tensor_img",tensor_img)

writer.close()