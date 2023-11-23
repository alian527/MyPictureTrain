import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10

dataset_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])



train_set: CIFAR10=torchvision.datasets.CIFAR10 (root="./dataset",train=True,transform=dataset_transform,download=True)
test_set: CIFAR10 = torchvision.datasets.CIFAR10(root="./dataset", train=False,transform=dataset_transform,download=True)

# print(test_set[0])
# print(test_set.classes)
#
# img,target=test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()
#
# print(test_set[0])

writer=SummaryWriter("p10")
for i in range(10):
    img,target=test_set[i]
    writer.add_image("test_set",img,i)

    writer.close()
#tensorboard --logdir=logs