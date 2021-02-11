import torch.utils.data as data
import torchvision
from torchvision import transforms

from global_params import *

def data_loader(train_path, test_path):
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    train_data = torchvision.datasets.ImageFolder(root=train_path, transform=TRANSFORM_IMG)
    train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_data = torchvision.datasets.ImageFolder(root=test_path, transform=TRANSFORM_IMG)
    test_data_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print(len(train_data))
    print(len(test_data))
    print(train_data.classes)

    return (train_data_loader, test_data_loader)