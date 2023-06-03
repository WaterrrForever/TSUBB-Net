from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, transform=None):
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):  # 调用对象的返回值
        img = Image.open(self.images_path[item]).convert('L')
        if self.transform is not None:
            img = self.transform(img)
        label = img.clone()

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
        return images, labels
