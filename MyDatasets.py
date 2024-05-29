# import os
# import xml.etree.ElementTree as ET
# from PIL import Image
# import torch
# from torch.utils.data import Dataset
#
#
# class MyDataset(Dataset):
#     def __init__(self, image_folder, label_folder, transform=None, max_objects=10):
#         self.image_folder = image_folder
#         self.label_folder = label_folder
#         self.transform = transform
#         self.max_objects = max_objects
#         self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         # 获取图像文件路径
#         image_file = self.image_files[idx]
#         image_path = os.path.join(self.image_folder, image_file)
#
#         # 获取标签文件路径
#         label_file = os.path.splitext(image_file)[0] + '.xml'
#         label_path = os.path.join(self.label_folder, label_file)
#
#         # 读取图像
#         image = Image.open(image_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#
#         # 读取并解析XML文件
#         tree = ET.parse(label_path)
#         root = tree.getroot()
#
#         # 初始化标签列表
#         boxes = []
#         labels = []
#
#         # 获取边界框和类别
#         for item in root.findall('./outputs/object/item'):
#             name = item.find('name').text
#             class_id = self.get_class_id(name)
#             bndbox = item.find('bndbox')
#             xmin = int(bndbox.find('xmin').text)
#             ymin = int(bndbox.find('ymin').text)
#             xmax = int(bndbox.find('xmax').text)
#             ymax = int(bndbox.find('ymax').text)
#             boxes.append([xmin, ymin, xmax, ymax])
#             labels.append(class_id)
#
#         # 将数据转换为张量
#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         labels = torch.as_tensor(labels, dtype=torch.int64)
#
#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = labels
#
#         return image, target
#
#     def get_class_id(self, class_name):
#         class_map = {
#             'windows': 1,
#             'door': 2
#         }
#         return class_map.get(class_name, 0)
#
#
# # 示例：定义数据集和数据加载器
# if __name__ == '__main__':
#     from torchvision import transforms
#
#     transform = transforms.Compose([
#         transforms.Resize((448, 448)),
#         transforms.ToTensor()
#     ])
#
#     dataset = MyDataset(image_folder='data/img', label_folder='data/label', transform=transform)
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
#
#     for images, targets in data_loader:
#         print(images[0].shape, targets)


import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 获取图像文件路径
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_file)

        # 获取标签文件路径
        label_file = os.path.splitext(image_file)[0] + '.xml'
        label_path = os.path.join(self.label_folder, label_file)

        # 读取图像
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 读取并解析XML文件
        tree = ET.parse(label_path)
        root = tree.getroot()

        # 初始化标签列表
        boxes = []
        labels = []

        # 获取边界框和类别
        for obj in root.findall('object'):
            name = obj.find('name').text
            class_id = self.get_class_id(name)
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_id)

            # 获取对象的部位信息
            for part in obj.findall('part'):
                part_name = part.find('name').text
                part_class_id = self.get_class_id(part_name)
                part_bndbox = part.find('bndbox')
                part_xmin = int(part_bndbox.find('xmin').text)
                part_ymin = int(part_bndbox.find('ymin').text)
                part_xmax = int(part_bndbox.find('xmax').text)
                part_ymax = int(part_bndbox.find('ymax').text)
                boxes.append([part_xmin, part_ymin, part_xmax, part_ymax])
                labels.append(part_class_id)

        # 将数据转换为张量
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return image, target

    def get_class_id(self, class_name):
        class_map = {
            'person': 1,
            'head': 2,
            'hand': 3,
            'foot': 4
        }
        return class_map.get(class_name, 0)


# 示例：定义数据集和数据加载器
if __name__ == '__main__':
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ])

    dataset = MyDataset(image_folder='data/img', label_folder='data/label', transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for images, targets in data_loader:
        print(images[0].shape, targets)
