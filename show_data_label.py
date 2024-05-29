import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# 文件夹路径
image_folder_path = 'data/img'
label_folder_path = 'data/label'


def visualize_image_with_label(image_file):
    # 构造文件路径
    image_path = os.path.join(image_folder_path, image_file)
    label_file = os.path.splitext(image_file)[0] + '.xml'
    label_path = os.path.join(label_folder_path, label_file)

    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"Image file {image_path} does not exist.")
        return
    if not os.path.exists(label_path):
        print(f"Label file {label_path} does not exist.")
        return

    # 读取图片
    image = Image.open(image_path)

    # 读取XML文件并解析
    tree = ET.parse(label_path)
    root = tree.getroot()

    # 获取边界框和类别
    boxes = []
    labels = []
    for idx, item in enumerate(root.findall('./outputs/object/item')):
        name = item.find('name').text
        bndbox = item.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes.append((xmin, ymin, xmax, ymax))
        labels.append(f"{name} {idx + 1}")  # 类别名称和索引号

    # 绘制边界框和标签
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box, label in zip(boxes, labels):
        draw.rectangle(box, outline='red', width=3)
        draw.text((box[0], box[1] - 10), label, fill='red', font=font)

    # 显示图片
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# 遍历图像文件夹中的所有图像文件并进行可视化
for image_file in os.listdir(image_folder_path):
    if image_file.endswith('.jpg'):  # 假定图片为PNG格式
        visualize_image_with_label(image_file)
