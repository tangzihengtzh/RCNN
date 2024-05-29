import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from MyModel import get_model


def load_model(model_path, num_classes):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()  # 设置模型为评估模式
    return model


def detect_and_visualize(image_path, model, transform):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度

    with torch.no_grad():
        prediction = model(image_tensor)

    # 可视化结果
    draw = ImageDraw.Draw(image)
    for element in range(len(prediction[0]['boxes'])):
        box = prediction[0]['boxes'][element].tolist()
        score = prediction[0]['scores'][element].item()
        label = prediction[0]['labels'][element].item()

        if score > 0.5:  # 只显示置信度大于0.5的检测结果
            draw.rectangle(box, outline='red', width=3)
            draw.text((box[0], box[1]), f"Label: {label} Score: {score:.2f}", fill='red')

    image.show()


if __name__ == '__main__':
    model_path = 'saved_model/fasterrcnn.pth'
    image_path = r"E:\python_prj\FRCNN\data\img\00036-4181474945.png"  # 替换为你的图片路径
    num_classes = 3  # 类别数量（包括背景）

    transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
    model = load_model(model_path, num_classes)

    detect_and_visualize(image_path, model, transform)
