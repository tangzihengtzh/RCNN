import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes):
    # 加载预训练的 Faster R-CNN 模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # 获取模型的分类器的输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 替换预训练模型的分类器，num_classes 是数据集中的类别数量（包括背景）
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# 示例：定义模型并打印模型结构
if __name__ == '__main__':
    num_classes = 3  # 类别数量（包括背景）
    model = get_model(num_classes)
    # print(model)

    # 初始化一个随机张量作为输入
    input_tensor = torch.randn(1, 3, 448, 448)  # 批大小为1，3通道（RGB），图像大小为448x448
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        output = model(input_tensor)

    # 打印输入和输出的张量形状
    print("Input tensor shape:", input_tensor.shape)

    # 打印输出的结构
    for i, out in enumerate(output):
        print(f"Output {i}:")
        for key, value in out.items():
            print(f"  {key}: {value.shape}")
