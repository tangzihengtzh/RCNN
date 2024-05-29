import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from MyDatasets import MyDataset
from MyModel import get_model
from tqdm import tqdm

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} CUDA device(s)!")
    for i in range(device_count):
        print(torch.cuda.get_device_name(i))
    device = torch.device("cuda:0")
else:
    print("No CUDA devices found.")
    device = torch.device("cpu")


def collate_fn(batch):
    return tuple(zip(*batch))


def train(num_epochs):
    batch_size = 8
    transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    train_dataset = MyDataset(image_folder='data/img', label_folder='data/label', transform=transform)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # device = torch.device("cpu")
    print('当前设备是:', device)
    model = get_model(num_classes=21)  # 类别数量（包括背景）
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    model_path = 'saved_model/fasterrcnn.pth'
    if os.path.exists(model_path):
        print('开始加载模型')
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        print("加载完毕")

    avg_loss_per_epoch = []  # 用于存储每个epoch的平均loss

    for epoch in range(num_epochs):
        print("epoch:", epoch)
        running_loss = 0.0
        model.train()
        # for inputs, labels in train_data_loader:
        # 使用 tqdm 创建进度条
        progress_bar = tqdm(train_data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for inputs, labels in progress_bar:
            inputs = list(image.to(device) for image in inputs)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

            optimizer.zero_grad()
            loss_dict = model(inputs, labels)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        lr_scheduler.step()

        avg_loss_per_epoch.append(running_loss / len(train_data_loader))
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_data_loader):.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            }, model_path)
            print(f"模型已保存至 {model_path}")

    # 绘制并保存loss折线图
    plt.figure()
    plt.plot(range(1, num_epochs + 1), avg_loss_per_epoch, marker='o')
    plt.title('Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.savefig('loss_plot.jpg')
    plt.close()


if __name__ == '__main__':
    train(20)
