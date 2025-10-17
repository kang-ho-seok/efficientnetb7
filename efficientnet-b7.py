import math
import pandas as pd
import torch
import torchvision
from tqdm import tqdm
import numpy as np
import timm
from torch import nn
from torchvision import transforms
from torchvision.ops import StochasticDepth
from torchinfo import summary
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
from torch.utils.data import random_split, Subset
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 채널 수(width) 조정
def _make_divisible(v, divisor, min_value = None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)

# 채널별 가중치 조정
class SE_Block(nn.Module):
    def __init__(self, in_channels, squeeze_channels):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, 1),
            nn.SiLU(inplace = True),
            nn.Conv2d(squeeze_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        se = self.squeeze(x)
        se = self.excitation(se)
        out = se * x
        return out
    
class MBConv(nn.Module):
    def __init__(self, kernel_size, in_channels, exp_channels, out_channels, stride, sd_prob):
        super().__init__()

        self.use_skip_connection = (stride == 1 and in_channels == out_channels)
        self.stochastic = StochasticDepth(sd_prob, mode = "row")

        expand = nn.Sequential(
            nn.Conv2d(in_channels, exp_channels, 1, bias = False),
            nn.BatchNorm2d(exp_channels, momentum = 0.99),
            nn.SiLU(inplace = True),
        )

        depthwise = nn.Sequential(
            nn.Conv2d(exp_channels, exp_channels, kernel_size, stride, padding = (kernel_size - 1) // 2, groups = exp_channels, bias = False),
            nn.BatchNorm2d(exp_channels, momentum = 0.99),
            nn.SiLU(inplace = True),
        )

        squeeze_channels = in_channels // 4     # reduction_ratio = 4
        se_block = SE_Block(exp_channels, squeeze_channels)

        pointwise = nn.Sequential(
            nn.Conv2d(exp_channels, out_channels, 1, bias = False),
            nn.BatchNorm2d(out_channels, momentum = 0.99),
            # No Activation
        )

        layers = []
        if in_channels < exp_channels:
            layers += [expand]
        layers += [depthwise, se_block, pointwise]

        self.residual = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_skip_connection:
            residual = self.residual(x)
            residual = self.stochastic(residual)
            return residual + x
        else:
            return self.residual(x)
        
class EfficientNet_B7(nn.Module):
  def __init__(self, num_classes=1000):
    super().__init__()

    self.wideth_mult=2.0
    self.depth_mult=3.1

    stochastic_depth_p = 0.2
    cfgs = [
           #[k, t,  c, n, s] # kernel, expansion, out_channels, num_blocks, stride
            [3, 1, 16, 1, 1],
            [3, 6, 24, 2, 2],
            [5, 6, 40, 2, 2],
            [3, 6, 80, 3, 2],
            [5, 6, 112, 3, 1],
            [5, 6, 192, 4, 2],
            [3, 6, 320, 1, 1],
        ]

    in_channels = _make_divisible(32 * self.wideth_mult, 8)

    self.first_layers = nn.Sequential(
        nn.Conv2d(3, in_channels, 3, stride = 2, padding = 1, bias = False),
        nn.BatchNorm2d(in_channels, momentum = 0.99),
        nn.SiLU(inplace = True),
    )

    layers = []
    num_block = 0
    num_total_layers = sum(math.ceil(cfg[-2] * self.depth_mult) for cfg in cfgs)
    for kernel_size, t, c, n, s in cfgs:
        n = math.ceil(n * self.depth_mult)
        for i in range(n):
            stride = s if i == 0 else 1
            exp_channels = _make_divisible(in_channels * t, 8)
            out_channels = _make_divisible(c * self.wideth_mult, 8)
            sd_prob = stochastic_depth_p * (num_block / (num_total_layers - 1))

            layers.append(MBConv(kernel_size, in_channels, exp_channels, out_channels, stride, sd_prob))
            in_channels = out_channels
            num_block += 1

    self.layers = nn.Sequential(*layers)

    last_channel = _make_divisible(1280 * self.wideth_mult, 8)

    self.last_layers = nn.Sequential(
        nn.Conv2d(in_channels, last_channel, 1, bias = False),
        nn.BatchNorm2d(last_channel, momentum = 0.99),
        nn.SiLU(inplace = True)
    )
    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(last_channel, num_classes)
    )

  def forward(self, x):
      x = self.first_layers(x)
      x = self.layers(x)
      x = self.last_layers(x)
      x = self.avgpool(x)
      x = torch.flatten(x, 1)
      x = self.fc(x)
      return x
  
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = 'checkpoint.pt'  # 모델 저장 경로

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def train_model(model, train_dataloader, val_dataloader, criterion, scheduler, optimizer, num_epochs) :
    model.to(device)
    torch.backends.cudnn.benchmark = True

    train_accuracy_list = []
    val_accuracy_list = []
    train_loss_list = []
    val_loss_list = []
    
    early_stopper = EarlyStopping(
        patience=10, 
        verbose=True, 
        delta=0.001,
    )

    for epoch in range(num_epochs) :
        print(f'Epoch {epoch + 1}/ {num_epochs}')
        print('*' * 30)

        # ====== 학습(Training) 단계 ======
        model.train() # 모델을 학습 모드로 설정

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_dataloader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()*inputs.size(0)
            running_corrects += torch.sum(preds == labels)

        epoch_train_loss = running_loss / len(train_dataloader.dataset)
        epoch_train_acc = running_corrects.double() / len(train_dataloader.dataset)

        scheduler.step()

        print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc.double():.4f}')

        # ====== 검증(Validation) 단계 ======
        model.eval() # 모델을 평가 모드로 설정

        best_val_acc = 0.0
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad(): # 검증 단계에서는 그라디언트 계산 비활성화
            for inputs, labels in tqdm(val_dataloader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_running_loss += loss.item()*inputs.size(0)
                val_running_corrects += torch.sum(preds == labels)

        epoch_val_loss = val_running_loss / len(val_dataloader.dataset)
        epoch_val_acc = val_running_corrects.double() / len(val_dataloader.dataset)

        print(f'Validation Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc.double():.4f}')
        print('*' * 30)
        
        early_stopper(epoch_val_loss, model)

        train_accuracy_list.append(epoch_train_acc.item())
        train_loss_list.append(epoch_train_loss)
        val_accuracy_list.append(epoch_val_acc.item())
        val_loss_list.append(epoch_val_loss)

        if early_stopper.early_stop:
            print(f"Early Stopping: Training stopped at epoch {epoch + 1}")
            break
        
    try:
        model.load_state_dict(torch.load(early_stopper.path))
        print(f"Loaded best model from {early_stopper.path}")
    except FileNotFoundError:
        print("Warning: Checkpoint file not found. Returning last trained model.")

    return train_accuracy_list, val_accuracy_list, train_loss_list, val_loss_list

def test_model(model, dataloader) :
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load("checkpoint.pt"))
    accuracy_list = []
    loss_list = []
    with torch.no_grad() :
        for inputs, labels in tqdm(dataloader) :
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            accuracy_list.append(torch.sum(preds == labels).item())
            loss_list.append(loss.item())
    return accuracy_list, loss_list

batch_size = 64
num_epochs = 100

root = './dataset'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

test_val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# dataset_train = torchvision.datasets.CIFAR10(root='../dataset', train=True,
#                                              download=True, transform=train_transforms)
# dataset_val = torchvision.datasets.CIFAR10(root='../dataset', train=True,
#                                              download=True, transform=test_val_transforms)
dataset_test = torchvision.datasets.CIFAR10(root='../dataset', train=False,
             download=True, transform=test_val_transforms)

train_full = datasets.CIFAR10(root='../dataset', train=True, download=True, transform=train_transforms)
val_full = datasets.CIFAR10(root='../dataset', train=True, download=True, transform=test_val_transforms)

# 인덱스만 분할
train_size = int(0.9 * len(train_full))
val_size = len(train_full) - train_size
indices = list(range(len(train_full)))
train_indices, val_indices = random_split(indices, [train_size, val_size])

# 분할된 인덱스로 각 변환이 적용된 데이터셋에서 Subset을 생성
dataset_train = Subset(train_full, train_indices)
dataset_val = Subset(val_full, val_indices)

train_loader = DataLoader(dataset_train, batch_size=batch_size,
                        shuffle=True, num_workers=2)
val_loader = DataLoader(dataset_val, batch_size=batch_size,
                       shuffle=False, num_workers=2)
test_loader = DataLoader(dataset_test, batch_size=batch_size,
                       shuffle=False, num_workers=2)

pre_trained_model = models.efficientnet_b7(weights='IMAGENET1K_V1')
model = EfficientNet_B7(num_classes=1000)

pre_trained_weights = pre_trained_model.state_dict()
model_state_dict = model.state_dict()
new_state_dict = {}

model_keys = list(model_state_dict.keys())
pre_trained_values = list(pre_trained_weights.values())

if len(model_keys) == len(pre_trained_values):
    for i in range(len(model_keys)):
        k_custom = model_keys[i]
        v_pre = pre_trained_values[i]
        if v_pre.shape == model_state_dict[k_custom].shape:
            new_state_dict[k_custom] = v_pre

model.load_state_dict(new_state_dict)

in_features = model.fc[1].in_features
model.fc[1] = nn.Linear(in_features=in_features, out_features=10, bias=True)

# for param in model.parameters():
#     param.requires_grad = True  # 나머지 freeze
# for param in pre_trained_model.classifier[1].parameters():
#     param.requires_grad = True   # 학습되는 레이어 설정

optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9 ,weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
lr_scheduler = lrs.StepLR(optimizer, step_size=2, gamma=0.97)

train_accuracy, val_accuracy, train_loss, val_loss = train_model(model, train_loader, val_loader, criterion, lr_scheduler, optimizer, num_epochs=num_epochs)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="w")
epochs = range(len(train_accuracy))
# 정확도(Accuracy) 그래프
ax1.plot(train_accuracy, label="Train Accuracy")
ax1.plot(val_accuracy, label="Validation Accuracy")
ax1.set_title("Accuracy over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True)
ax1.set_xticks(range(0, len(epochs)+1, 2))#epoch 2 간격으로 눈금 표시
# 손실(Loss) 그래프
ax2.plot(train_loss, label="Train Loss")
ax2.plot(val_loss, label="Validation Loss")
ax2.set_title("Loss over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)
ax2.set_xticks(range(0, len(epochs)+1, 2))

plt.tight_layout()
plt.savefig('train_accuracy_plot1.png')
plt.show()
# summary(model, input_size=(1, 3, 224, 224))
test_accuracy_list, test_loss_list = test_model(model, test_loader)

print(f'Test Accuracy : {np.sum(test_accuracy_list) / len(dataset_test):.4f}')