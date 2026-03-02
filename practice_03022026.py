# ====================================================
# 라이브러리
# ====================================================

import os
import sys
import requests
import time
import copy
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pythonlibs.torch_lib1
import matplotlib.font_manager as fm # 폰트 설정

from sklearn.metrics import confusion_matrix, classification_report
from pythonlibs.torch_lib1 import *
from IPython.display import display
from torchinfo import summary
from torchviz import make_dot
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingLR
import tqdm as tqdm_std

# ====================================================
# tqdm 사용할 수 있게 하기
# ====================================================
sys.modules['tqdm.notebook'] = tqdm_std
from tqdm import tqdm

try:
    import pythonlibs.torch_lib1
    pythonlibs.torch_lib1.tqdm = tqdm
except:
    pass

# ====================================================
# 폰트 설정
# ====================================================

plt.rc('font', family='Malgun Gothic')                                             # 윈도우용 한글 폰트 설정 (맑은 고딕)
plt.rcParams['axes.unicode_minus'] = False                                         # 마이너스 기호 깨짐 방지
print("✅ 한글 폰트 설정 완료 (Malgun Gothic)")


# ====================================================
# cuda
# ====================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")            # gpu 설정    
print(f'사용 디바이스 : {device}')
if torch.cuda.is_available():                                                      # gpu 이름 출력
    print(f'GPU 이름: {torch.cuda.get_device_name(0)}')


# ====================================================
# pythonlibs 폴더를 파이썬이 찾을 수 있도록 등록합니다.
# ====================================================

current_path = os.getcwd()                                                         # 현재 파일 위치
lib_path = os.path.join(current_path, 'pythonlibs')                                # 파일 위치 + /pythonlibs
if lib_path not in sys.path:                                                       # lib_path를 sys.path에 추가
    sys.path.append(lib_path)



# ====================================================
# 코드 정리 시작
# ====================================================

torch_seed(42)                                                                      # Random 안 바뀌게


# ====================================================
# 코드 정리  - transform
# ====================================================

train_transform = transforms.Compose([
    transforms.Resize(224),                                                        # 사이즈 재조정         
    transforms.RandomCrop(224),                                                    # 자르기
    # transforms.RandomResizedCrop(224),                                           # 사이즈 재조정 + 자르기   
    transforms.RandomHorizontalFlip(p=0.5),                                        # 좌우 대칭 만들기 - 새로 만들기 아님
    transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation =0.2),     # 밝기, 대비, 채도 바꾸기

    transforms.ToTensor(),                                                         # 0~1사이 텐서 값으로 만들기      
    transforms.RandomErasing(p=0.5, scale = (0.02,0.33),                           # 추출해 내기(지우기)  scale은 전체 크기의 0.2~0.33,
                              ratio = (0.3,3.3), value = 0, inplace = False ),     # ratio = 세로 가로 비율 3.3이 세로로 길다. 지운 값은 0으로 채우기
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))                             # -1과 1로 값을 맞추기 (x-mean)/std
])

test_transform = transforms.Compose([                                              # Random은 test에서 하지 않는다. 
    transforms.Resize(224),
    transforms.CenterCrop(224),                                                    # 이 때는 CenterCrop을 한다.   
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))                          

])


# ====================================================
# 코드 정리  - 전처리
# ====================================================

def data_preprocessing(train_datasets, test_datasets, batch_size = 64):
                 
    train_loader = DataLoader(
        train_datasets,                                                               # 첫번째는 배치 사이즈 만들거
        batch_size = batch_size,                                                      # 배치
        shuffle = True,                                                               # 섞을 것인지
        # num_workers = 4                                                             # cpu 할당 갯수 쓰지 말자.
    )

    test_loader = DataLoader(
        test_datasets,
        batch_size = batch_size,
        shuffle = False,                                                               # test 데이터는 섞지 않는다
    )

    return train_loader , test_loader



# ====================================================
# 코드 정리  - 데이터 가져오기
# ====================================================


url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'                    # 데이터 가져올 주소
zip_name = "hymenoptera_data.zip"                                                     # 저장할 짚파일 이름 
extract_dir = "data"                                                                  # 저장할 장소 

if not os.path.exists(zip_name):                                                      # 짚 파일이 존재하지 않을 경우 다운해라
    print("Downloading...")
    response = requests.get(url)
    with open(zip_name, "wb") as f:
        f.write(response.content)
    print("Download Complete!")

if os.path.exists(extract_dir):                                                       # 짚 파일이 존재할 경우 data 파일에 압축을 풀어라
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:                                   # zip_ref 가 
        zip_ref.extractall(extract_dir)
    print(f"Extraction Complete! Files are in '{extract_dir}' folder.")


data_dir= 'C:/ROKEY/data/hymenoptera_data'

train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'val')

classes = ['bee', 'ant']                                                              # label 개수

# 데이터셋 정의
# 훈련용
# 훈련1 (데이터 증강 적용할 때) - 배경을 지우는 게 좋다.
train_data = datasets.ImageFolder(train_dir, transform=train_transform)
# 훈련 2 (데이터 증강 미 적용할 때) - 이미지 출력용
train_data2 = datasets.ImageFolder(train_dir, transform=test_transform)
# 검증용
test_data = datasets.ImageFolder(test_dir, transform=test_transform)




# ====================================================
# 코드 정리  - 전이학습(모델 가져오기) - vgg19
# ====================================================

net_vgg19 = models.vgg19_bn(pretrained = True) 
net_vgg19 = net_vgg19.to(device)
in_features = net_vgg19.classifier[6].in_features
net_vgg19.classifier[6] = nn.Linear(in_features, len(classes))
net_vgg19.avgpool = nn.Identity()


# ====================================================
# 코드 정리  - 학습하기 - 훈련 함수(배치 O)
# ====================================================


def train_one_epoch(model, train_loader , criterion, optimizer , device):               # 한 epoch에서 모든 batch를 학습한 거다.
    
    model.train()                                                                       # 애 해줘야 학습률 이 높음

    running_loss = 0                                                                    # batch 안에서의 loss를 의미한다.
    correct = 0                                                                         # 맞는 개수 담기 위해서
    total = 0
    
    pbar = tqdm(train_loader, desc ='Training ... ')                                    # 게이지 바 생성을 위해 
                                                                                        # desc는 출력할 문구를 의미
    for inputs, labels in pbar:                                                         # for 문 대신 쓰는 거라 돌릴 data 집어 넣었다
        
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()                                                           # 기울기 초기화
        outputs = model(inputs)                                                         # 순전파 logits 생성. 은닉층 통과한거다.
                                                                                        # torch 생성할 때 이미 가중치(기울기)가 있다.  
        loss = criterion(outputs, labels)                                               # labels 랑 비교해서 loss 계산
        loss.backward()                                                                 # 기울기 수정
        optimizer.step()                                                                # 은닉층에 기울기 업데이트 

        predicted = torch.max(outputs, 1)[1]                                            # 확률 중 높은 걸 찾는다. 이 때 [1]은 인덱스를 의미 [0]은 값이다.
                                                                                        # outputs, 1 은 outputs의 행을 의미한다.
        total += labels.size(0)                                                         #
        running_loss += loss.item() * inputs.size(0)                                    # 데이터 하니씩 로스를 다 더해준다. 
                                                                                        # 이 때 배치 하나의 loss 값이라서 inputs 사이즈를 곱해준다. 
        correct += (predicted == labels).sum().item()                                   # 예상치와 label이 맞은 개수를 의미한다.


        pbar.set_postfix({'loss' : loss.item(), 'acc' : 100*correct/total})             # 한번 돌 때 마다 적어준다.
        
    
    epoch_loss = running_loss/total                                                     # 모든 배치를 돌았을 때 loss
    epoch_acc = 100*correct/total                                                       # 모든 배치를 돌았을 때 정확도
    
    return epoch_loss, epoch_acc

# ====================================================
# 코드 정리  - 학습하기 - 검증 함수(배치 O)
# ====================================================

def valiate(model, test_loader, criterion, device):                                     # optimizer가 없다. 기울기 업데이트 안해준다. 

    model.eval()                                                                        # dropout, batch_normalize를 수행하지 않는다.
    running_loss = 0                                                                    # batch 안에서의 loss를 의미한다.
    correct = 0                                                                         # 맞는 개수 담기 위해서
    total = 0

    pbar = tqdm(test_loader, desc ='Training ... ')                                    # 게이지 바 생성을 위해
    with torch.no_grad():
                                                                                        # desc는 출력할 문구를 의미
        for inputs, labels in pbar:                                                         # for 문 대신 쓰는 거라 돌릴 data 집어 넣었다
            
            inputs, labels = inputs.to(device), labels.to(device)
                                                                                            
            outputs = model(inputs)                                                         # 순전파 logits 생성. 은닉층 통과한거다.
                                                                                            # torch 생성할 때 이미 가중치(기울기)가 있다.  
            loss = criterion(outputs, labels)                                               # labels 랑 비교해서 loss 계산
    

            predicted = torch.max(outputs, 1)[1]                                            # 확률 중 높은 걸 찾는다. 이 때 [1]은 인덱스를 의미 [0]은 값이다.
                                                                                            # outputs, 1 은 outputs의 행을 의미한다.
            total += labels.size(0)                                                         #
            running_loss += loss.item() * inputs.size(0)                                    # 데이터 하니씩 로스를 다 더해준다. 
                                                                                            # 이 때 배치 하나의 loss 값이라서 inputs 사이즈를 곱해준다. 
            correct += (predicted == labels).sum().item()                                   # 예상치와 label이 맞은 개수를 의미한다.


            pbar.set_postfix({'loss' : loss.item(), 'acc' : 100*correct/total})             # 한번 돌 때 마다 적어준다.
        
    
    epoch_loss = running_loss/total                                                     # 모든 배치를 돌았을 때 loss
    epoch_acc = 100*correct/total                                                       # 모든 배치를 돌았을 때 정확도
    
    return epoch_loss, epoch_acc


# ====================================================
# 코드 정리  - early stopping
# ====================================================

class Earlystopping:                                                                    #  클래스다
    def __init__(self, patience = 5, delta = 0.0, path = 'best_model.pth'):             # patience 는 인내심이라고 5번 이상 개선이 안되면 멈춘다.

        self.patience = patience                                                        
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None                                                          # 계산을 안 돌렸으니 없다.
        self.early_stop = False                                                         # early stop을 안하겠다.
        self.val_loss_min = np.inf                                                      # 처음엔 무한대

    def __call__(self, val_loss, model):                                                # 함수 처럼 쓰겠다.
        score = -val_loss

        if self.best_score is None:                                                     # 처음일 때
            self.best_score = score                                                     # best score 생성
            self.save_checkpoint(val_loss, model)                                       # 모델 저장한다.
        elif score < self.best_score + self.delta:                                      # 개선이 안 되었을 때
            self.counter += 1                                                           # 스택 + 1 
            print(f'EarlyStopping counter : {self.counter}/{self.patience}')            
            if self.counter >= self.patience:                                           # 스택이 5 이상아면 stop
                self.early_stop = True
        else:                                                                           # 개선 되었을 때
            self.best_scroe = score                                                     # best_score 다시 저장
            self.save_checkpoint(val_loss, model)                                       # val_loss 랑 model 저장 
            self.counter = 0

    def save_checkpoint(self, val_loss, model):                                         # 가중치를 저장해 나중에 다시 쓰기 위해 저장하는 함수
        print(f'검증 손실 감소 : {self.val_loss_min:.5f} --> {val_loss:.6f}. 모델 저장 ....')
        torch.save(model.state_dict(), self.path)                                       # 저장 위치
        self.val_loss_min = val_loss                                                    # 최소 loss 업데이트

# ====================================================
# 코드 정리  - total 계산
# ====================================================

def total_caculation(model, train_loader, test_loader, num_epochs, optimizer, criterion, scheduler, early_stopping):
    model = model.to(device)
    history = { 
        'train_loss' : [],
        'train_acc': [],
        'val_loss' : [],
        'val_acc' : [],
        'lr' : []
    }

    start_time = time.time()

    for epoch in range(num_epochs):
        
        current_lr = optimizer.param_groups[0]['lr']

        train_loss , train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss , val_acc = valiate(model, test_loader, criterion, device ) 

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f'\n훈련 손실 : {train_loss:.4f}, 훈련 정확도 : {train_acc}')
        print(f'\n검증 손실 : {val_loss:.4f}, 검증 정확도 : {val_acc}')

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print('Early Stopping')
            break


    elapsed_time = time.time() - start_time
    print(f"학습 완료하는 데 걸린 총 소요시간: {elapsed_time/60:.2f}분")

    model.load_state_dict(torch.load('resnet18_best.pth'))

    return history

# ====================================================
# 코드 정리  - 시각화
# ====================================================

def plot_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 손실 곡선
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Curve', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 정확도 곡선
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-s', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Accuracy Curve', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # 학습률 변화
    axes[2].plot(epochs, history['lr'], 'g-^', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')  # 로그 스케일

    plt.tight_layout()
    plt.show()

    # 최종 결과 출력
    print('=' * 60)
    print('최종 학습 결과')
    print('=' * 60)
    print(f'최종 훈련 손실: {history["train_loss"][-1]:.4f}')
    print(f'최종 훈련 정확도: {history["train_acc"][-1]:.2f}%')
    print(f'최종 검증 손실: {history["val_loss"][-1]:.4f}')
    print(f'최종 검증 정확도: {history["val_acc"][-1]:.2f}%')
    print(f'최고 검증 정확도: {max(history["val_acc"]):.2f}%')
    print('=' * 60)


# ====================================================
# 코드 정리  - 계산
# ====================================================



lr = 0.001
num_epochs = 20
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net_vgg19.parameters(), lr = lr , momentum = 0.9, weight_decay = 1e-4)

scheduler = CosineAnnealingLR(optimizer,
                            T_max = num_epochs,
                            eta_min=1e-6)


train_loader, test_loader = data_preprocessing(train_data, test_data, batch_size = 32)          # 내 걸론 32가 최대임                         
early_stopping = Earlystopping(patience = 5 , delta = 0.001, path = 'resnet18_best.pth')

history = total_caculation(net_vgg19, train_loader, test_loader, num_epochs, optimizer, criterion, scheduler, early_stopping)


plot_history(history)

# ====================================================
# 코드 정리  - 혼동 행렬
# ====================================================

# def get_predictions(model, loader, device):
#   model.eval()

#   all_preds = []
#   all_labels = []

#   with torch.no_grad():
#     for inputs, labels in tqdm(loader, desc = 'Predicting'):
#       inputs = inputs.to(device)
#       outputs = model(inputs)
#       predicted = torch.max(outputs, 1)[1]

#       all_preds.extend(predicted.cpu().numpy()) # tensor를 cpu로 바꿔야 한다.
#       all_labels.extend(labels.cpu().numpy())

#       return np.array(all_preds), np.array(all_labels)
    
# def plot_confusion_matrix(y_true, y_pred, class_names):
#     # 혼동 행렬 계산
#     cm = confusion_matrix(y_true, y_pred)

#     # 시각화
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(
#         cm,
#         annot=True,
#         fmt='d',
#         cmap='Blues',
#         xticklabels=class_names,
#         yticklabels=class_names,
#         cbar_kws={'label': 'Count'}
#     )
#     plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
#     plt.ylabel('True Label', fontsize=12, fontweight='bold')
#     plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
#     plt.tight_layout()
#     plt.show()

#     # 클래스별 정확도 계산
#     class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100
#     # diagonal 은 1,1 2,2 등등 열과 행이 같은 거다.

#     print('\n클래스별 정확도:')
#     print('=' * 40)
#     for i, (name, acc) in enumerate(zip(class_names, class_accuracy)):
#         print(f'{name:12s}: {acc:6.2f}%')
#     print('=' * 40)

# # 예측 및 시각화
# y_pred, y_true = get_predictions(net_vgg19, test_loader, device)
# plot_confusion_matrix(y_true, y_pred, classes)

# # 분류 리포트 출력
# print('\n\n상세 분류 리포트:')
# print('=' * 60)
# print(classification_report(y_true, y_pred, target_names=classes, digits=4))