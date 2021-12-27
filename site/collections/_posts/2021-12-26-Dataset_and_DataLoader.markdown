---
date: 2021-12-26 00:00:00
title: Dataset과 과대적합
description: Pytorch에서 제공하는 Dataset과 DataLoader 이해하기
tags:
- pytorch
- Dataset
- DataLoader
- Dropout
- BatchNormalization
image: /images/posts/02_dataset_and_dataloader/thumbnail.png
---

<!-- 포스트 이미지 폴더: /images/posts/02_dataset_and_dataloader/ -->
<h1>Dataset과 과대적합</h1>

데이터가 소량일 때는 전체데이터를 학습해도 되지만, 데이터량이 많아지거나, 신경망 계층의 증가, 또는 파라미터가 늘어나면 전체 데이터를 메모리에서 처리하기 어려워진다. 이 문제를 해결하기 위한 방법이 ***mini-batch***이다. 또한 머신러닝을 하다보면 항상 피할 수없는 것이 과대적합이다. 과대적합에 대해 간단히 알아보고 이를 해결하는 방법을 PyTorch를 통해 알아보자.


## 1. Dataset과 DataLoader

파이토치에는 Dataset과 DataLoader라는 기능이 있어서 미니 배치 학습이나 데이터를 무작위로 섞고, 그리고 병령처리까지 수행할 수 있다.
**TensorDataset**은 **Dataset**을 상속한 클래스로 학습 데이터 X와 레이블 Y를 묶어놓은 데이터 오브젝트이다. 이 **TensorDataset**을 **DataLoader**에 전달하면 for 루프에서 데이터의 일부만 간단히 추출할 수 있게 된다. 
> <span style="color: crimson;">[!caution]</span>  
> TensorDataset에는 **텐서**만 전달할 수 있으며 Variable은 전달할 수 없다.


```python
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits

# Dataset 작성
digits = load_digits()
X = digits.data
Y = digits.target

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.int64)

ds = TensorDataset(X, Y)

# 순서로 섞어서 64개씩 데이터를 반환하는 DataLoader 작성
loader = DataLoader(ds, batch_size=64, shuffle=True)
```

---

## 2. Dropout을 사용한 정규화
신경망은 표현력이 매우 높은 모델이지만, 한편으로는 '훈련된 데이터와만' 궁합이 좋아서 다른 데이터에 적용할 수 없거나 훈련이 불안정해서 시간이 오래 걸리는 문제가 있다. 여기서는 이런 문제를 해결하기 위한 두 가지 대표적인 기법인 **Dropout**과 **Batch Normalization**에 대해 다룬다.

신경망뿐만 아니라 머신러닝의 공통적인 문제로 <span style="color:crimson;">과적합(overfitting)</span>을 들 수 있다. 과학습이란 훈련용 데이터에 파라미터가 과하게 최적화되어 다른 데이터에 대한 판별 능력이 떨어지게 되는 현상이다. 예를 들어, 사람이 시험 공부를 할 때 원리는 파악하지 않고 유형만 학습하여 응용력이 떨어지는 것과 같다. 특히, 층이 깊은 신경망은 파라미터가 많아서 충분한 데이터가 없으면 과학습이 발생할 수 있다. 과대적합을 막는데에는 다양한 방법이 있지만, 신경망에서는 Dropout이라는 것이 자주 사용된다. 몇 개의 노드(변수의 차원)를 랜덤으로 선택해서 의도적으로 사용하지 않는 방법이다. Dropout은 신경망 훈련 시에만 사용하고, 예측 시에는 사용하지 않는 것이 일반적이다. 파이토치에서는 모델의 train과 eval 메서드로 Dropout을 적용 또는 미적용할 수 있다.

코드로는 다음과 같이 구현한다.
```python
import torch
from sklearn.datasets import load_digits
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

k = 100

# Dataset 작성
digits = load_digits()
X = digits.data
Y = digits.target

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.int64)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

# 순서로 섞어서 64개씩 데이터를 반환하는 DataLoader 작성

net = nn.Sequential(
    nn.Linear(64, k),
    nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(k, k),
    nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(k, k),
    nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(k, k),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(k, 10)
)

# train과 eval 메서드로 Dropout 처리 적용
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

# 훈련용 데이터로 DataLoader를 작성
ds = TensorDataset(X_train, Y_train)
loader = DataLoader(ds, batch_size=32, shuffle=True)

train_losses = []
test_losses = []
for epoch in range(100):
    running_loss = 0.0

    # 신경망을 훈련 모드로 설정
    net.train()
    for i, (xx, yy) in enumerate(loader):
        y_pred = net(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / i)
    # 신경망을 평가 모드로 설정하고
    # 검정 데이터의 손실 함수를 계산
    net.eval()
    y_pred = net(X_test)
    test_loss = loss_fn(y_pred, Y_test)
    test_losses.append(test_loss.item())
```

---
3. Batch Normalization을 사용한 학습 가속

**SGD**를 사용한 신경망 학습에서는 각 변수의 차원이 동일한 범위의 값을 가지는 것이 중요하다. 변수 한개의 값이 너무 크게 되면 그 값이 모델의 결과를 지배하기 때문이다. 한 층으로 된 선형 모델 등에서는 사전에 데이터를 정규화해 두면되지만 깊은 신경망에서는 층이 늘어날 수록 데이터 분포가 바뀐다. 그렇기 때문에 입력 데이터의 정규화만으로는 부족하다. 또한, 앞에 있는 층의 학스에 의해 파라미터가 변화하므로 뒤에 있는 층의 학습이 불안정해지는 문제가 있다. 이런 문제를 해결하고 학습을 안정화하는 방법으로 <span style="color:crimson;"><b>Batch Normalization</b></span>이 있다.
마찬가지로 **Batch Normalization**도 훈련 시에만 적용하며, 평가 시에는 사용하지 않는다. 따라서 <span style="color:crimson;"><b>Dropout</b></span>

코드는 바뀐 부분이 거의 없으며 *net*의 표현 방벙만 `nn.Dropout() => nn.BatchNorm1d`으로 바뀐다.
```python
net = nn.Sequential(
    nn.Linear(64, k),
    nn.ReLU(),
    nn.BatchNorm1d(k),

    nn.Linear(k, k),
    nn.ReLU(),
    nn.BatchNorm1d(k),

    nn.Linear(k, k),
    nn.ReLU(),
    nn.BatchNorm1d(k),

    nn.Linear(k, k),
    nn.ReLU(),
    nn.BatchNorm1d(k),
    nn.Linear(k, 10)
)
```

지금까지는 PyTorch를 다루는데 있어서 기본적인 머신러닝에 대해서 알아보았다. 그리고 모델 노드 구성을 Sequential로 코드를 작성했는데, PyTorch의 경우는 `Class`기반으로 코드를 짜는 것이 많이 알려져있다. 따라서 다음 포스팅부터는 `Class`를 이용한 커스텀 계층을 만들어 보겠다.