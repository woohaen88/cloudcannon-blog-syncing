---
date: 2022-01-01 00:02:00
title: PyTorch-Training first Convolutional Neural Network (CNN)
description: 객체 지향 프로그래밍에서 자체 신경망 계층을 만들어서 재사용하거나 더 복잡한 신경망을 만드는 방법
tags:
- pytorch
- Module
image: /images/posts/06_cnn_pytorch/thumbnail.png
---

[project URL:https://github.com/woohaen88/PyTorchCNN/tree/main/training_convolutional_neural_network](https://github.com/woohaen88/PyTorchCNN/tree/main/training_convolutional_neural_network)

이번 포스트에서는 CNN학습을 어떻게 하고, 손으로 쓴 히라가나 글씨를 분류하는 것을 목표로한다. 일반적으로 알다시피 숫자형 데이터와는 다르지만 그렇다고 해서 PyTorch에서 구현하는 모델링을 하고 분류하는 방법자체가 많이 달라지지 않는다. 여전히 우리는 이러한 단계를 따른다.

1. 모델 아키텍쳐 정의
2. 데이터 로드(클라우드, 혹은 디스크)
3. epochs와 batches의 반복문을 실행한다.
4. 손실함수를 계산
5. 미분계수를 0으로 만들고, 역전파를 실행 그리고 나서 모델 파라미터를 업데이트한다.

데이터셋을 쉽게 다루게 해주는 <span style="background: #d9d9d9">DataLoader</span>를 로드한다. 이 <span style="background: #d9d9d9">DataLoader</span>는 PyTorch를 사용하는데 있어서 아주 중요한 skill이다. 

## 1. Training Convolutional Neural Network(CNN)

CNN을 구성하기 위해서는 필수 라이브러리가 필요하다. 바로 <span style="background: #d9d9d9">torch</span>와 <span style="background: #d9d9d9">torchvision</span>이다. 그리고 데이터분할을 손쉽게 다루는 <span style="background: #d9d9d9">scikit-learn</span>와 이미지 그래픽을 위한 <span style="background: #d9d9d9">opencv</span>또한 설치하도록 하자.

```bash
$ pip install torch torchvision
$ pip install opencv-contrib-python
$ pip install scikit-learn
$ pip install imutils
```
<br>


### The KMNIST dataset
<figure style="text-align: center">
<img src="/images/posts/06_cnn_pytorch/kmnist_dataset.png" width="100%">
	<span style="font-size: 0.8em; color:gray;" ><figcaption align="center">
		Figure 1: The KMNIST dataset
	</figcaption></span>
</figure>

KMNIST는 Kuzushiji-MNIST dataset의 줄임말이다. KMNIST dataset은 기존의 MNIST와 마찬가지로 70,000개의 이미지와 각각 상응하는 라벨이 있다.(60,000개의 훈련 이미지와 10,000개의 라벨 이미지)

KMNIST에는 10개의 클래스가 있고, 균등하게 분배되어 있다. 우리는 CNN을 이용해서 이 것을 60,000개의 이미지를 10개로 올바르게 분류하는게 목표다.

프로젝트의 구조는 다음과 같다.

```bash
$ tree . --dirsfirst
.
├── output
│   ├── model.pth
│   └── plot.png
├── ururuMllib
│   ├── __init__.py
│   └── lenet.py
├── predict.py
└── train.py
2 directories, 6 files
```


가장 먼저 살펴볼 script는 다음과 같다.
1. <span style="background: #d9d9d9">lenet.py</span> : 유명한 LeNet architecture
2. <span style="background: #d9d9d9">train.py</span> : KMNIST를 훈련하는 script
3. <span style="background: #d9d9d9">predict.py</span> : train model을 로드하고, 스크린에 결과를 보여준다.

`output` 디렉터리에는 `plot.png`(training/validation loss and accuracy)와 `model.pth`가 만들어진다.

## 2. Implementing a Convolutional Neural Network(CNN)
<figure style="text-align: center">
<img src="/images/posts/06_cnn_pytorch/Implementing_a_convnet.png" width="100%">
	<span style="font-size: 0.8em; color:gray;" ><figcaption align="center">
		Figure 2: The LeNet architecture
	</figcaption></span>
</figure>

Lenet 아키텍쳐는 다음과 같은 Layer 구조를 따른다.
<blockquote class="shadow-grey">
(CONV => RELU => POOL) * 2 => FC => RELU => FC => SOFTMAX
</blockquote>


CNN을 구현해보자. 첫번째로 먼저 필요한 라이브러리를 import 한다.
```python
# import the necessary packages
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from tqdm import tqdm
```

* <span class="shadow-grey">Module</span>: <span class="shadow-grey">Sequential</span>을 사용하기보다는 <span class="shadow-grey">Module</span> 의 서브클래스를 불러와서 사용한다.
* <span class="shadow-grey">Conv2d</span>: PyTorch의 CNN Layer
* <span class="shadow-grey">Linear</span>: Fully connected Layer
* <span class="shadow-grey">MaxPool2d</span>: 2D max-pooling을 적용
* <span class="shadow-grey">ReLU</span>: ReLU 활성화함수

```python
class LeNet(Module):
	def __init__(self, numChannels, classes):
		# call the parent constructor
		super(LeNet, self).__init__()

		# initialize first set of CONV => RELU => POOL layers
		self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
			kernel_size=(5, 5))
		self.relu1 = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

		# initialize second set of CONV => RELU => POOL layers
		self.conv2 = Conv2d(in_channels=20, out_channels=50,
			kernel_size=(5, 5))
		self.relu2 = ReLU()
		self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

		# initialize first (and only) set of FC => RELU layers
		self.fc1 = Linear(in_features=800, out_features=500)
		self.relu3 = ReLU()

		# initialize our softmax classifier
		self.fc2 = Linear(in_features=500, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)
```

이 코드블럭은 <span class="shadow-grey">LeNet</span> 클래스를 정의한다. 이렇게 <span class="shadow-grey">Module</span>을 상속해서 우리는 여러가지 이점을 얻을 수 있다.

* 변수 재활용
* 사용자 정의 함수나 subnetwork/components를 쉽게 만들 수 있다.
* 사용자 정의 함수로 <span class="shadow-grey">forward</span>를 만들 수 있다.

***이중 가장 좋은 점은 우리가 올바르게 모델 아키텍쳐를 정의하기만 하면 파이토치는 자동적으로 자동 미분과 역전파를 수행한다.*** 

<span class="shadow-grey">LeNet</span>클래스는 2가지 변수를 받는다.
1. <span class="shadow-grey">numChannels</span>: input이미지의 채널수(1: grayscale, 3: RGB)
2. <span class="shadow-grey">classes</span>: 중복이 되지 않는 데이터셋의 고유 클래스

위 코드는 <span class="shadow-grey">CONV => RELU => POOL</span> 레이어를 초기화한다. 첫 번재 CONV ㄹ이어는 총 5x5로 이루어진 20개의 필터를 학습한다. 그리고 나서 이미지차원을 줄이기 위해 ReLU 활성화함수를 적용하고 그 후에는 2x2 max-pooling 레이어와 2x2 stride를 적용한다.

그리고나서는 <span class="shadow-grey">CONV => RELU => POOL</span> 레이어를 또 한번 반복하는데 이때는 나머지는 그대로 두고 레이어 수가 50개로 증가한다.

그 다음 스텝으로 우리는 완전 연결 레이어를 추가한다. 이 때 input은 800, output은 500으로 한다.

마지막에는 클래스 함수에서 확률을 얻기 위해<span class="shadow-grey">LogSoftmax</span>를 적용한다.

이 시점에서 짚고 넘어갈 일은 초기화된 변수라는 것을 이해하는 것이 중요하다.이러한 변수는 본질적으로 <span class="shadow-grey">place holder</span>이다. PyTorch는 네트워크 아키텍처가 무엇인지 전혀 알지 못한다. 단지 일부 변수가 LeNet 클래스 정의 내에 존재한다는 것뿐이다.

네트워크 아키텍처 자체를 구축하려면(즉, 어떤 레이어가 다른 레이어에 입력되는지) Module 클래스의 forward 메서드를 재정의해야 한다.

<span class="shadow-grey">forward</span> 함수는 다음과 같은 목적을 가지고 있다.

1. 클래스의 생성자(예: \_\_init\_\_)에 정의된 변수에서 레이어/서브네트워크를 함께 연결한다.
2. 네트워크 아키텍처 자체를 정의한다.
3. 이를 통해 모델의 순방향 전달이 수행되어 결과적으로 출력 예측이 가능하다.
4. 그리고 PyTorch의 autograd 모듈 덕분에 자동 미분을 수행하고 모델 가중치를 업데이트할 수 있다.

<span class="shadow-grey">forward</span>는 코드로 구현하면 다음과 같다.

```python
	def forward(self, x):
		# pass the input through our first set of CONV => RELU =>
		# POOL layers
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)

		# pass the output from the previous layer through the second
		# set of CONV => RELU => POOL layers
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)

		# flatten the output from the previous layer and pass it
		# through our only set of FC => RELU layers
		x = flatten(x, 1)
		x = self.fc1(x)
		x = self.relu3(x)

		# pass the output to our softmax classifier to get our output
		# predictions
		x = self.fc2(x)
		output = self.logSoftmax(x)

		# return the output predictions
		return output
```

<span class="shadow-grey">forward</span>함수는 한개의 파라미터 <span class="shadow-grey">x</span>를 받아 들인다. 그리고 이 것은 network의 input데이터이다. 그리고 나서 <span class="shadow-grey">conv1</span>, <span class="shadow-grey">relu1</span>, <span class="shadow-grey">maxpool1</span>레이어를첫번째 등장하는 <span class="shadow-grey">CONV => RELU => POOL</span> 레이어에 연결한다.

<span class="shadow-grey">CONV => RELU => POOL</span> 레이어는 다차원 텐서다. 완전연결레이어에 연결하기 위해서 <span class="shadow-grey">flatten</span>메서드를 사용해야한다. 

이렇게 하기 위해서 우리는 <span class="shadow-grey">fc1</span>와 <span class="shadow-grey">relu3</span>레이어를 연결해야한다. 다음에는 <span class="shadow-grey">fc2</span>와 <span class="shadow-grey">logSoftmax</span>를 연결한다. <span class="shadow-grey">output</span>은 호출함수를 리턴한다.

### Creating our CNN training script with PyTorch

```python
# USAGE
# python train.py --model output/model.pth --plot output/plot.png

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from ururuMLlib.lenet import LeNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
```

<span class="shadow-grey">matplotlib</span>를 import하고 background engine을 "Agg"로 바꾼다.

command line arguments는 다음과 같다. 
```python
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to output trained model")
ap.add_argument("-p", "--plot", type=str, required=True,
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())
```

여기에서 우리는 2개의 arguments를 파싱한다.
1. <span class="shadow-grey">-\-model</span> : 모델이 저장될 경로
2. <span class="shadow-grey">-\-plot</span> : training history plot이 저장될 경로

모델을 학습하기 위한 파라미터를 정의한다.
```python
# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

# define the train and val splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

학습률, batch_size, epochf를 정의하고, train과 validation을 (4 : 1)로 분류한다.
데이터셋을 다운 받자. 코드는 다음과 같다.

```python
# load the KMNIST dataset
print("[INFO] loading the KMNIST dataset...")
trainData = KMNIST(root="data", train=True, download=True,
	transform=ToTensor())
testData = KMNIST(root="data", train=False, download=True,
	transform=ToTensor())

# calculate the train/validation split
print("[INFO] generating the train/validation split...")
numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValSamples = int(len(trainData) * VAL_SPLIT)
(trainData, valData) = random_split(trainData,
	[numTrainSamples, numValSamples],
	generator=torch.Generator().manual_seed(42))

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True,
	batch_size=BATCH_SIZE)
valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE
```

<span class="shadow-grey">DataLoader</span> object는 훈련에서는 <span class="shadow-grey">shuffle=True</span>, 테스트에서는 <span class="shadow-grey">shuffle=False</span>로 설정한다. 

이제 LeNet을 초기화 하자.

```python
# initialize the LeNet model
print("[INFO] initializing the LeNet model...")
model = LeNet(
	numChannels=1,
	classes=len(trainData.dataset.classes)).to(device)

# initialize our optimizer and loss function
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.NLLLoss()

# initialize a dictionary to store training history
H = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
}

# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()
```

위 코드는 우리의 <span class="shadow-grey">model</span>을ㄹ 초기화한다. KMNIST dataset이 grayscale이기 때문에 우리는 <span class="shadow-grey">numChannels=1</span>로 설정한다. 그리고 우리는 <span class="shadow-grey">datasest.classes</span>의 <span class="shadow-grey">classes</span>를 설정한다.


optimizer와 loss function을 초기화한다. 우리는 <span class="shadow-grey">Adam optimizer</span>와 <span class="shadow-grey">negative log-likelihood</span>를 사용한다. 그리고 이 것을 <span class="shadow-grey">nn.NLLoss</span>와 <span class="shadow-grey">LogSoftmax</span>와 연결한다. 

이제 모델 training을 해보자.
```python
# loop over our epochs
for e in tqdm(range(0, EPOCHS)):
	# set the model in training mode
	model.train()

	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0

	# initialize the number of correct predictions in the training
	# and validation step
	trainCorrect = 0
	valCorrect = 0

	# loop over the training set
	for (x, y) in trainDataLoader:
		# send the input to the device
		(x, y) = (x.to(device), y.to(device))

		# perform a forward pass and calculate the training loss
		pred = model(x)
		loss = lossFn(pred, y)

		# zero out the gradients, perform the backpropagation step,
		# and update the weights
		opt.zero_grad()
		loss.backward()
		opt.step()

		# add the loss to the total training loss so far and
		# calculate the number of correct predictions
		totalTrainLoss += loss
		trainCorrect += (pred.argmax(1) == y).type(
			torch.float).sum().item()
```

이제 우리는 다음과 같은 절차를 따른다.
1. model mode를 <span class="shadow-grey blue">train()</span> mode로 변경한다.
2. 우리의 training loss와 validation loss를 현재 epoch에서 초기화한다.
3. 현재 epoch에서 올바른 학습과 검증예측에대한 수를 초기화한다.

이게 완료되었으면 다음 스텝으로 넘어간다.
1. gradient를 0으로 만든다.
2. 역전파를 수행한다.
3. model weight를 업데이트한다.

***위 단계를 꼭 기억하자!*** 위 스텝을 정확히 지키지 않으면 심각한 오류를 초래한다. 


이제 우리의 모델을 검증데이터셋에서 평가한다.

```python
# switch off autograd for evaluation
with torch.no_grad():
    # set the model in evaluation mode
    model.eval()

    # loop over the validation set
    for (x, y) in valDataLoader:
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))

        # make the predictions and calculate the validation loss
        pred = model(x)
        totalValLoss += lossFn(pred, y)

        # calculate the number of correct predictions
        valCorrect += (pred.argmax(1) == y).type(
            torch.float).sum().item()
```

검증데이터셋에서 PyTorch 모델을 평가할 때는 다음을 명시하자.
1. <span class="shadow-grey blue">torch.no_grad()</span>: 자동미분을 끈다.
2. model를 평가모드로 바꾼다. <span class="shadow-grey blue">model.eval()</span>

```python
# calculate the average training and validation loss
avgTrainLoss = totalTrainLoss / trainSteps
avgValLoss = totalValLoss / valSteps

# calculate the training and validation accuracy
trainCorrect = trainCorrect / len(trainDataLoader.dataset)
valCorrect = valCorrect / len(valDataLoader.dataset)

# update our training history
H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
H["train_acc"].append(trainCorrect)
H["val_loss"].append(avgValLoss.cpu().detach().numpy())
H["val_acc"].append(valCorrect)

# print the model training and validation information
print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
    avgTrainLoss, trainCorrect))
print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
    avgValLoss, valCorrect))
```

위 코드블럭은 training과 validation loss의 평균을 계산한다. 훈련은 완료되었다.
```python
# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

# we can now evaluate the network on the test set
print("[INFO] evaluating network...")

# turn off autograd for testing evaluation
with torch.no_grad():
	# set the model in evaluation mode
	model.eval()
	
	# initialize a list to store our predictions
	preds = []

	# loop over the test set
	for (x, y) in testDataLoader:
		# send the input to the device
		x = x.to(device)

		# make the predictions and add them to the list
		pred = model(x)
		preds.extend(pred.argmax(axis=1).cpu().numpy())

# generate a classification report
print(classification_report(testData.targets.cpu().numpy(),
	np.array(preds), target_names=testData.classes))
```

이제 훈련 타이머를 멈추고 훈련 시간을 보여준다. 그리고 그런 다음 <span class="shadow-grey blue">torch.no_grad()</span>컨텍스트를 설정하고 model을 <span class="shadow-grey blue">eval()</span>로 변경한다.

```python
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# serialize the model to disk
torch.save(model, args["model"])
```

훈련 기록에 대한 matplotlib 그림을 생성한다.

그런 다음 PyTorch 모델 가중치를 디스크에 저장하기 위해 <span class="shadow-grey">torch.save</span>를 호출하여 디스크에서 로드하고 별도의 Python 스크립트에서 예측할 수 있다.

전체적으로 이 스크립트를 검토하면 PyTorch가 훈련 루프에 대해 얼마나 더 많은 제어를 제공하는지 알 수 있다. 

훈련 루프를 완전히 제어하고 사용자 지정 절차를 구현해야 하는 경우에 좋다.

### Training our CNN with PyTorch

```bash
[INFO] loading the KMNIST dataset...
[INFO] generating the train/validation split...
[INFO] initializing the LeNet model...
[INFO] training the network...
[INFO] EPOCH: 1/10
Train loss: 0.367026, Train accuracy: 0.8862
Val loss: 0.162807, Val accuracy: 0.9512

[INFO] EPOCH: 2/10
Train loss: 0.100468, Train accuracy: 0.9694
Val loss: 0.107167, Val accuracy: 0.9687

[INFO] EPOCH: 3/10
Train loss: 0.060439, Train accuracy: 0.9814
Val loss: 0.072758, Val accuracy: 0.9791

[INFO] EPOCH: 4/10
Train loss: 0.038513, Train accuracy: 0.9881
Val loss: 0.069339, Val accuracy: 0.9810

[INFO] EPOCH: 5/10
Train loss: 0.025257, Train accuracy: 0.9922
Val loss: 0.088133, Val accuracy: 0.9760

[INFO] EPOCH: 6/10
Train loss: 0.020032, Train accuracy: 0.9935
Val loss: 0.086066, Val accuracy: 0.9787

[INFO] EPOCH: 7/10
Train loss: 0.016911, Train accuracy: 0.9946
Val loss: 0.093041, Val accuracy: 0.9776

[INFO] EPOCH: 8/10
Train loss: 0.013512, Train accuracy: 0.9958
Val loss: 0.086559, Val accuracy: 0.9789

[INFO] EPOCH: 9/10
Train loss: 0.012416, Train accuracy: 0.9961
Val loss: 0.104030, Val accuracy: 0.9773

[INFO] EPOCH: 10/10
Train loss: 0.010962, Train accuracy: 0.9966
Val loss: 0.094252, Val accuracy: 0.9801

[INFO] total time taken to train the model: 294.22s
[INFO] evaluating network...
              precision    recall  f1-score   support

           o       0.92      0.96      0.94      1000
          ki       0.96      0.94      0.95      1000
          su       0.95      0.88      0.91      1000
         tsu       0.97      0.97      0.97      1000
          na       0.96      0.92      0.94      1000
          ha       0.94      0.93      0.93      1000
          ma       0.91      0.97      0.94      1000
          ya       0.93      0.97      0.95      1000
          re       0.98      0.98      0.98      1000
          wo       0.96      0.96      0.96      1000

    accuracy                           0.95     10000
   macro avg       0.95      0.95      0.95     10000
weighted avg       0.95      0.95      0.95     10000
```

<figure style="text-align: center">
<img src="/images/posts/06_cnn_pytorch/plot.png" width="100%">
	<span style="font-size: 0.8em; color:gray;" ><figcaption align="center">
		Figure 3: Plotting our training history with PyTorch
	</figcaption></span>
</figure>

CPU에서는 ≈160초 정도 걸리고 GPU에서는 ≈82초 걸린다.  

epoch의 마지막에서 우리는 99.67%의 정확도를 얻었다. 그리고 테스트셋에서는 98.01%의 결과를 얻었다.

우리의 얕은 모델 구조(그러나 PyTorch에서 CNN을 사용하기에는 VGG나 ResNet과 같은 모델을 사용하면 더 높은 정확도를 얻을 수 있지만 이런 모델은 더 복잡하다)에서 테스트셋이 정확도가 ≈95% 정도 도달하게 되면 꽤 좋은 결과다. 

그러나, 위 그림에서 보듯이 우리의 훈련 그래프는 꽤 smooth하다. 그리고 과대적합이 일어났는지를 입증해야 한다

```bash
$ ls output/
model.pth	plot.png
```
<span class="shadow-grey">model.pth</span>파일은 우리가 훈련한 모델이고 disk에 저장된다. 그리고 예측할 때는 이 모델을 불러와서 사용한다.

## Implementing our PyTorch prediction script

```python
# set the numpy seed for better reproducibility
import numpy as np
np.random.seed(42)

# import the necessary packages
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import argparse
import imutils
import torch
import cv2
```

필요한 라이브러리를 로드하고 코드의 재사용성을 위해 seed를 설정한다.
* <span class="shadow-grey">DataLoader</span> : 우리의 KMNIST test 데이터를 사용한다.  
* <span class="shadow-grey">Subset</span> : testing data  
* <span class="shadow-grey">ToTensor</span> : PyTorch tensor로 변환한다.  
* <span class="shadow-grey">KMNIST</span> : The Kuzushiji-MNIST dataset  
* <span class="shadow-grey">cv2</span> : display를 위한 라이브러리  

command line arguments를 parsing한다.
```python
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to the trained PyTorch model")
args = vars(ap.parse_args())
```

여기서는 우리는 1개의 argument만 사용한다. 
<span class="shadow-grey">-\-model</span> : 우리가 훈련한 모델 객체이다.

```python
# set the device we will be using to test the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the KMNIST dataset and randomly grab 10 data points
print("[INFO] loading the KMNIST test dataset...")
testData = KMNIST(root="data", train=False, download=True,
	transform=ToTensor())
idxs = np.random.choice(range(0, len(testData)), size=(10,))
testData = Subset(testData, idxs)

# initialize the test data loader
testDataLoader = DataLoader(testData, batch_size=1)

# load the model and set it to evaluation mode
model = torch.load(args["model"]).to(device)
model.eval()
```
KMNIST 데이터 세트에서 테스트 데이터를 로드한다. 우리는 Subset 클래스를 사용하여 이 데이터 세트의 총 10개의 이미지를 무작위로 샘플링한다.

모델을 통해 테스트 데이터의 하위 집합을 전달하기 위해 DataLoader가 생성한다.

그런 다음 디스크에서 PyTorch 모델을 로드하여 위세서 선언한 <span class="shadow-grey">devicec</span>로 전달한다. 마지막으로 모델은 평가 모드로 전환한다.

이제 테스트 세트의 샘플에 대해 예측해 보자.

```python
# switch off autograd
with torch.no_grad():
	# loop over the test set
	for (image, label) in testDataLoader:
		# grab the original image and ground truth label
		origImage = image.numpy().squeeze(axis=(0, 1))
		gtLabel = testData.dataset.classes[label.numpy()[0]]
        
		# send the input to the device and make predictions on it
		image = image.to(device)
		pred = model(image)
        
		# find the class label index with the largest corresponding
		# probability
		idx = pred.argmax(axis=1).cpu().numpy()[0]
		predLabel = testData.dataset.classes[idx]
```

그래디언트 추적을 끄고 테스트 세트의 하위 집합에 있는 모든 이미지를 반복한다.

그리고 각 이미지에 대해 다음을 수행한다.

1. 현재 이미지를 가져와 NumPy 배열로 변환(나중에 OpenCV로 그릴 수 있도록)
2. 실측 클래스 레이블을 추출
3. <span class="shadow-grey">image</span>를 적절한 <span class="shadow-grey">device</span>로 보냄
4. 훈련된 LeNet 모델을 사용하여 현재 <span class="shadow-grey">image</span>를 예측
5. 예측 확률이 가장 높은 클래스 레이블을 추출

남은 것은 약간의 시각화정도이다.

```python
		# convert the image from grayscale to RGB (so we can draw on
		# it) and resize it (so we can more easily see it on our
		# screen)
		origImage = np.dstack([origImage] * 3)
		origImage = imutils.resize(origImage, width=128)
		# draw the predicted class label on it
		color = (0, 255, 0) if gtLabel == predLabel else (0, 0, 255)
		cv2.putText(origImage, gtLabel, (2, 25),
			cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
		# display the result in terminal and show the input image
		print("[INFO] ground truth label: {}, predicted label: {}".format(
			gtLabel, predLabel))
		cv2.imshow("image", origImage)
		cv2.waitKey(0)
```

KMNIST 데이터 세트의 각 이미지는 단일 채널 회색조 이미지입니다. 그러나 OpenCV의 <span class="shadow-grey">cv2.putText</span> 함수를 사용하여 <span class="shadow-grey">image</span>에 예측된 클래스 레이블과 정답 레이블을 그려보자.

회색조 이미지에 RGB 색상을 그리려면 먼저 회색조 이미지를 깊이별로 총 3번 쌓아서 회색조 이미지의 RGB 표현을 만들어야 한다.

또한 화면에서 더 쉽게 볼 수 있도록 <span class="shadow-grey">origImage</span>의 크기를 조정한다.(기본적으로 KMNIST 이미지는 28×28 픽셀에 불과하므로 특히 고해상도 모니터에서 보기 어려울 수 있음).

여기에서 텍스트 <span class="shadow-grey">color</span>을 결정하고 출력 이미지에 레이블을 그린다.

화면에 출력 <span class="shadow-grey">origImage</span>를 표시하여 스크립트를 마무리한다.

### Making predictions with our trained PyTorch model

이제 훈련된 PyTorch 모델을 사용하여 예측할 준비가 되었다.

```bash
[INFO] loading the KMNIST test dataset...
[INFO] ground truth label: ki, predicted label: ki
[INFO] ground truth label: ki, predicted label: ki
[INFO] ground truth label: ki, predicted label: ki
[INFO] ground truth label: ha, predicted label: ha
[INFO] ground truth label: tsu, predicted label: tsu
[INFO] ground truth label: ya, predicted label: ya
[INFO] ground truth label: tsu, predicted label: tsu
[INFO] ground truth label: na, predicted label: na
[INFO] ground truth label: ki, predicted label: ki
[INFO] ground truth label: tsu, predicted label: tsu

```

<figure style="text-align: center">
<img src="/images/posts/06_cnn_pytorch/predict_handwritten_characters.png" width="100%">
	<span style="font-size: 0.8em; color:gray;" ><figcaption align="center">
		Figure 4: Making predictions on handwritten characters using PyTorch and our trained CNN
	</figcaption></span>
</figure>

출력에서 알 수 있듯이 PyTorch 모델을 사용하여 각 히라가나 문자를 성공적으로 인식할 수 있다.