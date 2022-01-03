---
date: 2022-01-01 00:02:00
title: PyTorch image classification with pre-trained networks
description: pytorch에서 KMNIST를 불러와서 이미지를 불러오고, opencv를 이용하여 간단한 시각화를 해본다.
tags:
  - pytorch
  - pretrained
  - CNN
image: /images/posts/07_CNN-image_classification_pretrained/thumbnail.png
---
이번 포스트에서는 PyTorch를 사용하여 사전 훈련된 네트워크로 이미지 분류를 수행하는 방법을 알아본다. 이러한 네트워크를 활용하면 몇 줄의 코드로 공통 개체 범주를 정확하게 분류할 수 있다.

# PyTorch image classification with pre-trained networks

PyTorch 라이브러리에 내장된 네트워크를 포함하여 사전 훈련된 이미지 분류 네트워크가 무엇인지 알아본다.

## 1. What are pre-trained image classification networks?

<figure style="text-align: center">
<img src="/images/posts/07_CNN-image_classification_pretrained/01.webp" width="100%" />
	<span style="font-size: 0.8em; color:gray;"><figcaption align="center">
		Figure 1: 가장 널리 사용되는 최첨단 신경망은 ImageNet 데이터 세트에서 사전 훈련된 가중치와 함께 제공된다.
	</figcaption></span>
</figure>

이미지 분류와 관련하여 [ImageNet](https://www.image-net.org/)보다 더 유명한 데이터 세트/챌린지는 가히 없다고 봐도 무방하다. **ImageNet의 목표는 입력 이미지를 컴퓨터 비전 시스템이 일상 생활에서 "보는" 1,000개의 공통 개체 범주 집합으로 정확하게 분류하는 것이다.**

PyTorch, Keras, TensorFlow, fast.ai 등을 포함하여 가장 널리 사용되는 딥러닝 프레임워크에는 *사전 훈련된 네트워크*가 포함된다. 이것은 컴퓨터 비전 연구원이 ImageNet 데이터 세트에 대해 훈련한 매우 정확한 최신 모델이다.

ImageNet에 대한 훈련이 완료된 후 연구원은 모델을 디스크에 저장한 다음 다른 연구자, 학생 및 개발자가 자신의 프로젝트에서 배우고 사용할 수 있도록 자유롭게 게시했다.

이번 포스팅에서는 PyTorch를 사용하여 다음과 같은 최첨단 분류 네트워크를 사용하여 입력 이미지를 분류하는 방법을 알아보자.

* VGG16
* VGG19
* Inception
* DenseNet
* ResNet

## 2. Configuring your development environment

PyTorch와 OpenCV는 모두 pip를 사용하여 매우 쉽게 설치할 수 있다.

```python
$ pip install torch torchvision
$ pip install opencv-contrib-python
```
<br>
## 3. Project structure
PyTorch로 이미지 분류를 구현하기 전에 먼저 프로젝트 디렉토리 구조를 확인해보자.

```txt
$ tree . --dirsfirst
.
├── images
│   ├── bmw.png
│   ├── boat.png
│   ├── clint_eastwood.jpg
│   ├── jemma.png
│   ├── office.png
│   ├── scotch.png
│   ├── soccer_ball.jpg
│   └── tv.png
├── ururuMLlib
│   └── config.py
├── classify_image.py
└── ilsvrc2012_wordnet_lemmas.txt
```
<span class="shadow-grey">ururuMLlib</span> 모듈 내부에는 <span class="shadow-grey">config.py</span> 라는 단일 파일이 있다. 이 파일은 다음과 같은 구성을 저장한다.

* 입력 이미지 크기
* 스케일링에 대한 평균 및 표준 편차
* 훈련에 GPU를 사용하는지 여부
* ImageNet 클래스 레이블 경로(i.e.,: <span class="shadow-grey">ilsvrc2012_wordnet_lemmas.txt</span>)
<span class="shadow-grey">classify_image.py</span> 스크립트는 <span class="shadow-grey">config</span>를 로드한 다음 VGG16, VGG19, Inception, DenseNet 또는 ResNet을 사용하여 입력이미지를 분류한다.(명령줄 인수로 제공하는 모델 아키텍처에 따라 다름)

<span class="shadow-grey">images</span> 디렉토리에는 이러한 이미지 분류 네트워크를 적용할 샘플 이미지가 있다.

## 4. Creating our configuration file

```python
# import the necessary packages
import torch

# specify image dimension
IMAGE_SIZE = 224

# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# determine the device we will be using for inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# specify path to the ImageNet labels
IN_LABELS = "ilsvrc2012_wordnet_lemmas.txt"
```
위 코드에서는 입력 이미지 공간 치수를 정의한다. 즉, 각 이미지는 분류를 위해 사전 훈련된 PyTorch 네트워크를 통과하기 전에 *224x224* 픽셀로 크기가 조정된다.

<span class="note">Note: ImageNet 데이터 세트에서 훈련된 대부분의 네트워크는 *224x224* 또는 *227x227* 이미지를 허용한다. 일부 네트워크, 특히 컨볼루션 네트워크는 더 큰 이미지 차원을 허용한다.<span>

여기에서 우리는 훈련 세트에서 RGB 픽셀 강도의 평균과 표준 편차를 정의한다. 분류를 위해 네트워크를 통해 입력 이미지를 전달하기 전에 먼저 평균을 뺀 다음 표준 편차로 나누어 이미지 픽셀 강도를 조정한다(표준화). 이 전처리는 ImageNet과 같은 크고 다양한 이미지 데이터 세트에서 훈련된 CNN에 일반적이다.

훈련에 CPU를 사용할지 GPU를 사용할지 지정하고 ImageNet 클래스 레이블의 입력 텍스트 파일에 대한 경로를 정의한다.

```bash
tench, Tinca_tinca
goldfish, Carassius_auratus
...
bolete
ear, spike, capitulum
toilet_tissue, toilet_paper, bathroom_tissue
```

<br>

### 5. Implementing our image classification script

구성 파일을 처리한 상태에서 사전 훈련된 PyTorch 네트워크를 사용하여 입력 이미지를 분류하는 데 사용되는 기본 드라이버 스크립트를 구현해 보겠습니다.

```python
# import the necessary packages
from pyimagesearch import config
from torchvision import models
import numpy as np
import argparse
import torch
import cv2
```

* <span class="shadow-grey">config</span>:  구성 파일
* <span class="shadow-grey">models</span>: PyTorch의 사전 훈련된 신경망 포함
* <span class="shadow-grey">numpy</span>: 숫자 배열 처리
* <span class="shadow-grey">torch</span>: PyTorch API에 액세스
* <span class="shadow-grey">cv2</span>: OpenCV 바인딩

이미지를 가져오기를 처리한 상태에서 입력 이미지를 수락하고 사전 처리하는 함수를 정의하자.

```python
def preprocess_image(image):
	# swap the color channels from BGR to RGB, resize it, and scale
	# the pixel values to [0, 1] range
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
	image = image.astype("float32") / 255.0

	# subtract ImageNet mean, divide by ImageNet standard deviation,
	# set "channels first" ordering, and add a batch dimension
	image -= config.MEAN
	image /= config.STD
	image = np.transpose(image, (2, 0, 1))
	image = np.expand_dims(image, 0)
	
	# return the preprocessed image
	return image
```
<span class="shadow-grey">preprocess_image</span>함수는 분류를 위해 사전 처리할 이미지인 <span class="shadow-grey">image</span> 라는 단일 인수를 사용한다.

다음과 같이 전처리 작업을 시작한다.

1. BGR에서 RGB 채널 순서로 스와핑(여기에서 사용하는 사전 훈련된 네트워크는 RGB 채널 순서를 사용하는 반면 OpenCV는 기본적으로 BGR 순서를 사용함)
2. 종횡비를 무시하고 이미지 크기를 고정 치수(예: 224×224)로 조정
3. 이미지를 부동 소수점 데이터 유형으로 변환한 다음 픽셀을 [0, 1] 범위로 조정

여기에서 두 번째 전처리 작업 세트를 수행한다.
그런 다음 전처리된 <span class="shadow-grey">image</span>가 호출 함수로 리턴된다.

```python
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-m", "--model", type=str, default="vgg16",
	choices=["vgg16", "vgg19", "inception", "densenet", "resnet"],
	help="name of pre-trained network to use")
args = vars(ap.parse_args())
```

두 개의 명령줄 인수가 있다.

1. <span class="shadow-grey">-\-image</span>: 분류하려는 입력 이미지의 경로
2. <span class="shadow-grey">-\-model</span>: 이미지를 분류하는 데 사용할 사전 훈련된 CNN 모델

이제 <span class="shadow-grey">-\-model</span> 명령줄 인수의 이름을 해당 PyTorch 함수에 매핑하는 <span class="shadow-grey">MODELS</span> dictionary를 정의하자.

```python
# define a dictionary that maps model names to their classes
# inside torchvision
MODELS = {
	"vgg16": models.vgg16(pretrained=True),
	"vgg19": models.vgg19(pretrained=True),
	"inception": models.inception_v3(pretrained=True),
	"densenet": models.densenet121(pretrained=True),
	"resnet": models.resnet50(pretrained=True)
}

# load our the network weights from disk, flash it to the current
# device, and set it to evaluation mode
print("[INFO] loading {}...".format(args["model"]))
model = MODELS[args["model"]].to(config.DEVICE)
model.eval()
```
<span class="shadow-grey">MODEL</span> dictionary를 생성한다.

* dictionary의 *핵심*은 human-readable 모델 이름이며 <span class="shadow-grey">-\-model</span> 명령줄 인수를 통해 전달된다.
* dictionary에 대한 *값*은 ImageNet에서 사전 훈련된 가중치로 모델을 로드하는 데 사용되는 해당 PyTorch 함수이다.

다음 사전 훈련된 모델을 사용하여 PyTorch로 입력 이미지를 분류할 수 있다.

1. VGG16
2. VGG19
3. Inception
4. DenseNet
4. ResNet

<span class="shadow-grey">pretrained=True</span> 플래그를 지정하면 PyTorch가 모델 아키텍처 정의를 로드할 뿐만 아니라 모델에 대해 사전 훈련된 ImageNet 가중치도 다운로드하도록 지시한다.

모델과 사전 훈련된 가중치를 로드한 다음(모델 가중치를 다운로드한 적이 없는 경우 자동으로 다운로드되어 캐시된다.) 그런 다음 <span class="shadow-grey">DEVICE</span>에 따라 CPU 또는 GPU에서 실행되도록 모델을 설정한다.

<span class="shadow-grey">model</span>을 평가 모드로 전환하여 PyTorch가 훈련 중 처리하는 방식과 다른 특수 계층(예: dropout 및 배치 정규화)을 처리하도록 지시한다. ***예측을 하기 전에 모델을 평가 모드로 전환하는 것이 중요하다.***

이제 모델이 로드되었으므로 입력 이미지가 필요하다.

```python
# load the image from disk, clone it (so we can draw on it later),
# and preprocess it
print("[INFO] loading image...")
image = cv2.imread(args["image"])
orig = image.copy()
image = preprocess_image(image)

# convert the preprocessed image to a torch tensor and flash it to
# the current device
image = torch.from_numpy(image)
image = image.to(config.DEVICE)

# load the preprocessed the ImageNet labels
print("[INFO] loading ImageNet labels...")
imagenetLabels = dict(enumerate(open(config.IN_LABELS)))
```
디스크에서 입력 <span class="shadow-grey">image</span>를 로드한다. 이 것을 복사하고 네트워크의 상위 예측을 시각화할 수 있도록 한다. 또한 위에서 설정한 <span class="shadow-grey">preprocess_image</span>함수를 사용하여 크기 조정 및 스케일링을 수행한다.

마지막으로 디스크에서 입력 ImageNet 클래스 레이블을 로드한다.

이제 <span class="shadow-grey">model</span>을 사용하여 입력 <span class="shadow-grey">image</span>를 예측할 준비가 되었다.
```python
# classify the image and extract the predictions
print("[INFO] classifying image with '{}'...".format(args["model"]))
logits = model(image)
probabilities = torch.nn.Softmax(dim=-1)(logits)
sortedProba = torch.argsort(probabilities, dim=-1, descending=True)

# loop over the predictions and display the rank-5 predictions and
# corresponding probabilities to our terminal
for (i, idx) in enumerate(sortedProba[0, :5]):
	print("{}. {}: {:.2f}%".format
		(i, imagenetLabels[idx.item()].strip(),
		probabilities[0, idx.item()] * 100))
```

위 코드는 네트워크의 순방향 전달을 수행하여 네트워크의 출력을 만든다. 이것을 <span class="shadow-grey">Softmax</span> 함수를 통해 전달하여 <span class="shadow-grey">model</span>이 훈련된 가능한 1,000개의 클래스 레이블 각각에 대한 예측 확률을 얻는다.

그런 다음 목록의 맨 앞에 더 높은 확률로 내림차순으로 확률을 정렬한다. 그런 다음 다음과 같이 터미널에 상위 5개 예측 클래스 레이블과 해당 확률을 표시하도록 한다.

* 상위 5개 예측에 대한 반복
* <span class="shadow-grey">imagenetLabels</span> 사전을 사용하여 클래스 레이블 이름 찾기
* 예측 확률 표시

최종 코드 블록은 출력 이미지에 상위 1개(즉, 상위 예측 레이블)를 그린다.

```python
# draw the top prediction on the image and display the image to
# our screen
(label, prob) = (imagenetLabels[probabilities.argmax().item()],
	probabilities.max().item())
cv2.putText(orig, "Label: {}, {:.2f}%".format(label.strip(), prob * 100),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)
```
### Image classification with PyTorch results

```bash
$ python classify_image.py --image images/boat.png
[INFO] loading vgg16...
[INFO] loading image...
[INFO] loading ImageNet labels...
[INFO] classifying image with 'vgg16'...
0. wreck: 99.99%
1. seashore, coast, seacoast, sea-coast: 0.01%
2. pirate, pirate_ship: 0.00%
3. breakwater, groin, groyne, mole, bulwark, seawall, jetty: 0.00%
4. sea_lion: 0.00%
```

<br> 

### The KMNIST dataset

<figure style="text-align: center">
<img src="/images/posts/07_CNN-image_classification_pretrained/wreck.png" width="100%" />
	<span style="font-size: 0.8em; color:gray;"><figcaption align="center">
		Figure 2: Using PyTorch and VGG16 to classify an input image.
	</figcaption></span>
</figure>

확실히 VGG16 네트워크는 99.99% 확률로 입력 이미지를 "난파선"으로 올바르게 분류할 수 있다.

"해변"이 모델에서 두 번째로 높은 예측이라는 것도 정확하다. 이번에는 DenseNet 모델을 사용하여 다른 이미지를 사용해 보겠다.

```python
$ python classify_image.py --image images/bmw.png --model densenet
[INFO] loading densenet...
[INFO] loading image...
[INFO] loading ImageNet labels...
[INFO] classifying image with 'densenet'...
0. convertible: 96.61%
1. sports_car, sport_car: 2.25%
2. car_wheel: 0.45%
3. beach_wagon, station_wagon, wagon, estate_car, beach_waggon, station_waggon, waggon: 0.22%
4. racer, race_car, racing_car: 0.13%
```

<figure style="text-align: center">
	<img src="/images/posts/07_CNN-image_classification_pretrained/convertible.png" width="100%" />
	<span style="font-size: 0.8em; color:gray;">
		<figcaption align="center">
			Figure 3: Applying DenseNet and PyTorch to classify an image.
		</figcaption>
	</span>
</figure>

DenseNet의 최고 예측은 96.61%의 정확도로 "contertible"이다. 두 번째 상위 예측인 "sports_car"도 정확하다.