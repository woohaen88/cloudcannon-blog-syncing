---
date: 2021-12-31 00:00:00
title: Intro to PyTorch-Training your first neural network using PyTorch
description: 객체 지향 프로그래밍에서 자체 신경망 계층을 만들어서 재사용하거나 더 복잡한 신경망을 만드는 방법
tags:
- pytorch
- Module
image: /images/posts/03_modulization/thumbnail.png
---

<!-- 포스트 이미지 폴더: /images/posts/02_dataset_and_dataloader/ -->

```bash
$ tree . --dirsfirst
.
├── ururuMllib
│   └── mlp.py
└── train.py
1 directory, 2 files
```

`mlp.py` 파일은 기본 MLP(다층 퍼셉트론) 구현을 저장한다.
그런 다음 데이터 세트에서 MLP를 훈련하는 데 사용할 `train.py`를 구현한다.

<figure style="text-align: center">
<img src="/images/posts/05_Intro_to_PyTorch/implementing_neuralnet_pytorch.png" width="70%">
	<span style="font-size: 0.8em; color:gray;" ><figcaption align="center">
		Figure 1: Implementing a basic multi-layer perceptron with PyTorch.
	</figcaption></span>
</figure>

이제 PyTorch로 첫 번째 신경망을 구현할 준비가 되었다.
이 네트워크는 다층 퍼셉트론(MLP)이라고 하는 매우 단순한 피드포워드 신경망입니다(하나 이상의 은닉층이 있음을 의미). 

```python
# import the necessary packages
from collections import OrderedDict
import torch.nn as nn

def get_training_model(inFeatures=4, hiddenDim=8, nbClasses=3):
	# construct a shallow, sequential neural network
	mlpModel = nn.Sequential(OrderedDict([
		("hidden_layer_1", nn.Linear(inFeatures, hiddenDim)),
		("activation_1", nn.ReLU()),
		("output_layer", nn.Linear(hiddenDim, nbClasses))
	]))

	# return the sequential model
	return mlpModel
```

* `OrderedDict`: A dictionary object - 객체가 추가된 순서를 기억한다. 우리는 이 정렬된 dictionary를 사용하여 네트워크의 각 계층에 사람이 읽을 수 있는 이름을 제공한다.
* `nn`: PyTorch’s neural network implementations, 그런 다음 세 개의 매개변수를 허용하는 `get_training_model` 함수를 정의한다. 

정의된 파라미터는 다음과 같다.
1. 신경망에 대한 입력 노드의 수
2. 네트워크의 은닉층에 있는 노드의 수
3. 출력 노드의 수

제공된 디폴트 값에 의하면 4-8-3 뉴럴넷은 Input 레이어는 4, Hidden 레이어는 8, Output 레이어는 3으로 구성되어있다.


The actual neural network architecture is then constructed on Lines 7-11 by first initializing a nn.Sequential object (very similar to Keras/TensorFlow’s Sequential class).

Inside the Sequential class we build an OrderedDict where each entry in the dictionary consists of two values:

A string containing the human-readable name for the layer (which is very useful when debugging neural network architectures using PyTorch)
The PyTorch layer definition itself
The Linear class is our fully connected layer definition, meaning that each of the inputs connects to each of the outputs in the layer. The Linear class accepts two required arguments:

The number of inputs to the layer
The number of outputs
On Line 8, we define hidden_layer_1 which consists of a fully connected layer accepting inFeatures (4) inputs and then producing an output of hiddenDim (8).

From there, we apply a ReLU activation function (Line 9) followed by another Linear layer which serves as our output (Line 10).

Notice that the second Linear definition contains the same number of inputs as the previous Linear layer did outputs — this is not by accident! The output dimensions of the previous layer must match the input dimensions of the next layer, otherwise PyTorch will error out (and then you’ll have the quite tedious task of debugging the layer dimensions yourself).

PyTorch is not as forgiving in this regard (as opposed to Keras/TensorFlow), so be extra cautious when specifying your layer dimensions.

The resulting PyTorch neural network is then returned to the calling function.

