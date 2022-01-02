---
date: 2021-12-29 00:00:00
title: 이미지 처리와 합성곱 신경망
description: 작성
tags:
- pytorch
- Module
image: /images/posts/04_image_handling/thumbnail.png
---

<!-- 포스트 이미지 폴더: /images/posts/02_dataset_and_dataloader/ -->
<h1>이미지 처리와 합성곱 신경망</h1>

2012년에 ILSVRC라는 이미지 인식 대회에서 다층 CNN을 사용한 모델이 다른 모델들을 압도하면서 딥러닝이 주목받게 된 기폭제가 됐다. 현재도 딥러닝이 가장 활발히 사용되는 분야가 컴퓨터 비전 분야로 특히, CNN이 활발히 연구되고 있다. 

## 1. 이미지와 합성곱 계산

이미지 분야에서 합성곱(Convolution)이란, 이미지 위에 작은 커널(필터)을 이동시켜 가면서 각 요소(픽셀)의 곱(또는 합이나 평균 등)을 계산해 가는 방식이다. 

## 2. CNN을 사용한 이미지 분류

CNN을 사용한 이미지 분류는 기본적으로 합성곱으로부터 ReLU 등의 활성화 함수를 적용하는 과정을 여러 번 실시한다. 이미지 데이터는 (C, H, W) 형식으로 H, W는 이미지의 세로, 가로 크기이며, C는 색수 또는 채널(channel)이라고도 불린다. 원래 C=1이나 C=3이지만, 최종적으로는 <span style="color:crimson;">마지막 합성공 계층의 커널 수</span>가 된다. 합성곱 처리 후에는 위치 감도를 높이는 풀링(pooling)을 적용하거나, Dropout, Batch Normalization을 함께 사용하는 경우도 많다. 

### 2.1 Fashion-Mnist

MNIST는 28X28픽셀의 흑백 손글씨 숫자 데이터이다. 이 데이터는 이미지 분류 시에 사용되는 머신러닝을 입문한다면 제일 처음에 만나게되는 대표적인 예제 데이터이다. 최근에는 이 MNIST가 너무 간단하다는 지적도 있어서 10가지 분류의 옷 및 엑세서리를 이미지 데이터로 이루어진 <span style="color:darkblue;"><b>Fashion-MNIST</b></span>

<figure>
<img src="/images/posts/04_image_handling/fashion_mnist.png">
	<span style="font-size: 0.8em; color:gray;"><figcaption align="center">
		Figure 1: Fashion_MNIST
	</figcaption></span>
</figure>

파이토치의 확장 기능인 torchvision이라는 라이브러리를 사용하면 Fashion-MNIST 데이터 다운로드부터, Dataset으로 변환, DataLoader 작성까지 할 수 있다.

