---
date: 2021-12-13-Fine_Tuning
title: Fine_Tuning
subtitle: 작물 잎 사진으로 질병 분류하기(2)
image: /images/projects/Fine_Tuning/landing.png
---

# 데이터 증식을 사용한 특성 추출

이 방법은 훨씬 느리고 비용이 많이 들지만 데이터 증식 기법을 사용할 수 있다. conv_base 모델을 확장하고 비용이 많이 들지만 훈련하는 동안 데이터 증식 기법을 사용할 수 있다. conv_base 모델을
확장하고 입력 데이터를 사용하여 End-to-End로 실행한다.

<p style="font-size: 0.8rem;"><em>Note :이 기법은 연산 비용이 크기 때문에 GPU를 사용할 수 있을 때 시도하는게 좋다.</em></p>

이번에는 `VGG16`모델을 불러와서 진행한다.


---
parameter load
---

```python
from tensorflow.keras.applications import VGG16

conv_base = VGG16(weights="imagenet",
                  include_top=False,
                  input_shape=(150, 150, 3))
```

모델은 층과 동일하게 작동하므로 층을 추가하듯이 `Sequential` 모델에 다른 모델을 추가한다.

이 모델의 구조는 다음과 같다.
```bash
>>> model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type) Output Shape Param #
=================================================================
vgg16 (Functional) (None, 4, 4, 512) 14714688

flatten (Flatten) (None, 8192) 0

dense (Dense) (None, 256) 2097408

dense_1 (Dense) (None, 40) 10280

=================================================================
Total params: 16,822,376
Trainable params: 16,822,376
Non-trainable params: 0
```

여기서 볼 수 있듯이 `VGG16`의 합성곱 기반 층은 14,714,688개의 매우 많은 파라미터를 가지고 있으며, 합성곱 기반 위에 추가한 분류기는 2,097,408개의 파라미터를 가진다.

모델을 컴파일하고 훈련하기 전에 합성곱 기반 층을 <strong><span style="font-size: 1.2rem; color: crimson;">동결</span></strong>하는 것이 아주 중요하다. 하나 이상의 층을 동결 한다는 것은 훈련하는 동안 가중치가 업데이트 되지 않도록 막는다는 뜻이다. 이렇게 하지 않으면 합성곱 기반 층에 의해 사전에 학습된 표현이 훈련하는 동안 수정될 것이다. 맨위의 Dense층은 랜덤하게 초기화되었기 때문에 매우 큰 가중치 업데이트 값이 네트워크에 전파될 것이다. 이는 사전에 학습된 표현을 크게 훼손하게 되고 이는 곧 시간과 비용이 크게 발생하게 된다.

케라스에서는 trainable 속성을 False로 설정하여 네트워크를 동결한다.

합성곱 기반 층의 구조를 다시 살펴보면
```bash
>>> conv_base.summary()
Model: "vgg16"
_________________________________________________________________
Layer (type) Output Shape Param #
=================================================================
input_1 (InputLayer) [(None, 150, 150, 3)] 0

block1_conv1 (Conv2D) (None, 150, 150, 64) 1792

block1_conv2 (Conv2D) (None, 150, 150, 64) 36928

...


block4_pool (MaxPooling2D) (None, 9, 9, 512) 0

block5_conv1 (Conv2D) (None, 9, 9, 512) 2359808

block5_conv2 (Conv2D) (None, 9, 9, 512) 2359808

block5_conv3 (Conv2D) (None, 9, 9, 512) 2359808

block5_pool (MaxPooling2D) (None, 4, 4, 512) 0

=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
_________________________________________________________________
```
마지막 3개의 합성곱 층을 미세 조정한다. 다시 말해 *block4_pool (MaxPooling2D) (None, 9, 9, 512) 0
*까지 모든층은 동결되고 그 이후는 학습대상이 된다.

```python
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == "block5_conv1":
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
```

모델을 수정했으니 컴파일을 다시한다.