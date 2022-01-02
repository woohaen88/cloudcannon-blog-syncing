---
date: 2021-12-28 00:00:00
title: 신경망의 모듈화
description: 객체 지향 프로그래밍에서 자체 신경망 계층을 만들어서 재사용하거나 더 복잡한 신경망을 만드는 방법
tags:
- pytorch
- Module
image: /images/posts/03_modulization/thumbnail.png
---

<!-- 포스트 이미지 폴더: /images/posts/02_dataset_and_dataloader/ -->
<h1>1. 커스텀 계층 만들기</h1>

파이토치에서 자체 신경망을 만들려면 `nn.Module`을 상속해서 클래스를 정의한다. nn.Module을 상속하게 되면 forward 메서드를 구현하면 자동 미분까지 가능해진다.

> 이미 특정 Variable형의 x를 net(x) 형식으로 사용했다. nn.Module의 `__call__` 메서드는 내부에서 forward 메서드를 사용하고 있으므로 `net(x)` 형식이 가능하다.

```python
from torch import nn


# 방법 1
class CustoemLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, p=0.5):
        super(self, CustoemLinear).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.drop(x)
        return x

mlp = nn.Sequential(
    CustoemLinear(64, 200),
    CustoemLinear(200, 200),
    CustoemLinear(200, 200),
    nn.Linear(200, 10)
)

# 방법2
class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        self.ln1 = CustoemLinear(in_features, 200)
        self.ln2 = CustoemLinear(200, 200)
        self.ln3 = CustoemLinear(200, 200)
        self.ln4 = CustoemLinear(200, out_features)

    def forward(self, x):
        x = self.ln1(x)
        x = self.ln2(x)
        x = self.ln3(x)
        x = self.ln4(x)
        return x

mlp = MLP(64, 10)
```