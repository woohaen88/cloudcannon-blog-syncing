---
date: 2021-12-13-Fine_Tuning
title: nlp-classification
subtitle: 국민청원 분류하기
image: /images/projects/nlp-classification/thumbnail.png
---

# 국민청원 분류하기

텍스트 데이터를 모델링하는 분야를 자연어 처리(Natural language Processing, NLP)라고 한다. 자연어 처리에는 여러가지가 있는데 그중 대표적으로 

1. 텍스트 분류(Text Classification)
2. 감정 분석(Sentiment Analysis)
3. 요약(Summarization)
4. 기계 번역(Machine Translation)
5. 질문 응답(Question Answering)


<blockquote>
이번 프로젝트의 목표는 국민청원 글에 <b>TextCNN</b> 이라는 모델을 적용하여 특정 글에서 청원 참여인원이 <b>1,000명</b> 이상 달성할지 여부를 <b>분류</b>하는 것을 목표로 한다.
</blockquote>

다시말해서 수많은 청원 글 중 주목받을 만한 글을 예측하는 것이 목표라 할 수 있겠다. <문구수정> <span style="color: #ff8300;">관심이 필요한 많은 사연들에 사람들의 눈길이 한 번 더 닿도록 하기 위함이다. 국민청원의 몇몇 사연들은 언론이나 SNS등의 도움을 받아 20만 동의 단숨에 달성하곤 한다. 반면 중대하지만 눈에 띄지 않고, 도움이 반드시 필요하지만 관심을 받지 못한 사연들은 많은 동의를 받기 어렵다. 사람들의 관미미이 일부 청원 글에 집중되기보다 사회의 다양한 사연들에 전해지도록 하는 것이 이 프로젝트의 궁극적인 목적이다. </span> <span문구수정>

## 1. Intro

프로젝트의 목표는 <span style="color: crimson;">'주목받을 만한 청원 분류하기'</span>이다. 하지만 '주목받을 만한'이라는 기준이 매우 애매하며 이는 사람마다 다를 수 있다. 사연의 경중을 판단하는 것은 입장차이마다 다를 수 있으며 매우 주관적인 영역이기에 읽는 사람마다 다른 판단이 내려진다. 우리는 이러한 주관전 판단을 배제할 수 있는 방법으로 딥러닝을 도입할 것이다. 딥러닝 모델을 통하여 **높은** 청원 참여인원을 기록한 글들의 특징을 학습하여, **새로운 글**이 입력되었을 때 청원 참여인원이 높은 글들과의 **유사성**을 계산하여 주목받을 만한 글인지 아닌지를 판단하도록 한다.

<figure style="text-align:center;"><img width="2940" height="512" src="/images/projects/nlp-classification/model_flow.png" />
    <figcaption style="font-size: 0.8rem; margin-top: 1rem;">Figure 1. 모델 전체 흐름</figcaption>
</figure>

---
## 2. crawling(크롤링)

크롤링이란, 웹 페이지에서 원하는 데이터를 추출하여 수집하는 방법이다. 이 기법을 사용하여 2021년 1월 4일부터 2021년 12월 14일까지의 등록된 국민청원 글을 얻을 수 있다. 하나의 국민청원 글에서 청원 제목, 참여인원, 카테고리, 청원시작일, 청원마감일, 청원 내용 총 6개 항목을 추출한다. 가장 먼저, 데이터를 수집하기 위해 청와대 홈페이지의 국민청원 [https://www1.president.go.kr/petitions](https://www1.president.go.kr/petitions) 에 접속해서 보면 청원 글은 마지막 숫자가 1씩 변하는 것을 알 수 있다. 이러한 규칙을 이용하여 for문을 사용해 크롤링 코드를 추가한다.

```python
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time

result = pd.DataFrame()
for i in range(595230, 603000+1):
    URL = f"https://www1.president.go.kr/petitions/{i}"

    response = requests.get(URL)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.find("h3", class_='petitionsView_title')
    count = soup.find('span', class_='counter')

    for content in soup.select('div.petitionsView_write > div.View_write'):
        a = []
        for tag in soup.select('ul.petitionsView_info_list > li'):
            a.append(tag.contents[1])

        if len(a) != 0:
            df1 = pd.DataFrame({
                "start" : [a[1]],
                "end" : [a[2]],
                "category" : [a[0]],
                "count" : [count.text],
                "title" : [title.text],
                "content" : [content.text.strip()[0 : 13000]]
            })
            result = pd.concat([result, df1])
            result.index = np.arange(len(result))

        if i % 60 == 0:
            print("Sleep 90seconds. Count:" + str(i) + ", Local Time:"
                  + time.strftime("%Y-%m-%d", time.localtime(time.time()))
                  +", Data Length:"+str(len(result)))
```

엑셀에 데이터를 저장하기 위해 데이터의 길이를 13,000으로 제한한다. 엑셀에서 한 셀에 넣을 수 있는 글자수가 32,767자이기 때문에 13,000자까지 크롤링한 후 토크나이징 및 청원 제목과 병합하여 32,767자를 초과하지 않도록 하기 위함이다.

