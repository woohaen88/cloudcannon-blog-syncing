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

```python
# 데이터 저장
result.to_csv("petitions.csv" ,index=False)
```

---
## 3. 전처리

```python
result = pd.read_csv("petitions.csv")

import re
df = result

def remove_white_space(text): # 공백문자를 제거하는 함수를 정의  \t => tap, \r\n => 엔터, \n => 줄바꿈, \f => 새 페이지, \v => 수직 탭
    
    text = re.sub(r'[\t\r\n\f\v]', '', str(text))
    return text

def remove_special_char(text): # ㄱ-ㅣ(자음 ㄱ-ㅎ, ㅏ~ㅣ) 가-힣, 0-9에 해당하지 않는 문자가 등장하면 공백으로 치환, 영어가 등장해도 공백으로 교체
    text = re.sub('[^ ㄱ-ㅣ|가-힣 0-9]+', '', str(text))
    return text

df.title = df.title.apply(remove_white_space)
df.title = df.title.apply(remove_special_char)

df.content = df.content.apply(remove_white_space)
df.content = df.content.apply(remove_special_char)
```

### 3.1 토크나이징 및 변수 생성

토크나이징이란 문장을 의미 있는 부분으로 나누는 과정을 말하며, 그 나누어진 부분을 토큰(Token)이라고 부른다. 간단하게 **형태소**로 이해할 수 있다. 예를 들어, <span style="color: #288ba8;">"나는 치킨을 먹는다"</span> 라는 문장을 형태소 단위로 토크나이징하면 <span style="color:#288ba8">["나", "는", "치킨", "을", "먹", "는", "다"]</span> 라는 7개의 토큰을 얻을 수 있다. 
분석에 필요한 모든 문장을 토크나이징 해주어야 하는데 그 이유는 컴퓨터는 다른 형태의 단어는 다른 단어라고 인식하기 때문이다. 예를 들어, <span style="color: #288ba8">"먹습니다", "먹다", "먹어요", "먹네요", "먹었다"</span>는 모두 "먹다"의 의미지만 컴퓨터는 당연하게도 모두 다른 단어라고 인식한다. 이 것을 해결하기 위해 모든 단어를 형태소로 분할한다.
"먹습니다" -> ["먹", "습니다"]
"먹" -> ["먹", "다"] 등과 같이 전부 나누게 되면 <span style="color: #288ba8">"먹"</span> 의미를 가진 최소 단위인 "먹"이 추출되고, 컴퓨터는 이 토큰을 근거로 위 문장들이 <span style="color:crimson;">모두 유사한 의미</span>를 지녔다고 판단한다.

우리는 청원 제목과 청원 내용을 토크나이징해야 한다. 청원 제목은 형태소 단위로, 청원 내용은 명사 단위로 문장을 나눈다. 청원 내용에서는 명사만 추출하여 학습하는데, 그 이유는 학습효율과 키워드 중심의 분석을 하기 위함이다. 청원 내용은 글이 길어 형태소 단위로 토크나이징해 학습하기에는 비교적 많은 시관과 자원이 요구된다. 따라서 명사만 추출하여 키워드 중심의 학습을 진행한다.

```python
from konlpy.tag import Okt
```

konlpy에서 Okt를 임포트하기 위해서는 2가지가 필요하다.
1. Java가 설치되어 있어야 하며
2. tweepy가 < 4.0.0 이어야 한다.

따라서 `AttributeError: module 'tweepy' has no attribute 'StreamListener'` Error가 나타나면 
```bash
pip install tweepy==3.10.0
```
로 tweepy버전을 낮추자.

```python
okt = Okt()
df["title_token"] = df.title.apply(okt.morphs)  # 청원 제목을 형태소 단위로 토크나이징 하여 title_totken에 저장
df["content_token"] = df.content.apply(okt.nouns) # 청원 내용을 명사(Nouns)단위로 토크나이징하여 저장


# 파생변수 생성
df["token_final"] = df.title_token + df.content_token
df["count"] = df["count"].replace({"," : ""}, regex=True).apply(lambda x : int(x)) # 참여인원은 천 단위마다","가 있어 object형태로 인식함 참여인원에 ","를 제거하고 int형으로 변환

# 분석에 필요한 toekn_final과 label만 추출하여 df_drop에 저장
df["label"] = df["count"].apply(lambda x: "yes" if x>1000 else "No")

df_drop = df[["token_final", "label"]]
```

## 4. 단어 임베딩

최종적으로 사용할 데이터는 국민청원의 전처리 결과인 "token_final"과 참여인원 1,000명 이상 여부를 나타내는 'label'이다.
딥러닝 모델은 Input으로 숫자데이터를 입력해주어야 한다. 따라서 딥러닝 모델을 학습시키기 위해서는 이러한 문자데이티러르 숫자로 변환하여 컴퓨터가 이해할 수 있도록 해야 한다. 이러한 과정을 <span style="color:crimson">단어 임베딩(word embedding)</span>이라고 한다. 다양한 임베딩 방법이 있지만 널리 사용하는 <span style="color:crimson"><b>Word2Vec</b></span>을 사용하겠다.

### 4.1 Word2Vec을 들어가기 전에
* Text1. ['음주운전', '사고', '가해자', '강력', '처벌'] -> [0, 1, 2, 3, 4]
* Text2. ['음주운전', '역주행', '사건', '집행유예', '처벌'] -> [0,5, 6, 7, 4]
* Text3. ['음주운전', '사고', '면허', '취소', '규정'] -> [0, 1, 8, 9, 10]

위 인덱스는 임의로 토큰이 등장하는 순서대로 숫자를 부여했다. 음주운전 0, 사고는 1의 값을 갖는다. 하지만 이 방법은 단순히 토큰에 숫자로 치환한 것 그 이상의 의미는 갖지 못한다.

> **Word2VEC**

**Word2Vec**은 단어의 의미와 유사도를 반영하여 단어를 벡터로 표현하는 방식이다. 예를 들어 <em>"왕-남자 = 여왕"</em>을 벡터로 표현하여 연산하는 것이다. Word2Vec을 관통하는 핵심은 ***'토큰의 의미는 주변 토큰의 정보로 표현된다고'*** 가정한다. 즉 특정 토큰 근처에 있는 토큰들은 비슷한 위치의 벡터로 표현된다는 것이다.

***One-Hot Encoding Vecotr*** 에 가중치 행렬을 곱하여 ***Word Embding Vector*** 를 생성할 수 있다. 가중치 행렬의 차원(Dimension)은 사용자가 지정해주어야 하는 값으로, 이 프로젝트에서는 ***100차원***을 지정한다. 가중치 행렬을 학습하는 과정으로 ***CBOW***와 ***Skip-Gram***이 있으며 두 가지 방법 모두 문장을 ***윈도우 형태***로 일부분만 보는 것을 기본으로 합니다. 중심 토큰의 양옆 토큰을 포함한 윈도우가 이동하면서 중심 토큰과 주변 토큰의 관계를 학습한다. CBOW의 목적은 윈도우 크기만 큼 앞뒤 주변 토큰을 벡터로 변환해 더한 후 중앙토큰을 맞추는 것이고, 반대로 Skip-gram의 목적은 중심 토큰을 벡터로 변환한 후 윈도우 크기만큼 주변 토큰을 맞추는 것이다. 일반적으로 Skip-gram의 성능이 더 좋다고 알려져있다.


```python
from gensim.models import Word2Vec # Word2Vec, Doc2Vec, FastText, LDA Model 등과 같이 자연어 처리에 사용되는 모델을 지원하는데 그중 Word2Vec을 사용한다.

embedding_model = Word2Vec(df_drop["token_final"], # 임베딩 벡터를 생성할 대상이 되는 데이터
                           sg=1, # Word2Vec의 모델 구조 옵션을 지정(1: Skip-Gram, 0:Cbow)
                           vector_size=100, # 임베딩 벡터의 크기(Dimension)을 지정
                           window=2, # 임베딩 벡터 생성 시 문맥 파악을 위해 고려해야 할 앞, 뒤 토큰 수를 지정
                           min_count=1, # 전체 토큰에서 일정횟수 이상 등장하지 않는 토큰은 임베딩 벡터엣서 제외
                           workers=4) # 실행할 병령 프로세서의 수

print(embedding_model)

model_result = embedding_model.wv.most_similar("음주운전") # 9
print(model_result)


##### 임베딩 모델 저장 및 로드
from gensim.models import KeyedVectors # 임베딩 모델을 불러오기 위한 클래스르 불러옴

import os
if not os.path.isdir("data"):
    os.makedirs("data")

embedding_model.wv.save_word2vec_format("data/petitions_tokens_w2v") # 임베딩 모델을 저장

loaded_model = KeyedVectors.load_word2vec_format("data/petitions_tokens_w2v") # 폴더에 저장되어 있는 임베딩 모델을 불러와 "loaded_model"에 저장

model_result = loaded_model.most_similar("음주운전") # 임베딩 모델이 이상 없이 로드되었는지 확인하기 위해 "음주운전" 유사한 단어와 벡터값이 이전 결과와 같은지 확인
print(model_result)
```

