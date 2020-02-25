---
published: true
layout: single
title: "설명 가능한 추천 시스템 논문 Review"
category: Paper
toc: true
use_math: true
---

## 들어가며...

딥러닝 모델의 경우 블랙박스를 거쳐 결과물이 나오기 때문에 해석에 어려움이 있다. 그 문제를 해결하기 위해 설명 가능한 인공지능(XAI) 기술이 발전하고 있다. 설명 가능한 인공지능은 Why? 에 답을 해줄 수 있다.

최근에는 기존에 존재하던 추천 시스템과 결합하여 '설명 가능한 추천 시스템'이 연구되고 있다. 설명 가능한 추천 시스템에 대해 잘 정리된 논문이 있어 논문을 보며 공부한 내용을 정리했다(수식 등의 내용은 빼고 간단하게 설명만 해보고자 한다). 

원본이 궁금한 사람은 구글에 검색하면 나오니 참고하길 바란다. 



- Explainable Recommendation: A Survey and New Perspectives - Yongfeng Zhang, Xu Chen



## 추천 시스템

설명 가능한 추천 시스템이 연구되기 전부터 추천 시스템 자체의 연구는 오랫동안 진행되었다. 설명 가능한 추천 시스템을 보기 전에 기존의 추천 시스템 몇 가지를 확인해보자.



1. Relevant User or Item

   나와 비슷한 사용자(평점을 동일하게 줬다던가, 검색 내역이 동일하던가 등)가 구매(혹은 관심을 보인)한 물건들을 추천해주는 시스템이다. 비슷한 유저를 찾지 못하면 추천이 불가능하다는 단점이 존재한다. 협업 필터링 추천 시스템과 관련이 있다.

   

2. Feature-based

   사용자가 관심을 가지는 특징을 분석하여 추천해주는 시스템이다. 한마디로 사용자의 취향을 분석한다고 볼 수 있다. 콘텐츠 기반 추천 시스템과 관련이 있다.

   

3. Opinion-based

   인스타그램, 블로그 후기 등 사용자가 만들어낸 의견을 분석하여 추천해주는 시스템이다.

     

4. Sentence

   대상 사용자에게 문장을 생성해 추천해주는 시스템이다. 주로 미리 설정된 템플릿의 주요 단어를 개개인에 맞게 바꾸어 문장을 생성해낸다.

   

5. Visual

   사용자에게 사진 기반 추천을 해준다. 전체 이미지 혹은 관심을 가질법한 부분을 하이라이트 표시하여 보여준다.

   

6. Social

   사용자의 사회 관계망을 기반으로 추천을 제공한다. 사용자의 신뢰를 높일 수 있다.



1~3번은 추천 시스템 알고리즘이고, 4~6번은 사용자에게 설명을 어떻게 보여줄 것인지에 촛점이 맞추어져 있다.



## 설명 가능한 추천 모델

위에서 기존에 연구하던 추천 시스템 몇 가지를 다루어 보았고, 본격적으로 설명 가능한 추천 모델을 소개한다. 세세한 모델이나 용어까지 다루기 어려워 Keyword에 정리하여 적어둘테니 관심있는 사람은 직접 찾아보길 바란다.



1. Factorization Model

   matrix factorization을 활용한 모델이다. matrix factorization이란 사용자나 물건의 잠재 임베딩 차원에 투영하는 것을 말한다. 쉽게 말해 사용자나 물건의 잠재적 특성을 추출해 내는 것을 말한다. 잠재적 특성이란 사용자의 결정에 영향을 끼치는 특정 요소이다. 하지만 각 요소가 어떤 의미를 가지는 지 모른다는 단점이 있다.

   - **Keyword** : Explicit Factor Models(EFM), Attention-driven Factor Model(AFM), Sentiment Utility Logistic Model(SULM), Explainable Matrix Factorization(EMF)

   

2. Topic Modeling

   Latent Dirichlet Allocation(LDA)를 통해 숨은 주제를 찾아내어 추천에 활용하는 모델이다.

   - **Keyword** : Hidden Factor and Topic Model(HFT), Factorized Latent Aspect Model(FLAME)

     

3. Graph-based Models

   사용자와 사용자 또는 사용자와 물건 관계를 그래프로 표현하는 모델이다. 사회 관계망 추천 시스템에서 활용된다.

   - **Keyword** : TriRank, UniWalk algorithm

     

4. Deep Learning

   CNN, RNN, LSTM 등 여러가지 딥러닝 기법을 사용한 모델이다. 상위 n개 항목 추천, 시계열 추천, 점수 예측 등에 사용된다. 자연어 처리에서는 자연어를 생성하여 추천해주기도 한다.

   - **Keyword** : Deep Explicit Attentive Multi-view Learning Model(DEAML)

   

5. Knowledge Graph-based

   지식 그래프를 활용한 모델이다. 지식 그래프란 개념들 사이의 관계를 나타내는 그래프이다. 예를 들어 '사용자는 물건을 <u>구입</u>했다'라는 문장이 있으면 사용자와 물건 사이에는 <u>구입</u>이라는 관계가 존재하고 지식 그래프는 이러한 관계를 나타낸다. 

   - **Keyword** : Key-Value Memory Networks(KV-MN)

   

6. Rule Mining

   흔히 장바구니 분석으로 알려진 연관 규칙을 활용한 모델이다. 유튜브 추천 시스템이 해당 알고리즘을 통해 제작되었다. 각 항목 별 관계점수를 계산하여 개인에 맞게 추천해준다.

   

7. Model Agnostic and Post Hoc

   모델 불가지론 또는 사후 분석이란 추천 이후에 설명을 하는 것을 말한다. 즉, 추천 시스템 자체가 먼저 추천을 제공하고 추천된 이유만을 따로 분석한다. 블랙박스를 설명하는 것을 목표로 한다.

   - **Keyword** : Local Interpretable Model-agnositc Explanation(LIME), Fast Influence Analysis(FIA)



정리하다보니 내용이 너무 방대하여 세세한 부분까지 작성하지 못하였다(설명하기엔 지식이 부족하기도 하다). 간단하게 이런 방법들이 있구나 정도로만 생각하고 자세한 내용은 Keyword를 바탕으로 검색해보자.



## 설명 가능한 추천 시스템의 평가

추천 시스템의 평가 방법에 대해 알아보자.



1. User Study

   사용자 연구는 간단히 말해 설문조사이다. 사용자들에게 직접 추천 시스템을 경험시킨 후 평가를 수집하는 방식으로 상당히 정확하다.

   

2. Online Evaluation

   온라인 평가는 말 그대로 온라인에서 진행하는 평가 방법이다. A/B 테스트를 통해 추천 시스템의 효과를 측정해 볼 수 있다. 평가 지표는 Click-through-rate(CTR), 전환율이 대표적이다.

   

3. Offline Evaluation

   오프라인 평가는 계산 가능한 공식을 통해 평가한다. 크게 두 가지 지표로 나눌 수 있는데, **추천 항목의 설명률**과 **설명 내용의 질**이다. 추천 항목의 설명률이란 추천된 물건들 중 몇 %나 설명할 수 있는지에 대한 지표이다. 설명 내용의 질이란 말 그대로 내용의 질이 어떤지를 나타내는 지표이다. 평가 지표는 BLEU 점수, ROUGE 점수가 대표적이다.

   

4. Qualitative Evaluation by Case Study

   사례 연구는 정성적 연구를 하는 데 도움이 된다. 기존에 존재하는 데이터들을 활용해 사용자 선호도, 물건의 질 등을 연구한다.



## 마치며...

원본 논문이 103쪽이나 되는 긴 논문이다(해당 내용을 자세히 공부하려면 꼭 읽어보길 바란다). 전체 내용을 옮겨봤자 단순 번역일 것 같아 나중에 찾아볼 수 있도록 나에게 필요한 부분만 발췌하여 정리하였다. 이 포스팅을 읽는 누군가에게 설명 가능한 추천 시스템의 개략적인 소개가 되었으면 하는 바람이다.



다음 포스팅은 자연어 생성 모델을 활용한 추천 시스템 논문을 살펴볼 예정이다.