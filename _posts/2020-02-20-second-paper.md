---
published: true
layout: single
title: "설명 가능한 추천을 위한 자연어 생성 논문 Review"
category: Paper
toc: true
use_math: true
---

## 들어가며...

지난 포스팅에 설명 가능한 추천 시스템에 대한 간략한 소개를 했다.

비전공자에 대학원생도 아니기 때문에 전문성이 부족하다(특히 공식 부분은 100% 이해하진 못한다).

설명이란 자고로 글로 된 설명이 좋은 것 같아 이번에는 자연어 생성 모델에 대해 공부했다. 

해당 논문도 구글에 제목을 검색하면 찾을 수 있다.



- Generate Natural Language Explainations for Recommendation - Hanxiong Chen, Xu Chen, Shaoyun Shi, Yongfeng Zhang



## 소개

대표적인 기존 자연어 설명 모델은 템플릿 기반(Template-based) 모델로 미리 제작된 템플릿에 주요 단어만 변경하는 방식이다. 하지만 템플릿 기반 모델은 표현의 제한이 심하고 다양성이 부족하다는 단점이 있다. 검색 기반 방식(Retrieval-based)은 사용자의 후기를 검색하여 설명 문장으로 사용하는 방식이다. 하지만 존재하는 문장을 기반으로 하기 때문에 설명이 제한적이고 새로운 문장을 만들어내지 못한다는 단점이 있다.

개인화된 자연어 설명 시스템에는 크게 3가지의 문제점이 있다. 이러한 문제를 해결하기 위해서 해당 논문은 자연어 생성 모델을 제안한다. 



1. Data bias

   사용자의 후기를 바탕으로 학습하기 때문에 자료의 편향이 발생한다. 자료에 노이즈가 많아 자동 노이즈 제거 기술을 필요로 한다.

    

2. Personalization

   사람마다 중요시하는 물건의 특징이 다르기 때문에 개인화 기술이 중요하다.

   

3. Evaluation

   자연어에 대한 성능 지표가 제한적이기 때문에 검증이 중요하다. 본 논문에서는 오프라인 지표를 사용하여 성능을 평가한다(BELU, ROUGE, feature coverage). 지표에 대해서는 아래에 상세하게 설명한다.



## 관련 연구

협업 필터링 기법은 최신 개인화 추천 시스템의 중요한 접근법이다. 초기 협업 필터링은 사용자 기반(user-based)나 물건 기반(item-based) 기법을 활용했다. 후기 협업 필터링은 matrix factorization 알고리즘을 활용하여 성능을 향상시켰다. 최근에는 딥러닝 기법을 활용한다.



## 프레임워크

본 논문에서 제안하는 모델은 2가지 모듈로 이루어져 있다. **점수 예측 모듈**과 **자연어 설명 생성 모듈**이다. 두 모듈은 사용자와 물건의 잠재 요소를 입력값으로 받는다.



### 점수 예측 모듈

본 모듈의 목표는 회귀 신경망을 활용하여 사용자와 물건의 잠재 요소를 사용하여 사용자의 점수를 예측하는 것이다. U는 사용자 잠재 요소이고 V는 물건의 잠재 요소이다. MLP를 활용하여 점수 r을 최적화 시킨다. RMSE를 성능지표로 사용한다.



### 개인화 자연어 설명 생성 모듈

본 모듈의 목표는 개인화된 자연어 설명을 생성하는 것이다. 논문에서는 3가지 방식을 소개한다.

1.  auto-denosing starategy

   중요 문장을 학습하고 중요하지 않은 문장을 무시한다. 분자는 i번째 문장의 특징 단어의 수이고, 분모는 i번째 문장의 전체 단어의 수이다. 훈련 과정에서 베타값을 손실함수에 곱하여 중요도를 조절할 수 있다.
   
   
   $$
   \beta_i = \frac {N^i_k} {N^i_w}
   $$
   
   
2. feature-aware attention for personalized explaination generation

   feature words는 제품의 특징을 설명하는 특징 단어이다. 사용자들은 각 제품별로 관심있는 특징 단어가 다르기 때문에 feature-aware attention 모델을 통해 추천 시스템을 개인화한다. 시간 t에 대한 특징 단어 관심도는 i번째 문장을 만드는 initial hidden state가 된다.

   

   $$
   o_t=\sum^{|K|}_{i=1}\alpha(i,h_t)k_i
   $$

   

3. hierarchical GRU model for sentence generation

   GRU 모델은 문장 생성을 목표로한다. 문맥 수준의 GRU와 단어 수준의 GRU 2가지로 구성되어 있다. 

   

   1) 문맥 수준의 GRU

   ​	사용자와 물건 잠재 요소 쌍 하나로 여러개의 문장을 만들 수 있다. 각 문장은 손실 함수를 따로 가지고 있으며, 훈련하는 동안 자동 노이즈 제거 전략을 사용하여 불필요한 문장을 제거한다.

   

   2) 단어 수준의 GRU

   ​	설명 문장에 들어갈 단어를 생성한다. 문맥 수준의 GRU가 문장을 만들었다면 개개인에 맞는 단어를 생성해 주는 것이 단어 수준의 GRU 역할이다. 



## 멀티 태스킹 학습

두 모듈을 합쳐서 만든 최종적인 목적 함수 공식이며 각 요소들을 살펴보면 다음과 같다. 



$$
\cal J = \min_{U,V,E,\Theta}(\cal L^r+\sum^{|Review|}_{i=1}\beta_i\cal L^s_i+\lambda(||\sf U||^2_2+||\sf V||^2_2+||\Theta||^2_2))
$$



E : 단어 임베딩 행렬

Θ : 신경망 파라미터

λ : 규제 가중치

L<sup>r</sup> : 회귀 분석의 손실 함수

β<sub>i</sub> : i번째 문장의 감독 요소(= auto-denosing 공식)



## 평가

점수 예측 모듈과 자연어 생성 모듈에 대해 각각 평가 지표를 살펴본다.



### 점수 예측 평가

논문에 소개된 모델과의 비교를 위해 3가지 베이스라인을 소개한다. BiasedMF와 SVD++ 방식은 예측을 위해 점수 정보만을 활용하고,  DeepCoNN 방식은 예측을 위해 사용자 생성 후기를 포함한다.

- BiasedMF
- SVD++
- DeepCoNN 

평가 지표는 RMSE를 활용한다.



$$
RMSE = \sqrt {\frac 1 N \sum_{ u\in U,i\in V}(r_{u,i}-\hat r_{u,i})^2}
$$



### 생성된 설명 문장 평가

논문에 소개된 모델과의 비교를 위해 Att2SeqA를 베이스라인으로 한다. 생성된 설명 문장을 평가하는 3가지 평가 지표를 소개한다.



- BLEU : n-gram의 %를 활용해 얼마나 사람과 비슷하게 문장을 생성했는지 평가한다(정밀도와 관련이 있다).
  
  
  $$
  BLEU = \frac {\sum_{C\in{(Candidates)}}\sum_{ngram\in C}Count_{clip}(ngram)} {\sum_{C^\prime \in(Candidates)}\sum_{ngram^\prime \in C^\prime}Count(ngram^\prime)}
  $$
  
  
- ROUGE: 생성된 문장의 얼마나 많은 단어가 사람과 일치하는지 평가한다(재현율과 관련이 있다). 정밀도, 재현율, F점수를 사용하여 평가한다.

  
  
  $$
  ROUGE-N = \frac {\sum_{S\in(References)}\sum_{ngram\in S}Count_{match}(ngram)} {\sum_{S\in(References)}\sum_{ngram\in S}Count(ngram)}
  $$
  
  
- Feature words coverage : 사용자 개인의 선호도를 얼마나 잘 반영했는지 평가한다.
  
  
  $$
  Coverage_{feature} = \frac {N_r} {N_c}
  $$
  



## 마치며...

이후 내용은 실험 결과에 대한 내용이며 당연하게도 본 논문에서 소개된 HSS 모델이 가장 성능이 좋은 것으로 나온다(해당 내용은 생략했다). 

논문을 눈으로만 읽다가 정리하면서 읽으니 눈에도 더 잘 들어오고 핵심내용 정리가 잘 되었다. 앞으로도 공부한 논문을 꾸준히 정리하여 업로드할 예정이다.
