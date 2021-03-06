---
published: true
layout: single
title: "핸즈온 머신러닝 정리 - 5장"
category: Book
toc: true
use_math: true
---

# 서포트 벡터 머신

서포트 벡터 머신은 매우 강력하고 선형이나 비선형 분류, 회귀, 이상치 탐색에도 사용할 수 있는 다목적 머신러닝 모델입니다. 머신러닝에서 가장 인기 있는 모델에 속하고 머신러닝에 관심 있는 사람이라면 반드시 알고 있어야 하는 모델입니다. SVM은 특히 복잡한 분류 문제에 잘 들어맞으며 작거나 중간 크기인 데이터셋에 적합합니다.



## 선형 SVM 분류

SVM 분류기를 클래스 사이에 가장 폭이 넓은 도로를 찾는 것으로 생각할 수 있습니다. 그래서 라지 마진 분류라고 합니다. 도로 바깥쪽에 훈련 샘플을 더 추가해도 결정 경계에는 전혀 영향을 미치지 않습니다. 도로 경계에 위치한 샘플에 의해 전적으로 결정됩니다. 이런 샘플을 **서포트 벡터**라고 합니다.



### 소프트 마진 분류

모든 샘플이 도로 바깥쪽에 올바르게 분류되어 있다면 이를 **하드 마진 분류**라고 합니다. 하드 마진 분류에는 두 가지 문제점이 있습니다. 데이터가 선형적으로 구분될 수 있어야 제대로 작동하며, 이상치에 민감합니다.

이런 문제를 피하려면 좀 더 유연한 모델이 필요합니다. 도로의 폭을 가능한 한 넓게 유지하는 것과 **마진 오류** 사이에 적절한 균형을 잡아야 합니다. 이를 **소프트 마진 분류**라고 합니다.



## 비선형 SVM 분류

선형 SVM 분류기가 효율적이고 많은 경우에 아주 잘 작동하지만, 선형적으로 분류할 수 없는 데이터셋이 많습니다. 비선형 데이터셋을 다루는 한 가지 방법은 다항 특성과 같은 특성을 더 추가하는 것입니다. 이렇게 하면 선형적으로 구분되는 데이터셋이 만들어질 수 있습니다.



### 다항식 커널

다항식 특성을 추가하는 것은 간단하고 모든 머신러닝 알고리즘에서 잘 작동합니다. 하지만 낮은 차수의 다항식은 매우 복잡한 데이터셋을 잘 표현하지 못하고 높은 차수의 다항식은 굉장히 많은 특성을 추가하므로 모델을 느리게 만듭니다.

다행히도 SVM을 사용할 땐 **커널 트릭**이라는 거의 기적에 가까운 수학적 기교를 적용할 수 있습니다. 실제로는 특성을 추가하지 않으면서 다항식 특성을 많이 추가한 것과 같은 결과를 얻을 수 있습니다.



### 유사도 특성 추가

비선형 특성을 다루는 또 다른 기법은 각 샘플이 특정 **랜드마크**와 얼마나 닮았는지 측정하는 유사도 함수로 계산한 특성을 추가하는 것입니다. 랜드마크를 선정 후 가우시안 **방사 기저 함수**를 유사도 함수로 정의하여 계산하면 선형적으로 구분이 가능합니다.

단점은 훈련 세트에 있는 n개의 특성을 가진 m개의 샘플이 m개의 특성을 가진 m개의 샘플로 변환된다는 것입니다. 훈련 세트가 매우 클 경우 동일한 크기의 아주 많은 특성이 만들어집니다.



### 가우시안 RBF 커널

다항 특성 방식과 마찬가지로 유사도 특성 방식도 머신러닝 알고리즘에 유용하게 사용될 수 있습니다. 추가 특성을 모두 계산하려면 연산 비용이 많이 드는데 특히 훈련 세트가 클 경우 더 그렇습니다. 하지만 커널 트릭이 한 번 더 SVM의 마법을 만듭니다. 유사도 특성을 많이 추가하는 것과 같은 비슷한 결과를 실제로 특성을 추가하지 않고 얻을 수 있습니다.



### 커널의 선택

다른 커널도 있지만 거의 사용되지 않습니다. 예를 들어 어떤 커널은 특정 데이터 구조에 특화되어 있습니다. 문자열 커널이 가끔 텍스트 문서나 DNA 서열을 분류할 때 사용됩니다.

여러 가지 커널 중 어떤 것을 사용해야 할까요? 언제나 선형 커널을 가장 먼저 시도해봐야 합니다. 특히 훈련 세트가 아주 크거나 특성 수가 많을 경우에 그렇습니다. 훈련 세트가 너무 크지 않다면 가우시안 RBF 커널을 시도해보면 좋습니다. 대부분의 경우 이 커널이 잘 들어맞습니다.



### 계산 복잡도

| 파이썬 클래스 |   시간 복잡도   | 외부 메모리 학습 지원 | 스케일 조정의 필요성 | 커널 트릭 |
| :-----------: | :-------------: | :-------------------: | :------------------: | :-------: |
|   LinearSVC   |     O(mxn)      |        아니오         |          예          |  아니오   |
| SGDClassifier |     O(mxn)      |          예           |          예          |  아니오   |
|      SVC      | O(m²xn)~O(m³xn) |        아니오         |          예          |    예     |



## SVM 회귀

선형, 비선형 분류뿐만 아니라 선형, 비선형 회귀에도 사용할 수 있습니다. 회귀에 적용하는 방법은 목표를 반대로 하는 것입니다. 일정한 마진 오류 안에서 두 클래스 간의 도로 폭이 가능한 한 최대가 되도록 하는 대신, SVM 회귀는 제한된 마진 오류 안에서 도로 안에 가능한 한 많은 샘플이 들어가도록 학습합니다.

마진 안에서는 훈련 샘플이 추가되어도 모델의 예측에는 영향이 없습니다. 그래서 이 모델을 **ε에 민감하지 않다**고 말합니다.



## SVM 이론

이 절에서 SVM의 예측은 어떻게 이뤄지는지, 그리고 SVM의 훈련 알고리즘이 어떻게 작동하는지 설명합니다.



### 결정 함수와 예측

선형 SVM 분류기 모델은 단순히 결정 함수를 계산해서 새로운 샘플 x의 클래스를 예측합니다. 선형 SVM 분류기가 훈련한다는 것은 마진 오류를 하나도 발생하지 않거나(하드 마진) 제한적인 마진 오류를 가지면서 (소프트 마진) 가능한 한 마진을 크게 하는 w와 b를 찾는 것입니다.



### 목적 함수

결정 함수의 기울기를 생각해보면 이는 가중치 벡터의 노름 llwll와 같습니다. 이 기울기를 2로 나누면 결정 함수의 값이 ±1이 되는 점들이 결정 경계로부터 2배 만큼 더 멀어집니다. 즉, 기울기를 2로 나누는 것은 마진에 2를 곱하는 것과 같습니다. 가중치 벡터 w가 작을수록 마진은 커집니다.

하드 마진 선형 SVM 분류기의 목적 함수를 제약이 있는 최적화 문제로 표현 할 수 있습니다.


$$
minimize \frac 1 2 w^t\cdot w
$$


소프트 마진 분류기의 목적 함수를 구성하려면 각 샘플에 대해 슬랙 변수 ζ >= 0를 도입해야 합니다.


$$
minimize \frac 1 2 w^t\cdot w +C\sum^m_{i=1}\zeta^{(i)}
$$


### 콰드라틱 프로그래밍

하드 마진과 소프트 마진 문제는 모두 선형적인 제약 조건이 있는 볼록 함수의 이차 최적화 문제입니다. 이런 문제를 콰드라틱 프로그래밍 문제라고 합니다.



### 쌍대 문제

원 문제라는 제약이 있는 최적화 문제가 주어지면 쌍대 문제라고 하는 깊게 관련된 다른 문제로 표현할 수 있습니다. 일반적으로 쌍대 문제 해는 원 문제 해의 하한값이지만, 어떤 조건 하에서는 원 문제와 똑같은 해를 제공합니다. 다행히도 SVM 문제는 이 조건을 만족시킵니다. 따라서 원 문제 또는 쌍대 문제 중 하나를 선택하여 풀 수 있습니다. 둘 다 같은 해를 제공합니다.

훈련 샘플 수가 특성 개수보다 작을 때 원 문제보다 쌍대 문제를 푸는 것이 더 빠릅니다. 더 중요한 것은 원 문제에서는 적용이 안 되는 커널 트릭을 가능하게 합니다.



### 커널 SVM

모든 훈련 샘플에 변환을 적용하면 쌍대 문제에 점곱이 포함될 것입니다. 하지만 변환이 2차 다항식 변환이라면 변환된 벡터의 점곱을 간단하게 바꿀 수 있습니다. 그래서 실제로 훈련 샘플을 변환할 필요가 전혀 없습니다. 즉 점곱을 제곱으로 바꾸기만 하면 됩니다. 바로 이것이 커널 트릭입니다.
$$
선형:\;K(a,b)=a^T\cdot b\\
다항식:\;K(a,b)=(\gamma a^T\cdot b+r)^d\\
가우시안\;RBF:\;K(a,b)=\exp(-\gamma\|a-b\|^2)\\
시그모이드:\;K(a,b)=\tanh(\gamma a^T\cdot b+r)
$$


### 온라인 SVM

선형 SVM 분류기에 사용할 수 있는 한 가지 방법은 원 문제로부터 유도된 비용 함수를 최소화하기 위한 경사 하강법을 사용하는 것입니다. 하지만 이 방식은 QP 기반의 방법보다 훨씬 느리게 수렴합니다.



# 연습문제

1. 서포트 벡터 머신의 근본 아이디어는 무엇인가요?

   서포트 벡터 머신의 근본적인 아이디어는 클래스 사이에 가능한 한 가장 넓은 '도로'를 내는 것입니다. 다시 말해 두 클래스를 구분하는 결정 경계와 샘플 사이의 마진을 가능한 한 가장 크게 하는 것이 목적입니다. 소프트 마진 분류를 수행할 때는 SVM이 두 클래스를 완벽하게 나누는 것과 가장 넓은 도로를 만드는 것 사이에 절충안을 찾습니다(즉, 몇 개의 샘플은 도로 안에 놓일 수 있습니다). 또 하나의 핵심적인 아이디어는 비선형 데이터셋에서 훈련할 때 커널 함수를 사용하는 것입니다.

   

2. 서포트 벡터가 무엇인가요?

   서포트 벡터는 SVM이 훈련된 후에 경계를 포함해 도로에 놓인 어떤 샘플입니다. 결정 경계는 전적으로 서포트 벡터에 의해 결정됩니다. 서포트 벡터가 아닌(즉, 도로 밖에 있는) 어떤 샘플도 영향을 주지 못합니다. 이런 샘플은 삭제하고 다른 샘플을 더 추가하거나, 다른 곳으로 이동시킬 수 있습니다. 샘플이 도로 밖에 있는 한 결정 경계에 영향을 주지 못할 것입니다. 예측을 계산할 때는 전체 훈련 세트가 아니라 서포트 벡터만 관여됩니다.

   

3. SVM을 사용할 때 입력값의 스케일이 왜 중요한가요?

   SVM은 클래스 사이에 가능한 한 가장 큰 도로를 내는 것이므로 훈련 세트의 스케일이 맞지 않으면 크기가 작은 특성을 무시하는 경향이 있습니다.

   

4. SVM 분류기가 샘플을 분류할 때 신뢰도 점수와 학률을 출력할 수 있나요?

   SVM 분류기는 테스트 샘플과 결정 경계 사이의 거리를 출력할 수 있으므로 이를 신뢰도 점수로 사용할 수 있습니다. 그러나 이 점수를 클래스 확률의 추정값으로 바로 반환할 수는 없습니다. 사이킷런에서 SVM 모델을 만들 때 probability=True로 설정하면 훈련이 끝난 후 (훈련 데이터에 5-겹 교차 검증을 사용하여 추가로 훈련시킨) SVM의 점수에 로지스틱 회귀를 훈련시켜 확률을 계산합니다. 이 설정은 SVM 모델에 predict_proba()와 predict_log_proba() 메서드를 추가시킵니다.

   

5. 수백만 개의 샘플과 수백 개의 특성을 가진 훈련 세트에 SVM 모델을 훈련시키려면 원 문제와 쌍대 문제 중 어떤 것을 사용해야 하나요?

   커널 SVM은 쌍대 형식만 사용할 수 있기 때문에 이 질문은 선형 SVM에만 해당합니다. 원 문제의 계산 복잡도는 훈련 샘플 수 m에 비례하지만, 쌍대 형식의 계산 복잡도는 m<sup>2</sup>과 m<sup>3</sup> 사이의 값에 비례합니다. 그러므로 수백만 개의 샘플이 있다면 쌍대 형식은 너무 느려질 것이므로 원 문제를 사용해야 합니다.

   

6. RBF 커널을 사용해 SVM 분류기를 훈련시켰더니 훈련 세트에 과소적합된 것 같습니다. γ(gamma)를 증가시켜야 할까요, 감소시켜야 할까요? C의 경우는 어떤가요?

   RBF 커널에 훈련된 SVM 분류기가 훈련 세트에 과소적합이라면 규제가 너무 큰 것일 수 있습니다. 규제를 줄이려면 gamma나 C(또는 둘 다) 값을 증가시켜야 합니다.