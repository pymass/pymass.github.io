---
published: true
layout: single
title: "핸즈온 머신러닝 정리 - 8장"
category: Book
comments: true
toc: true
toc_sticky: true
use_math: true
---

# 차원 축소

많은 경우 머신러닝 문제는 훈련 샘플 각각이 수천 심지어 수백만 개의 특성을 가지고 있습니다. 이는 훈련을 느리게 할 뿐만 아니라, 앞으로 보게 되겠지만 좋은 솔루션을 찾기 어렵게 만듭니다. 이런 문제를 종종 **차원의 저주**라고 합니다.

훈련 속도를 높이는 것 외에 차원 축소는 데이터 시각화에도 아주 유용합니다. 차원 수를 둘로 (또는 셋으로) 줄이면 고차원 훈련 세트를 하나의 그래프로 그릴 수 있고 군집 같은 시각적인 패턴을 감지해 중요한 통찰을 얻는 경우가 많습니다.



## 차원의 저주

단위 면적에서 임의의 두 점을 선택하면 두 점 사이의 거리는 평균적으로 대략 0.52가 됩니다. 3차원 큐브에서 임의의 두 점을 선택하면 평균 거리는 대략 0.66입니다. 만약 1,000,000차원의 초입방체에서 두 점을 무작위로 선택하면 어떨까요? 믿거나 말거나 평균 거리는 약 428.25입니다! 이 사실은 고차원의 데이터셋이 매우 희박한 상태일 수 있음을 의미합니다. 즉, 대부분의 훈련 데이터가 서로 멀리 떨어져 있습니다. 물론 이는 새로운 샘플도 훈련 샘플과 멀리 떨어져 있을 가능성이 높다는 뜻입니다. 이 경우 예측을 위해 많은 외삽을 해야 하기 때문에 저차원일 때보다 예측이 더 불안정합니다. 간단히 말해 훈련 세트의 차원이 클수록 과대 적합 위험이 커집니다.

이론적으로 차원의 저주를 해결하는 해결책 하나는 훈련 샘플의 밀도가 충분히 높아질 때까지 훈련 세트의 크기를 키우는 것입니다. 불행하게도 실제로는 일정 밀도에 도달하기 위해 필요한 훈련 샘플 수는 차원 수가 커짐에 따라 기하급수적으로 늘어납니다.



## 차원 축소를 위한 접근 방법

### 투영

대부분의 실전문제는 훈련 샘플이 모든 차원에 걸쳐 균일하게 퍼져 있지 않습니다. 많은 특성은 거의 변화가 없는 반면, 다른 특성들은 서로 강하게 연관되어 있습니다. 결과적으로 모든 훈련 샘플이 사실 고차원 공간 안의 저차원 **부분 공간**에(또는 가까이) 놓여 있습니다.

그러나 차원 축소에 있어서 투영이 언제나 최선의 방법은 아닙니다. 많은 경우에 스위스 롤 데이터셋처럼 부분 공간이 뒤틀리거나 휘어 있기도 합니다.



### 매니폴드 학습

스위스 롤은 2D 매니폴드의 한 예입니다. 간단히 말해 2D 매니폴드는 고차원 공간에서 휘어지거나 뒤틀린 2D 모양입니다. 더 일반적으로 d차원 매니폴드는 국부적으로 d차원 초평면으로 보일 수 있는 n차원 공간의 일부입니다(d < n).

많은 차원 축소 알고리즘이 훈련 샘플이 놓여 있는 **매니폴드**를 모델링하는 식으로 작동합니다. 이를 **매니폴드 학습**이라고 합니다. 이는 대부분 실제 고차원 데이터셋이 더 낮은 저차원 매니폴드에 가깝게 놓여 있다는 **매니폴드 가정** 또는 **매니폴드 가설**에 근거합니다. 경험적으로도 이런 가정은 매우 자주 발견됩니다.

요약하면 모델을 훈련시키기 전에 훈련 세트의 차원을 감소시키면 훈련 속도는 빨라지지만 항상 더 낫거나 간단한 솔루션이 되는 것은 아닙니다. 이는 전적으로 데이터셋에 달렸습니다.



## PCA

주성분 분석은 가장 인기 있는 차원 축소 알고리즘입니다. 먼저 데이터에 가장 가까운 초평면을 정의한 다음, 데이터를 이 평면에 투영시킵니다.



### 분산 보존

다른 방향으로 투영하는 것보다 분산이 최대로 보존되는 축을 선택하는 것이 정보가 가장 적게 손실되므로 합리적으로 보입니다. 이 선택을 다른 방식으로 설명하면 원본 데이터셋과 투영된 것 사이의 평균 제곱 거리를 최소화하는 축입니다. 이 방식이 PCA를 더 간단하게 설명할 수 있습니다.



### 주성분

PCA는 훈련 세트에서 분산이 최대인 축을 찾습니다. 또한 첫 번째 축에 직교하고 남은 분산을 최대한 보존하는 두 번째 축을 찾습니다. 고차원 데이터셋이라면 PCA는 이전의 두 축에 직교하는 세 번째 축을 찾으며 데이터셋에 있는 차원의 수만큼 네 번째, 다섯 번째, ... 축을 찾습니다. *i*번째 축을 정의하는 단위 벡터를 *i*번째 주성분(PC)이라고 부릅니다.

TIP : 주성분의 방향은 일정치 않습니다. 훈련 세트를 조금 섞은 다음 다시 PCA를 적용하면 새로운 PC 중 일부가 원래 PC와 반대 방향일 수 있습니다. 그러나 일반적으로 같은 축에 놓여 있을 것입니다. 어떤 경우에는 한 쌍의 PC가 회전하거나 서로 바뀔 수 있지만 보통은 같은 평면을 구성합니다.

그럼 훈련 세트의 주성분을 어떻게 찾을까요? 다행히 특잇값 분해(SVD)라는 표준 행렬 분해 기술이 있어서 모든 주성분이 V에 담겨 있습니다.



### d차원으로 투영하기

주성분을 모두 추출해냈다면 처음 d개의 주성분으로 정의한 초평면에 투영하여 데이터셋의 차원을 d차원으로 축소시킬 수 있습니다. 이 초평면은 분산을 가능한 한 최대로 보존하는 투영임을 보장합니다.

초평면에 훈련 세트를 투영하기 위해서는 행렬 X와 첫 d개의 주성분을 담은 (즉, V의 첫 d열로 구성된) 행렬 W<sub>d</sub>를 점곱하면 됩니다.



### 적절한 차원 수 선택하기

축소할 차원 수를 임의로 정하기보다는 충분한 분산 (예를 들면 95%)이 될 때까지 더해야 할 차원 수를 선택하는 쪽을 더 선호합니다. 물론 데이터 시각화를 위해 차원을 축소할 경우에는 차원을 2개나 3개로 줄이는 것이 일반적입니다.



### 압축을 위한 PCA

원본 데이터와 재구성된 데이터 (압축 후 원복한 것) 사이의 평균 제곱 거리를 **재구성 오차**라고 합니다. 



### 점진적 PCA

PCA 구현의 문제는 SVD 알고리즘을 실행하기 위해 전체 훈련 세트를 메모리에 올려야 한다는 것입니다. 다행히 **점진적 PCA** 알고리즘이 개발되었습니다. 훈련 세트를 미니배치로 나눈 뒤 IPCA 알고리즘에 한 번에 하나씩 주입합니다. 이런 훈련 방식은 훈련 세트가 클 때 유용하고 온라인으로 (즉, 새로운 데이터가 준비되는 대로 실시간으로) PCA를 적용할 수도 있습니다.



## 커널 PCA

고차원 특성 공간에서의 선형 결정 경계는 **원본 공간**에서는 복잡한 비선형 결정 경계에 해당한다는 것을 배웠습니다.

같은 기법을 PCA에 적용해 차원 축소를 위한 복잡한 비선형 투형을 수행할 수 있습니다. 이를 **커널 PCA**라고 합니다. 이 기법은 투영된 후에 샘플의 군집을 유지하거나 꼬인 매니폴드에 가까운 데이터셋을 펼칠 때도 유용합니다.



### 커널 선택과 하이퍼파라미터 튜닝

kPCA는 비지도 학습이기 때문에 좋은 커널과 하이퍼파라미터를 선택하기 위한 명확한 성능 측정 기준이 없습니다. 하지만 차원 축소는 종종 지도 학습(예를 들면 분류)의 전처리 단계로 활용되므로 그리드 탐색을 사용하여 주어진 문제에서 성능이 가장 좋은 커널과 하이퍼파라미터를 선택할 수 있습니다.

완전한 비지도 학습 방법으로, 가장 낮은 재구성 오차를 만드는 커널과 하이퍼파라미터를 선택하는 방식도 있습니다. 커널 트릭 덕분에 훈련 세트를 특성 맵을 사용한 무한 차원의 특성 공간에 매핑한 다음, 변환된 데이터셋을 선형 PCA를 사용해 2D로 투영한 것과 수학적으로 동일합니다. 축소된 공간에 있는 샘플에 대해 선형 PCA를 역전시키면 재구성된 데이터 포인트는 원본 공간이 아닌 특성 공간에 놓이게 됩니다. 이 특성 공간은 무한 차원이기 때문에 재구성된 포인트를 계산할 수 없고 재구성에 따른 실제 에러를 계산할 수 없습니다. 이를 **재구성 원상**이라고 부릅니다. 원상을 얻게 되면 원본 샘플과의 제곱 거리를 측정할 수 있습니다. 그래서 재구성 원상의 오차를 최소화하는 커널과 하이퍼파라미터를 선택할 수 있습니다.



## LLE

지역 선형 임베딩(LLE)은 또 다른 강력한 **비선형 차원 축소** 기술(NLDR)입니다. 이전 알고리즘처럼 투영에 의존하지 않는 매니폴드 학습입니다. 간단히 말해 LLE는 먼저 각 훈련 샘플이 가장 가까운 이웃에 얼마나 선형적으로 연관되어 있는지 측정합니다. 그런 다음 국부적인 관계가 가장 잘 보존되는 훈련 세트의 저차원 표현을 찾습니다. 이는 특히 잡음이 너무 많지 않은 경우 꼬인 매니폴드를 펼치는 데 잘 작동합니다.



## 다른 차원 축소 기법

- **다차원 스케일링**(MDS)은 샘플 간의 거리를 보존하면서 차원을 축소합니다.
- **Isomap**은 각 샘플을 가장 가까운 이웃과 연결하는 식으로 그래프를 만듭니다. 그런 다음 샘플 간의 **지오데식 거리**를 유지하면서 차원을 축소합니다.
- **t-SNE**는 비슷한 샘플은 가까이, 비슷하지 않은 샘플은 멀리 떨어지도록 하면서 차원을 축소합니다. 주로 시각화에 많이 사용되며 특히 고차원 공간에 있는 샘플의 군집을 시각화할 때 사용됩니다.
- **선형 판별 분석**(LDA)은 사실 분류 알고리즘입니다. 하지만 훈련 과정에서 클래스 사이를 가장 잘 구분하는 축을 학습합니다. 이 축은 데이터가 투영되는 초평면을 정의하는 데 사용할 수 있습니다. 이 알고리즘의 장점은 투영을 통해 가능한 한 클래스를 멀리 떨어지게 유지시키므로 SVM 분류기 같은 다른 분류 알고리즘을 적용하기 전에 차원을 축소시키는 데 좋습니다.



# 연습문제

1. 데이터셋의 차원을 축소하는 주요 목적은 무엇인가요? 대표적인 단점은 무엇인가요?

   차원 축소의 주요 목적과 단점은 다음과 같습니다.

   - 차원 축소의 주요 목적

     - 훈련 알고리즘의 속도를 높이기 위해(어떤 경우에는 잡음과 중복된 특성을 삭제할 수도 있어 훈련 알고리즘의 성능을 높입니다.)
     - 데이터를 시각화하고 가장 중요한 특성에 대한 통찰을 얻기 위해
     - 메모리 공간을 절약하기 위해(압축)

   - 주요 단점

     - 일부 정보를 잃어버려 훈련 알고리즘의 성능을 감소시킬 수 있습니다.

     - 계산 비용이 높습니다.

     - 머신러닝 파이프라인의 복잡도를 증가시킵니다.

     - 변환된 데이터를 이해하기 어려운 경우가 많습니다.

       

2. 차원의 저주란 무엇인가요?

   차원의 저주는 저차원 공간에는 없는 많은 문제가 고차원 공간에서 일어난다는 사실을 뜻합니다. 머신러닝에서 무작위로 선택한 고차원 벡터는 매우 희소해서 과대적합의 위험이 크고, 많은 양의 데이터가 있지 않으면 데이터에 있는 패턴을 잡아내기 매우 어려운 것이 흔한 현상입니다.

   

3. 데이터셋의 차원을 축소시키고 나서 이 작업을 원복할 수 있나요? 할 수 있다면 어떻게 가능할까요? 가능하지 않다면 왜일까요?

   여기에서 설명한 알고리즘 중 하나를 사용해 데이터셋의 차원이 축소되면 일부 정보가 차원 축소 과정에서 사라지기 때문에 이를 완벽하게 되돌리는 것은 불가능합니다. (PCA 같은) 일부 알고리즘은 비교적 원본과 비슷한 데이터셋을 재구성할 수 있는 간단한 역변환 방법을 가지고 있지만, (T-SNE같은) 다른 알고리즘들은 그렇지 않습니다.

   

4. 매우 비선형적인 데이터셋의 차원을 축소하는 데 PCA를 사용할 수 있을까요?

   PCA는 불필요한 차원을 제거할 수 있기 때문에 매우 비선형적이더라도 대부분의 데이터셋에서 차원을 축소하는 데 사용할 수 있습니다. 그러나 불필요한 차원이 없다면(예를 들면 스위스 롤 데이터셋) PCA의 차원 축소는 너무 많은 정보를 잃게 만듭니다. 즉, 스위스 롤은 펼쳐야 하며 말려진 것을 뭉개면 안됩니다.

   

5. 설명된 분산을 95%로 지정한 PCA를 1,000개의 차원을 가진 데이터셋에 적용한다고 가정하겠습니다. 결과 데이터셋의 차원은 얼마나 될까요?

   이 질문에는 속임수가 있습니다. 답은 데이터셋에 따라 다릅니다. 극단적인 두 가지 사례를 살펴보겠습니다. 먼저 거의 완벽하게 일렬로 늘어선 데이터 포인트로 구성된 데이터셋을 생각해보겠습니다. 이 경우 PCA는 분산의 95%를 유지하면서 데이터셋을 단 하나의 차원으로 줄일 수 있습니다. 이번에는 완전히 무작위로 1,000개의 차원에 걸쳐 흩어져 있는 데이터셋을 생각해보겠습니다. 이 경우 PCA는 분산의 95%를 보존하려면 거의 950개의 차원이 필요할 것입니다. 그러므로 답은 데이터셋에 따라 달라지고 1에서 950 사이의 어떤 수도 될 수 있습니다. 차원 수에 대한 함수로 설명된 분산의 그래프를 그려보는 것이 데이터셋에 내재된 차원 수를 대략 가늠할 수 있는 한 가지 방법입니다.

    

6. 기본 PCA, 점진적 PCA, 랜덤 PCA, 커널 PCA는 어느 경우에 사용될까요?

   기본 PCA가 우선적으로 사용되지만 데이터셋 크기가 메모리에 맞을 때에 가능합니다. 점진적 PCA는 메모리에 담을 수 없는 대용량 데이터셋에 적합합니다. 하지만 기본 PCA보다 느리므로 데이터셋이 메모리 크기에 맞으면 기본 PCA를 사용해야 합니다. 점진적 PCA는 새로운 샘플이 발생될 때마다 실시간으로 PCA를 적용해야 하는 온라인 작업에 사용 가능합니다. 랜덤 PCA는 데이터셋이 메모리 크기에 맞고 차원을 크게 축소시킬 때 사용됩니다. 이 경우에는 기본 PCA보다 훨씬 빠릅니다. 커널 PCA는 비선형 데이터셋에 유용합니다.

   

7. 어떤 데이터셋에 적용한 차원 축소 알고리즘의 성능을 어떻게 평가할 수 있을까요?

   직관적으로 데이터셋에서 너무 많은 정보를 잃지 않고 차원을 많이 제거할 수 있다면 차원 축소 알고리즘이 잘 작동한 것입니다. 이를 측정하는 한 가지 방법은 역변환을 수행해서 재구성 오차를 측정하는 것입니다. 하지만 모든 차원 축소 알고리즘이 역변환을 제공하지는 않습니다. 만약 차원 축소를 다른 단계로 사용한다면 두 번째 알고리즘의 성능을 측정해볼 수 있습니다. 즉, 차원 축소가 너무 많은 정보를 잃지 않았다면 원본 데이터셋을 사용했을 때와 비슷한 성능이 나와야 합니다.

   

8. 두 개의 차원 축소 알고리즘을 연결할 수 있을까요?

   당연히 두 개의 차원 축소 알고리즘을 연결할 수 있습니다. PCA로 불필요한 차원을 대폭 제거하고 난 다음 LLE 같이 훨씬 느린 알고리즘을 적용하는 것이 대표적인 사례입니다. 이런 2단계 방식은 LLE만 사용했을 때와 거의 비슷한 성능을 내지만 속도가 몇 분의 1로 줄어들 것입니다.