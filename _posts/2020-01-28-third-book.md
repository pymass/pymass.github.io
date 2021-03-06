---
published: true
layout: single
title: "핸즈온 머신러닝 정리 - 3장"
category: Book
toc: true
use_math: true
---

## 분류

### 정밀도와 재현율

불균형한 데이터셋을 다룰 때 정확도를 분류기의 성능 지표로 선호하지 않습니다. 분류기의 성능을 평가하는 더 좋은 방법은 오차 행렬을 조사하는 것입니다. 정밀도와 재현율을 F1 점수라고 하는 하나의 숫자로 만들면 편리할 때가 많습니다. F1 점수는 정밀도와 재현율의 조화 평균입니다.



### 정밀도/재현율 트레이드오프

정밀도를 올리면 재현율이 줄고 그 반대도 마찬가지입니다. 이를 정밀도/재현율 트레이드오프라고 합니다. 분류기의 predict() 메서드 대신 decision_function() 메서드를 호출하면 각 샘플의 점수를 얻을 수 있습니다. 이 점수를 기반으로 원하는 임곗값을 정해 예측을 만들 수 있습니다.

적절한 임곗값을 구하기 위해서는 cross_val_predict() 함수를 사용해 훈련 세트에 있는 모든 샘플의 점수를 구해야 합니다. 하지만 이번에는 예측 결과가 아니라 결정 점수를 반환받도록 지정해야 합니다. 이 점수로 precision_recall_curve() 함수를 사용하여 모든 임곗값에 대한 정밀도와 재현율을 계산할 수 있습니다.



### ROC 곡선

수신기 조작 특성(ROC) 곡선도 이진 분류에서 널리 사용하는 도구입니다. ROC 곡선은 민감도(재현율)에 대한 1 - 특이도 그래프입니다. 곡선 아래의 면적(AUC)을 측정하면 분류기들을 비교할 수 있습니다.

일반적인 법칙은 양성 클래스가 드물거나 거짓 음성보다 거짓 양성이 더 중요할 때 PR 곡선을 사용하고 그렇지 않으면 ROC 곡선을 사용합니다.



### 다중 분류

다중 분류기는 둘 이상의 클래스를 구별할 수 있습니다. 일부 알고리즘은 여러 개의 클래스를 직접 처리할 수 있는 반면, 다른 알고리즘은 이진 분류만 가능합니다. 하지만 이진 분류기를 여러 개 사용해 다중 클래스를 분류하는 기법도 많습니다. 

예를 들어 특정 숫자 하나만 구분하는 숫자별 이진 분류기 10개를 훈련시켜 클래스가 10개인 숫자 이미지 분류 시스템을 만들 수 있습니다. 이미지를 분류할 때 각 분류기의 결정 점수 중에서 가장 높은 것을 클래스로 선택하면 됩니다. 이를 일대다(OvA) 전략이라고 합니다.

또 다른 전략은 0과 1 구별, 0과 2 구별, 1과 2 구별 등과 같이 각 숫자의 조합마다 이진 분류기를 훈련시키는 것입니다. 이를 일대일(OvO) 전략이라고 합니다. 클래스가 N개라면 분류기는 N x (N - 1) / 2개가 필요합니다. 분류기 모두를 통과시켜서 가장 많이 양성으로 분류된 클래스를 선택합니다.

다중 클래스 분류 작업에 이진 분류 알고리즘을 선택하면 사이킷런이 자동으로 감지해 OvA(SVM 분류기일 때는 OvO)를 적용합니다.



### 에러 분석

가능성이 높은 모델을 하나 찾았다고 가정하고 이 모델의 성능을 향상시킬 방법을 찾아보겠습니다. 한 가지 방법은 만들어진 에러의 종류를 분석하는 것입니다. 오차 행렬을 분석하면 분류기의 성능 향상 방안에 대한 통찰을 얻을 수 있습니다. 예를 들어 잘못 분류한 데이터를 더 모을 수 있습니다. 또는 분류기에 도움 될 만한 특성을 더 찾아볼 수 있습니다.



### 다중 레이블 분류

여러 개의 이진 레이블을 출력하는 분류 시스템을 다중 레이블 분류 시스템이라고 합니다. 다중 레이블 분류기를 평가하는 방법은 많습니다. 예를 들어 각 레이블의 F1 점수를 구하고 간단하게 평균 점수를 계산합니다.



### 다중 출력 분류

다중 레이블 분류에서 한 레이블이 다중 클래스가 될 수 있도록 일반화한 것을 다중 출력 다중 클래스 분류라고 합니다.

