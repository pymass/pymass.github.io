---
published: false
layout: single
title: "핸즈온 머신러닝 정리 - 9장"
category: Book
toc: true
use_math: true
---

1장부터 8장까지는 Part1 '머신러닝'에 관련한 내용이었고, 9장부터 16장은 Part2 '신경망과 딥러닝' 내용이다.

텐서플로를 설명하다보니 코드에 관한 내용이 많아서 코드블럭으로 작성하였다.



# 텐서플로 시작하기

텐서플로는 수치 계산을 위한 강력한 오픈소스 소프트웨어 라이브러리로 특히 대규모 머신러닝에 맞춰 세밀하게 튜닝되어 있습니다.



## 첫 번째 계산 그래프를 만들어 세션에서 실행하기

```python
import tensorflow as tf

x = tf.Variable(3, name='x')
y = tf.variable(4, name='y')
f = x*x*y + y + 2
```

꼭 이해해야 할 중요한 점은 이 코드가(특히 마지막 줄에서) 뭔가 계산하는 것 같아 보이지만 실제로 어떤 계산도 수행하지 않는다는 사실입니다. 단지 계산 그래프만 만들 뿐입니다. 사실 변수조차도 초기화되지 않습니다. 이 계산 그래프를 평가하려면 텐서플로 세션을 시작하고 변수를 초기화한 다음 f를 평가해야 합니다.

```python
>>> sess = tf.Session()
>>> sess.run(x.initializer)
>>> sess.run(y.initializer)
>>> result = sess.run(f)
>>> print(result)
42
>>> sess.close()
```

매번 sess.run()을 반복하면 번거로운데, 다행히 더 나은 방법이 있습니다.

```python
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
```

세션은 with 블록이 끝나면 자동으로 종료됩니다.



각 변수의 초기화를 일일이 실행하는 대신 global_variables_initializer() 함수를 사용할 수 있습니다. 이 함수는 초기화를 바로 수행하지 않고 계산 그래프가 실행될 때 모든 변수를 초기화할 노드를 생성합니다.

``` python
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()
```

주피터나 파이썬 셸에서는 InteractiveSession을 만드는 편이 편리할 수 있습니다. 일반적인 Session과 유일하게 다른 점은 InteractiveSession이 만들어질 때 자동으로 자신을 기본 세션으로 지정한다는 점입니다. 그러므로 with 블록을 사용할 필요가 없습니다(하지만 사용이 끝났을 때는 수동으로 세션을 종료해주어야 합니다).

```python
>>> sess = tf.InteractiveSession()
>>> init.run()
>>> result = f.eval()
>>> print(result)
42
>>> sess.close()
```

일반적으로 텐서플로 프로그램은 두 부분으로 나뉩니다. 첫 부분은 계산 그래프를 만들고(**구성 단계**), 두 번째 부분은 이 그래프를 실행합니다(**실행 단계**). 구성 단계에서는 훈련에 필요한 계산과 머신러닝 모델을 표현한 계산 그래프를 만듭니다. 실행 단계에서는 훈련 스텝을 반복해서 평가하고, 모델 파라미터를 점진적으로 개선하기 위해 반복 루프를 수행합니다.



## 계산 그래프 관리

노드를 만들면 자동으로 기본 계산 그래프에 추가됩니다.

```python
>>> x1 = tf.Variable(1)
>>> x1.graph is tf.get_default_graph()
True
```

대부분의 경우 이것으로 충분하지만, 가끔은 독립적인 계산 그래프를 여러 개 만들어야 할 때가 있습니다. 이렇게 하려면 다음과 같이 새로운 Graph 객체를 만들어 with 블록 안에서 임시로 이를 기본 계산 그래프로 사용할 수 있습니다.

```python
>>> graph = tf.Graph()
>>> with graph.as_default():
    	x2 = tf.Variable(2)
...
...
>>> x2.graph is graph
True
>>> x2.graph is tf.get_default_graph()
False
```



## 노드 값의 생애주기

한 노드를 평가할 때 텐서플로는 이 노드가 의존하고 있는 다른 노드들을 자동으로 찾아 먼저 평가합니다.

```python
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval()) # 10
    print(z.eval()) # 15
```

먼저 이 코드는 매우 간단한 그래프를 정의하고 있습니다. 그런 다음 세션을 시작하고 y를 평가하기 위해 계산 그래프를 실행합니다. 텐서플로는 자동으로 y가 x에 의존한다는 것과 x가 w에 의존한다는 것을 감지합니다. 그래서 먼저 w를 평가하고 그다음에 x를, 그다음에 y를 평가해서 y 값을 반환합니다. 마지막으로 z를 평가하기 위해 그래프를 실행합니다. 다시 한번 텐서플로는 먼저 w와 x를 평가해야 한다는 것을 감지합니다. 이전에 평가된 w와 x를 재사용하지 않는다는 점이 중요합니다. 간단히 말해 위 코드는 w와 x를 두 번 평가합니다.

모든 노드의 값은 계산 그래프 실행 간에 유지되지 않습니다. 변숫값은 예외로, 그래프 실행 간에도 세션에 의해 유지됩니다. 변수는 초기화될 때 일생이 시작되고 세션이 종료될 때까지 남아 있습니다.

이전 코드에서처럼 w와 x를 두 번 평가하지 않고 y와 z를 효율적으로 평가하려면 텐서플로가 한 번의 그래프 실행에서 y와 z를 모두 평가하도록 만들어야 합니다.

```python
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val) # 10
    print(z_val) # 15
```



## 텐서플로를 이용한 선형 회귀

텐서플로 연산은 여러 개의 입력을 받아 출력을 만들 수 있습니다. 예를 들어 덧셈과 곱셈 연산은 두 개의 입력을 받아 하나의 출력을 만듭니다. 상수와 변수 연산은 입력이 없습니다(**소스 연산**이라고 합니다). 입력과 출력은 **텐서**라는 다차원 배열입니다. 넘파이 배열과 비슷하게 텐서는 데이터 타입과 크기를 가집니다. 사실 파이썬 API에서 텐서는 넘파이 ndarray로 나타납니다. 보통은 실수로 채워지지만 문자열을 저장할 수도 있습니다.

아래는 캘리포니아 주택 가격 데이터셋에 정규방정식을 이용하여 선형 회귀를 수행한 예제입니다.

```python
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name='y')
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()
```



## 경사 하강법 구현

정규방정식 대신 경사 하강법을 사용해보겠습니다. 먼저 그래디언트를 수동으로 계산해보고 그다음에 텐서플로의 자동 미분 기능을 사용해 그래디언트를 자동으로 계산해보겠습니다. 마지막으로 텐서플로에 내장된 옵티마이저를 사용하겠습니다.



### 직접 그래디언트 계산

- random_uniform() 함수는 난수를 담은 텐서를 생성하는 노드를 그래프에 생성합니다. 넘파이의 rand() 함수처럼 크기와 난수의 범위를 입력받습니다.
- assign() 함수는 변수에 새로운 값을 할당하는 노드를 생성합니다.
- 반복 루프는 훈련 단계를 계속 반복해서 실행하고(n_epoch만큼), 100번 반복마다 현재의 평균 제곱 에러(코드에서 mse 변수)를 출력합니다. MSE는 매 반복에서 값이 줄어들어야 합니다.