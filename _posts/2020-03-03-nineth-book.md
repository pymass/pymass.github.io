---
published: true
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

```python
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print('Epoch', epoch, 'MSE =', mse.eval())
        sess.run(training_op)
        
    best_theta = theta.eval()
```



### 자동 미분 사용

텐서플로의 자동 미분 기능이 있습니다. 이 기능은 자동으로 그리고 효율적으로 그래디언트를 계산합니다. 앞 절의 경사 하강법 코드에서 gradients = ...줄만 다음 코드로 바꾸면 이상 없이 잘 작동할 것입니다.  

`gradients = tf.gradients(mse, [theta])[0]`  

gradients() 함수는 하나의 연산과 변수 리스트를 받아 각 변수에 대한 연산의 그래디언트를 계산하는 새로운 연산을 만듭니다.

자동으로 그래디언트를 계산하는 방법은 네 가지입니다. 텐서플로는 후진 모드 자동 미분을 사용합니다. 신경망에서처럼 입력이 많고 출력이 적을 때 완벽한 (효율적이고 정확한) 방법입니다.

| 기법                | 모든 그래디언트를 계산하기 위한 그래프 순회 수 | 정확도 | 임의의 코드 지원 | 비고                     |
| ------------------- | ---------------------------------------------- | ------ | ---------------- | ------------------------ |
| 수치 미분           | n_inputs + 1                                   | 낮음   | 지원             | 구현하기 쉬움            |
| 기호 미분           | 해당 없음                                      | 높음   | 미지원           | 상이한 그래프 생성       |
| 전진 모드 자동 미분 | n_inputs                                       | 높음   | 지원             | 이원수 사용              |
| 후진 모드 자동 미분 | n_outputs + 1                                  | 높음   | 지원             | 텐서플로가 사용하는 방식 |



### 옵티마이저 사용

텐서플로는 경사 하강법 옵티마이저를 포함하여 여러 가지 내장 옵티마이저를 제공합니다.

```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
```

다른 옵티마이저를 사용하고 싶으면 한 줄만 바꾸면 됩니다. 예를 들어 모멘텀 옵티마이저는 다음과 같이 사용할 수 있습니다.

`optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)`



## 훈련 알고리즘에 데이터 주입

플레이스홀더 노드는 실제로 아무 계산도 하지 않는 특수한 노드입니다. 실행 시에 주입한 데이터를 출력하기만 합니다. 이 노드는 전형적으로 훈련을 하는 동안 텐서플로에 훈련 데이터를 전달하기 위해 사용됩니다. 실행 시 플레이스홀더에 값을 지정하지 않으면 예외가 발생합니다.

```python
>>> A = tf.placeholder(tf.float32, shape=(None, 3))
>>> B = A + 5
>>> with tf.Session() as sess:
...    B_val_1 = B.eval(feed_dict={A: [[1,2,3]]})
...    B_val_2 = B.eval(feed_dict={A: [[4,5,6], [7,8,9]]})
...
>>> print(B_val_1)
[[6. 7. 8.]]
>>> print(B_val_2)
[[ 9. 10. 11.]
 [12. 13. 14.]]
```

미니배치 경사 하강법을 구현하기 위해서는 기존 코드를 조금만 변경하면 됩니다.

```python
X = tf.placeholder(tf.float32, shape=(None, n+1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
```

그런 다음 배치 크기와 전체 배치 횟수를 정의합니다.

```python
batch_size = 100
n_batchs = int(np.ceil(m / batch_size))
```

마지막으로 실행 단계에서 X와 y에 의존하는 노드를 평가할 때 미니배치를 하나씩 추출하여 feed_dict 매개변수로 전달합니다.

```python
def fetch_batch(epoch, batch_index, batch_size):
	[...]
	return X_batch, y_batch

with tf.Session() as sess:
	sess.run(init)
	
	for epoch in range(n_epochs):
		for batch_index in range(n_batchs):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
			
	best_theta = theta.eval()
```



## 모델 저장과 복원

텐서플로에서 모델을 저장하는 일은 매우 쉽습니다. 구성 단계의 끝에서 (모든 변수 노드를 생성한 후) Saver 노드를 추가하고, 실행 단계에서 모델을 저장하고 싶을 때 save() 메서드에 세션과 체크포인트 파일의 경로를 전달하여 호출하면 됩니다.

```python
[...]
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name='theta')
[...]
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)
	
	for epoch in range(n_epochs):
		if epoch % 100 == 0:
			save_path = saver.save(sess, '/tmp/my_model.ckpt')
			
		sess.run(training_op)
		
	best_theta = theta.eval()
	save_path = saver.save(sess, '/tmp/my_model_final.ckpt')
```

모델을 복원하는 것도 쉽습니다.

```python
with tf.Session() as sess:
	saver.restore(sess, '/tmp/my_model_final.ckpt')
	[...]
```

Saver는 기본적으로 모든 변수를 각자의 이름으로 저장하고 복원합니다. 하지만 더 세부적으로 제어하려면 저장 또는 복원할 변수를 지정하거나 별도의 이름을 사용할 수 있습니다.

`saver = tf.train.Saver({'weights': theta})`

또한 save() 메서드는 기본적으로 .meta 확장자를 가진 동일 이름의 두 번째 파일에 그래프의 구조를 저장합니다. tf.train.import_meta_graph()를 사용해 이 그래프 구조를 읽어 들일 수 있습니다.

```python
saver = tf.train.import_meta_graph('/tmp/my_model.ckpt.meta')

with tf.Session() as sess:
	saver.restore(sess, '/tmp/my_model_final.ckpt')
	[...]
```



## 이름 범위

이름 범위를 만들어 관련 있는 노드들을 그룹으로 묶어야 합니다. 예를 들어 이전 코드를 수정해 'loss' 이름 범위 안에 있는 error와 mse를 정의해보겠습니다.

```python
with tf.name_scope('loss') as scope:
	error = y_perd - y
	mse = tf.reduce_mean(tf.square(error), name='mse')
```

이제 이 범위 안에 있는 모든 연산의 이름에는 'loss/' 접두사가 붙습니다.

```python
>>> print(error.op.name)
loss/sub
>>> print(mse.op.name)
loss/mse
```



## 모듈화

```python
n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')

w1 = tf.Variable(tf.random_normal((n_features, 1)), name='weights1')
w2 = tf.Variable(tf.random_normal((n_features, 1)), name='weights2')
b1 = tf.Variable(0.0, name='bias1')
b2 = tf.Variable(0.0, name='bias2')

z1 = tf.add(tf.matmul(X, w1), b1, name='z1')
z2 = tf.add(tf.matmul(X, w2), b1, name='z2')

relu1 = tf.maximum(z1, 0., name='relu1')
relu2 = tf.maximum(z2, 0., name='relu2')

output = tf.add(relu1, relu2, name='output')
```

이런 반복적인 코드는 유지 보수하기 어렵고 에러가 발생하기 쉽습니다. 다행히 텐서플로에서 DRY 원칙을 유지하게 도와줍니다. 간단하게 ReLU를 구현하는 함수를 만들면 됩니다.

```python
def relu(X):
	w_shape = (int(X.get_shape()[1]), 1)
	w = tf.Variable(tf.random_normal(w_shape), name='weights')
	b = tf.Variable(0.0, name='bias')
	z = tf.add(tf.matmul(X, w), b, name='z')
	return tf.maximum(z, 0., name='relu')
	
n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name='output')
```

이름 범위를 사용하면 그래프르 훨씬 깔끔하게 표현할 수 있습니다. relu() 함수 안의 내용을 모두 이름 범위 아래로 옮기면 됩니다.

```python
def relu(X):
	with tf.name_scope('relu'):
		[...]
```



## 변수 공유

그래프의 여러 구성 요소 간에 변수를 공유하고 싶다면, 간단한 해결 방법은 변수를 먼저 만들고 필요한 함수에 매개변수로 전달하는 것입니다.

```python
def relu(X):
	with tf.name_scope('relu'):
		[...]
		return tf.maximum(z, threshold, name='max')

threshold = tf.Variable(0.0, name='threshold')
X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name='output')
```

threshold 변수로 모든 ReLU의 임곗값을 조절할 수 있습니다. 하지만 이런 공유 변수가 많으면 항상 매개변수로 전달해야 하므로 번거로워집니다. 많은 사람이 모델에 있는 모든 변수를 담을 파이썬 딕셔너리를 만들고 함수마다 이를 전달하는 방식을 사용합니다. 어떤 사람은 모듈화를 위해 파이썬 클래스를 만듭니다. 또 하나의 선택은 relu()를 맨 처음 호출할 때 함수의 속성으로 다음과 같이 공유 변수를 지정하는 것입니다.

```python
def relu(X):
	with tf.name_scope('relu'):
		if not hasattr(relu, 'threshold'):
			relu.threshold = tf.Variable(0.0, name='threshold')
		[...]
		return tf.maximum(z, relu.threshold, name='max')
```

텐서플로에서는 이것보다 더 깔끔하고 모듈화하기 좋은 다른 방법을 제공합니다. 기본 아이디어는 get_variable() 함수를 사용해 공유 변수가 아직 존재하지 않을 때는 새로 만들고 이미 있을 때는 재사용하는 것입니다.

```python
with tf.name_scope('relu'):
	threshold = tf.get_variable('threshold', shape=(), initializer=tf.constant_initializer(0.0))
```

변수를 재사용하고 싶다면 명시적으로 변수 범위의 reuse 속성을 True로 지정해야 합니다.

```python
with tf.name_scope('relu', reuse=True):
	threshold = tf.get_variable('threshold')
```

이 코드는 이미 존재하는 relu/threshold 변수를 가져오며, 존재하지 않거나 get_variable()로 만들지 않은 변수일 경우에는 예외가 발생합니다. 또한 변수 범위의 블록 안에서 reuse_variable() 메서드를 호출하여 reuse 속성을 True로 설정할 수도 있습니다.

```python
with tf.variable_scope('relu') as scope:
	scope.reuse_variables()
	threshold = tf.get_variable('threshold')
```

이제 매개변수로 전달하지 않고 threshold 변수를 공유하도록 relu() 함수를 만들기 위한 준비가 되었습니다.

```python
def relu(X):
	with tf.name_scope('relu', reuse=True):
		threshold = tf.get_variable('threshold')
		[...]
		return tf.maximum(z, threshold, name='max')
		
X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
with tf.name_scope('relu'):
	threshold = tf.get_variable('threshold', shape=(), initializer=tf.constant_initializer(0.0))
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name='output')
```

모든 ReLu 코드가 relu() 함수 안에 있지만 threshold 변수는 함수 밖에서 정의되어야 합니다. 이를 개선하기 위해 다음 코드는 처음 호출될 때 relu() 함수 안에서 threshold 변수를 생성하고 그다음부터 호출될 때는 이 변수를 재사용하고 있습니다.

```python
def relu(X):
	with tf.name_scope('relu', reuse=True):
		threshold = tf.get_variable('threshold')
		[...]
		return tf.maximum(z, threshold, name='max')
		
X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
relus = []
for relu_index in range(5):
	with tf.variable_scope('relu', reuse=(relu_index >= 1)) as scope:
		relus.append(relu(X))
output = tf.add_n(relus, name='output')
```



# 연습문제

1. 계산을 직접 실행하지 않고 계산 그래프를 만드는 주요 장점과 단점은 무엇인가요?

   바로 계산을 실행하지 않고 그래프를 만드는 주요 장점과 단점은 다음과 같습니다.

   - 주요 장점

     - 텐서플로가 자동으로 그래디언트를 계산할 수 있습니다(후진 모드 자동 미분을 사용하여).
     - 텐서플로가 여러 스레드에서 연산을 병렬로 실행할 수 있습니다.
     - 동일한 모델을 여러 장치에 걸쳐 실행시키기 편리합니다.
     - 내부 구조를 살피기 쉽습니다. 예를 들어 텐서보드에서 모델을 시각화할 수 있습니다.

   - 주요 단점

     - 익숙하게 다루려면 시간이 필요합니다.

     - 단계별 디버깅을 수행하기 어렵습니다.

       

2. a_val = a.eval(session=sess)와 a_val = sess.run(a)는 동일한 문장인가요?

   예. a_val = a.eval(session=sess)는 a_val = sess.run(a)와 완전히 동일합니다.

   

3. a_val, b_val = a.eval(session=sess), b.eval(session=sess)와 a_val, b_val = sess.run([a, b])는 동일한 문장인가요?

   아니오. a_val, b_val = a.eval(session=sess), b.eval(session=sess)는 a_val, b_val = sess.run([a, b])와 동일하지 않습니다. 첫 번째 문장은 그래프를 두 번(한 번은 a, 또 한 번은 b를 계산하기 위해) 실행하지만, 두 번째 문장은 그래프를 한 번만 실행합니다. 이 연산(또는 의존하는 다른 연산)이 부수효과를 일으키면(예를 들어 변수가 수정되거나, 큐에 아이템이 추가되거나, 리더가 파일을 읽으면) 결과가 달라질 것입니다. 만약 부수효과가 없다면 두 문장은 동일한 결과를 반환하지만 두 번째 문장이 첫 번째 문장보다 속도가 더 빠를 것입니다.

   

4. 같은 세션에서 두 개의 그래프를 실행할 수 있나요?

   아니오. 같은 세션에서 두 개의 그래프를 실행할 수 없습니다. 먼저 두 개의 그래프를 하나의 그래프로 합쳐야 합니다.

   

5. 만약 변수 w를 가진 그래프 g를 만들고 스레드 두 개를 시작해 각 스레드에서 동일한 그래프 g를 사용하는 세션을 열면, 각 세션은 변수 w를 따로 가지게 될까요? 아니면 공유할까요?

   로컬 텐서플로에서는 세션이 변숫값을 관리하므로 만약 변수 w를 가진 그래프 g를 만들고 동일한 그래프 g를 사용하는 두 개의 스레드를 시작해 각 스레드에서 로컬 세션을 열면 각 세션은 변수 w의 복사본을 각자 가지게 될 것입니다. 그러나 분산 텐서플로에서는 변숫값이 클러스터에 의해 관리되는 컨테이너에 저장됩니다. 그러므로 두 개의 세션이 같은 클러스터에 접속하여 같은 컨테이너를 사용하면 동일한 변수 w의 값을 공유할 것입니다.

   

6. 변수는 언제 초기화되고 언제 소멸되나요?

   변수는 초기화 함수가 호출될 때 초기화되고, 세션이 종료될 때 소멸됩니다. 분산 텐서플로에서는 변수가 클러스터에 있는 컨테이너에 존재하기 때문에 세션을 종료해도 변수가 소멸되지 않으며 변수를 삭제하려면 컨테이너를 리셋해야 합니다.

   

7. 플레이스홀더와 변수의 차이점은 무엇인가요?

   변수와 플레이스홀더는 매우 다르지만 초보자는 혼돈하기 쉽습니다.

   - 변수는 값을 가진 연산입니다. 변수를 실행하면 값이 반환됩니다. 변수는 실행하기 전에 초기화해야 합니다. 또한 변수의 값을 바꿀 수 있습니다(예를 들면 할당 연산을 사용하여). 변수는 상태를 가집니다. 즉, 그래프를 연속해서 실행할 때 변수는 동일한 값을 유지합니다. 일반적으로 변수는 모델 파라미터를 저장하는 데 사용하지만 다른 목적으로도 쓰입니다(예를 들면 전체 훈련 스텝을 카운트하기 위해).

   - 플레이스홀더는 기술적으로 봤을 때 많은 일을 하지 않습니다. 표현하려는 텐서의 크기와 타입에 관한 정보를 가지고 있을 뿐 아무런 값도 가지고 있지 않습니다. 실제로 플레이스홀더에 의존하고 있는 연산을 평가하려면 플레이스홀더의 값을 (feed_dict 매개변수를 통해) 텐서플로에 제공해야 합니다. 그렇지 않으면 예외가 발생할 것입니다. 일반적으로 플레이스홀더는 실행 단계에서 텐서플로에 훈련 데이터와 테스트 데이터를 주입하기 위해 사용됩니다. 또한 변수의 값을 바꾸기 위해 할당 연산 노드에 값을 전달하는 용도로도 사용됩니다.

     

8. 플레이스홀더에 의존하는 연산을 평가하기 위해 그래프를 실행할 때 플레이스홀더에 값을 주입하지 않으면 어떻게 될까요? 플레이스홀더에 의존하지 않는 연산이라면 어떻게 될까요?

   플레이스홀더에 의존하는 연산을 평가할 때 플레이스홀더에 값을 주입하지 않으면 예외가 발생합니다. 플레이스홀더에 의존하지 않는 연산이라면 예외가 발생하지 않습니다.

   

9. 그래프를 실행할 때 어떤 연산자의 출력값을 주입할 수 있나요? 아니면 플레이스홀더의 값만 가능할까요?

   그래프를 실행할 때 플레이스홀더뿐만 아니라 어떤 연산의 출력값도 주입할 수 있습니다. 그러나 실제로는 이는 매우 드문 경우입니다.

   

10. (실행 단계에서) 변수에 원하는 값을 어떻게 설정할 수 있나요?

    그래프를 구성할 때 변수의 초기화 값을 지정할 수 있고, 나중에 실행 단계에서 변수의 초기화 함수를 실행할 때 초기화될 것입니다. 실행 단계에서 변수의 값을 변경하는 간단한 방법은 (구성 단계에서) tf.assign()을 이용한 할당 노드를 만들고 매개변수로 변수와 플레이스홀더를 전달하는 것입니다. 그리고 실행 단계에서 플레이스홀더를 사용해 변수의 새로운 값을 주입하여 할당 연산을 실행합니다.

    

11. 후진 모드 자동 미분으로 변수 10개에 대한 비용 함수의 그래디언트를 계산하려면 그래프를 몇 번 순회해야 하나요? 전진 모드 자동 미분이나 기호 미분의 경우는 어떨까요?

    (텐서플로에 구현된) 후진 모드 자동 미분은 변수 개수에 상관없이 변수에 대한 비용 함수의 그래디언트를 계산하기 위해 그래프를 두 번 순회해야 합니다. 반면 전진 모드 자동 미분은 각 변수마다 한 번씩 실행해야 합니다(그러므로 10개의 다른 변수에 대한 그래디언트를 계산하려면 10번 실행해야 합니다). 기호 미분에서는 그래디언트 계산을 위해 다른 그래프를 만듭니다. 그래서 원본 그래프를 순회하지 않습니다(그래디언트를 위한 새 그래프를 만들 때는 제외). 최적화가 매우 잘된 기호 미분 시스템은 모든 변수에 대한 그래디언트를 계산하기 위해 딱 한 번 새 그래디언트 그래프를 실행할 수 있습니다. 하지만 새 그래프가 매우 복잡하고 원본 그래프에 비해 비효율적일 수 있습니다.