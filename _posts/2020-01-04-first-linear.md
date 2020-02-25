---
published: true
layout: single
title: "한양대 선형대수학 강의 Review - 1부"
category: Math
toc: true
use_math: true
---

데이터 분석에 필요한 수학을 꼽으라면 첫번째가 통계학 두번째가 선형대수인 것 같다.

통계학 강의는 작년에 서울대 류근관 교수님의 '경제통계학' 강의를 듣고 이미 포스팅을 올려두었다.

이번에는 선형대수 공부를 하고자 한양대 이상화 교수님의 '선형대수' 강의를 완강했다.

유튜브에 공개되어 있으니 선형대수에 대해 공부하고 싶다면 추천하는 강의다.

통계학 때와 마찬가지로 공부했던 내용을 정리해서 포스팅한다.



## 기본 개념

첫 번째 챕터에 들어가기 전에 필요한 기본 개념이 있어 정리한다.



### 선형성 정의

선형성이란 아래의 두 조건을 동시에 만족하는 것을 말한다.
$$
1.~superposition~:~f(x_1 + x_2) = f(x_1) + f(x_2)\\
2.~homogeniety~:f(ax) = af(x)
$$
결론적으로 아래 식을 만족한다 == 선형성을 가진다.(해당 조건을 가지려면 원점을 지난다.)
$$
f(a_1x_1 + a_2x_2) = a_1f(x_1) + a_2f(x_2)
$$


### Linear Combination(선형결합)

벡터들을 스칼라배와 벡터 덧셈을 통해 조합하여 새로운 벡터를 얻는 연산이다.

$$
c_1v_1 + c_2v_2 + c_3v_3 +\cdots+ c_nv_n
$$
쉽게 말해 위와 같은 표현식을 Linear Combination이라 한다.



### Inner Product(내적)

반대편 벡터에 대해 수선의 발을 내려 투영한다. (결과 값은 실수)
$$
v_1\cdot v_2 = |v_1||v_2|cos\theta
$$


## Chapter1. Gauss Elimination

### 가우스 소거법

아래와 같은 1차 연립방정식을 가우스 소거법으로 풀어보자.
$$
\eqalign{
2u+v+w=5\\
4u-6v=-2\\
-2u+7v+2w=9
}
$$
먼저, 행렬식으로 표현해보자.
$$
\begin{bmatrix}
2 & 1 & 1\\
4 & -6 & 0\\
-2 & 7 & 2
\end{bmatrix}
=
\begin{bmatrix}
5\\
-2\\
9
\end{bmatrix}
$$
가우스 소거법을 적용해보자. 아래와 같이 대각성분 위쪽만 값이 있는 것을 Upper Triangular Form이라 한다.
$$
\begin{bmatrix}
2 & 1 & 1\\
0 & -8 & -2\\
0 & 0 & 1
\end{bmatrix}
=
\begin{bmatrix}
5\\
-12\\
2
\end{bmatrix}
$$


### Matrix multiplication(행렬 곱셈)

1차 연립방정식을 아래와 같이 표현할 수 있다.
$$
\eqalign{
2u+v+w=5\\
4u-6v=-2\\
-2u+7v+2w=9
}
~~~\longrightarrow ~~~
u
\begin{bmatrix}
2\\
4\\
-2
\end{bmatrix}
+
v
\begin{bmatrix}
1\\
-6\\
7
\end{bmatrix}
+
w
\begin{bmatrix}
1\\
0\\
2
\end{bmatrix}
=
\begin{bmatrix}
5\\
-2\\
9
\end{bmatrix}
$$


### LU 분할

특정 행렬 A를 Lower Triangular와 Upper Triangular의 곱으로 분리 가능하다.

Lower Triangular 행렬은 Upper Triangular의 형태로 만들기 위한 값의 모음이다. (가우스 소거법을 하는 과정)



### Permutation Matrix

행의 위치를 바꾸는 연산을 해주는 행렬을 Permutation Matrix라 한다.

Permutation Matrix의 특징은 역행렬과 Transpose한 행렬이 동일하다.

가우스 소거법 시 pivoting이 필요하면 A = P^T^LU의 형태로 풀어준다.



### 역행렬의 특징

1. 역행렬이 항상 존재하지 않는다. 
2. 가우스 소거법이 n개의 피봇을 가질때만 역행렬이 존재한다.
3. 역행렬은 항상 유니크하다. (∴ 역행렬 존재 시 Ax = b에서 x도 유니크하다)
4. Ax = 0이라는 식에서 x가 0이 아니면 A는 역행렬이 존재하지 않는다.
5. 2x2행렬일 때 ad-bc가 0이 아니면 역행렬이 존재한다.
6. Diagonal Matrix의 성분 중 하나라도 0이면 역행렬이 존재하지 않는다.
7. 역행렬은 역순을 갖는다. (ABC)^-1^=C^-1^B^-1^A-^1^