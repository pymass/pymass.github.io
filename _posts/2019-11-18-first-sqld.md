---
published: true
layout: single
title: "SQLD 이론 정리 - 1부"
category: SQL
comments: true
toc: true
toc_sticky: true
use_math: true
---

비전공자로서 데이터 분석 분야로 취업하려니 내 실력을 객관적으로 증명할 길은 자격증(혹은 공모전 수상)밖에 없는 것 같다.

7월에 ADSP는 취득했고, 11월 30일 SQLD 시험 접수를 해두었다.

SQLD 시험범위는 두 부분으로 나뉘며 1부에는 데이터 모델의 이해 및 분석 부분을 요약해 보겠다.

<br/>

# 1. 데이터 모델링의 이해

**데이터 모델링의 특징**

- 추상화 : 현실세계를 간략하게 표현한다.
- 단순화 : 누구나 쉽게 이해할 수 있도록 표현한다.
- 명확성 : 명확하게 의미가 해석되어야 하고 한 가지 의미를 가져야 한다.

<br/>

**데이터 모델링 단계**

데이터 모델링은 개념적 → 논리적 → 물리적 모델링 단계로 이루어진다.

1. 개념적 모델링
   - 전사적 관점
   - 추상화 수준이 가장 높음
   - 업무 측면에서 모델링

2. 논리적 모델링
   - 특정 데이터베이스 모델에 종속
   - 식별자 정의
   - 정규화를 통한 재사용성

3. 물리적 모델링
   - 데이터베이스 구축

<br/>

**데이터 모델링 관점**

- 데이터 : 구조분석, 정적분석
- 프로세스 : 시나리오 분석, 도메인 분석, 동적 분석
- 데이터와 프로세스 : CRUD 분석

<br/>

**ERD 작성절차**

1. 엔터티를 도출하고 그린다
2. 엔터티를 배치한다
3. 엔터티 간의 관계를 설정한다
4. 관계명을 서술한다
5. 관계 참여도를 표현한다
6. 관계의 필수 여부를 표현한다

<br/>

**데이터 모델링 고려사항**

1. 데이터 모델의 독립성
   - 중복된 데이터가 없어야한다(정규화를 통해 제거 가능)

2. 고객 요구사항의 표현
   - 간결하게 표현 가능해야 한다

3. 데이터 품질 확보

<br/>

**3층 스키마**

- 사용자, 설계자, 개발자가 데이터베이스를 보는 관점에 따라 데이터베이스를 기술하고 관계를 정의한 ANSI 표준
- 데이터베이스의 독립성 확보를 위한 방법
  - 논리적 독립성 : 저장구조가 변경되어도 응용 프로그램 및 개념 스키마에 영향이 없다
  - 물리적 독립성 : 데이터베이스 논리적 구조가 변경되어도 응용 프로그램에 변화가 없다

<br/>

**3층 스키마 구조**

- 외부 스키마(사용자 관점)
  - 데이터베이스의 뷰를 표시한다
  - 응용 프로그램이 접근하는 데이터베이스를 정의한다

- 개념 스키마(설계자 관점)
  - 통합 데이터베이스의 구조이다

- 내부 스키마(개발자 관점)
  - 물리적 저장구조이다

<br/>

**엔터티**

엔터티는 저장되고 관리되어야 하는 데이터이다

- 식별자 : 유일한 식별자가 있어야 한다
- 인스턴스 집합 : 2개 이상의 인스턴스가 있어야 한다
- 속성 : 반드시 속성을 가지고 있다
- 관계 : 다른 엔터티와 최소 한 개 이상 관계가 있어야 한다
- 업무 : 업무에서 관리되어야 하는 집합이다

<br/>

**엔터티 종류**

1. 유형과 무형에 따른 엔터티 종류
   - 유형 엔터티 : 지속적으로 사용되는 엔터티
   - 개념 엔터티 : 물리적 형태가 없는 엔터티
   - 사건 엔터티 : 비즈니스 프로세스를 실행하면서 생성되는 엔터티

2. 발생시점에 따른 엔터티 종류
   - 기본 엔터티(키 엔터티) : 독립적으로 생성되는 엔터티
   - 중심 엔터티 : 기본 엔터티로부터 발생되고 행위 엔터티를 생성하는 것
   - 행위 엔터티 : 2개 이상의 엔터티로부터 발생하는 엔터티

<br/>

**속성**

- 속성은 업무에서 관리되는 정보이다
- 속성은 하나의 값만 가진다
- 기본키가 변경되면 속성값도 변경된다

<br/>

**속성의 종류**

1. 분해여부에 따른 속성의 종류
   - 단일 속성 : 하나의 의미로 구성된 것
   - 복합 속성 : 여러개의 의미로 구성된 것
   - 다중값 속성 : 여러 개의 값을 가질 수 있는 것

2. 특성에 따른 속성의 종류
   - 기본 속성 : 본래의 속성
   - 설계 속성 : 데이터 모델링 과정에서 발생되는 속성 (유일한 값을 부여)
   - 파생 속성 : 다른 속성에 의해서 만들어지는 속성

<br/>

**관계**

- 관계는 엔터티 간의 관련성을 의미한다
- 존재 관계와 행위 관계로 나누어진다

<br/>

**관계의 종류**

- 존재 관계 : 엔터티 간의 상태를 의미한다.
- 행위 관계 : 엔터티 간에 어떤 행위가 있는 것을 말한다.

<br/>

**관계 차수**

관계 차수는 두 개의 엔터티 간에 관계를 참여하는 수를 의미한다

1. 1대1 관계
   - 완전 1대1 : 하나의 엔터티에 관계되는 엔터티의 관계가 하나 있는 경우(반드시 존재)
   - 선택적 1대1 : 하나의 엔터티에 관계되는 엔터티가 하나이거나 없을 수도 있다

2. 1대N 관계
   - 엔터티에 행이 하나 있을 때 다른 엔터티의 값이 여러 개 있는 관계

3. M대N 관계
   - 두 개 엔터티가 서로 여러 개의 관계를 가지고 있는 것

4. 필수적 관계와 선택적 관계
   - 필수적 관계 : 반드시 하나가 있어야 하는 관계 ('l'로 표현)
   - 선택적 관계 : 없을 수도 있는 관계('O'로 표현)

<br/>

**식별 관계와 비식별 관계**

1. 식별 관계

   - 식별 관계란 강한 개체의 기본키를 하나로 공유하는 것
- 강한 개체는 어떤 다른 엔터티에 의존하지 않고 독립적으로 존재한다
   - 강한 개체는 다른 엔터티에게 기본키를 공유한다
   - 강한 개체는 식별 관계로 표현된다
   
2. 비식별 관계
   - 비식별 관계란 강한 개체의 기본키를 다른 엔터티의 기본키가 아닌 일반 칼럼으로 관계를 가지는 것

<br/>

**주식별자(기본키, Primary key)**

- 유일성과 최소성을 만족
- 엔터티를 대표
- 엔터티의 인스턴스를 유일하게 식별
- 자주 변경되지 않아야 한다

<br/>

**키의 종류**

- 기본키 : 후보키 중에서 엔터티를 대표할 수 있는 키이다
- 후보키 : 후보키는 유일성과 최소성을 만족하는 키이다
- 슈퍼키 : 슈퍼키는 유일성은 만족하지만 최소성(Not Null)을 만족하지 않는 키이다
- 대체키 : 대체키는 여러 개의 후보키 중에서 기본키를 선정하고 남은 키이다

<br/>

**식별자의 종류**

1. 대표성 여부에 따른 식별자의 종류
   - 주식별자 : 엔터티를 대표하는 식별자, 다른 엔터티와 참조 관계로 연결 가능
   - 보조 식별자 : 유일성과 최소성은 만족하지만 <u>대표성을 만족하지 못하는 식별자</u>

2. 생성 여부에 따른 식별자의 종류
   - 내부 식별자 : 엔터티 내부에서 스스로 생성하는 식별자이다
   - 외부 식별자 : 다른 엔터티의 관계로 인하여 만들어지는 식별자이다

3. 속성의 수에 따른 식별자의 종류
   - 단일 식별자 : 하나의 속성으로 구성
   - 복합 식별자 : 두 개 이상의 속성으로 구성

4. 대체 여부에 따른 식별자의 종류
   - 본질 식별자 : 비즈니스 프로세스에서 만들어지는 식별자
   - 인조 식별자 : 인위적으로 만들어지는 식별자

<br/>

# 2. 데이터 모델과 성능

**정규화**

- 정규화는 데이터 중복을 제거하고 데이터 모델의 독립성을 확보하는 방법
- 정규화는 데이터를 분해하는 과정
- 정규화는 제1정규화부터 제5정규화까지 있지만, 실질적으로 제3정규화까지만 수행
- 정규화를 수행하면 데이터 모델의 변경 최소화 가능
- 정규화된 모델은 테이블이 분해된다

<br/>

**정규화 절차**

1. 제1정규화 : 속성의 원자성 확보, 기본키를 설정
2. 제2정규화 : 기본키가 2개 이상의 속성으로 이루어진 경우, 부분 함수 종속성을 제거한다
3. 제3정규화 : 기본키를 제외한 칼럼 간에 종속성을 제거한다(= 이행 함수 종속성 제거)
4. BCNF : 기본키를 제외하고 후보키가 있는 경우, 후보키가 기본키를 종속시키면 분해한다
5. 제4정규화 : 여러 칼럼들이 하나의 칼럼을 종속시키는 경우 분해하여 다중 값 종속성을 제거한다
6. 제5정규화 : 조인에 의해서 종속성이 발생되는 경우 분해한다

<br/>

**함수적 종속성**

X→Y이면 Y는 X에 함수적으로 종속한다고 말한다. 함수적 종속성은 X가 변화하면 Y도 변화하는지 확인한다

<br/>

**정규화의 문제점**

- 정규화는 데이터를 분해해서 중복을 제거하기 때문에 데이터 모델의 유연성을 높인다
- 정규화는 데이터 조회시에 조인을 유발하기 때문에 CPU와 메모리를 많이 사용한다
- 정규화는 조인으로 인하여 성능이 저하된다. 그래서 이 문제를 해결하기 위해 반정규화를 사용한다

<br/>

**반정규화**

- 데이터베이스의 성능 향상을 위하여, 데이터 중복을 허용하고 조인을 줄이는 데이터베이스 성능 향상 방법이다
- 반정규화는 조회 속도를 향상하지만, 데이터 모델의 유연성은 낮아진다

<br/>

**반정규화를 수행하는 경우**

- 정규화에 충실하면 종속성, 활용성은 향상되지만 수행 속도가 느려지는 경우
- 다량의 범위를 자주 처리해야 하는 경우
- 특정 범위의 데이터만 자주 처리하는 경우
- 요약/집계 정보가 자주 요구되는 경우

<br/>

**반정규화 절차**

1. 대상 조사 및 검토 : 데이터 처리 범위, 통계성 등을 확인해서 반정규화 대상을 조사한다
2. 다른 방법 검토 : 반정규화 수행 전 다른 방법을 검토한다
3. 반정규화 수행 : 테이블, 속성, 관계 등을 반정규화 한다

<br/>

**반정규화 기법**

1. 계산된 칼럼 추가 : 결과를 미리 계산하여 칼럼에 추가한다
2. 테이블 수직분할 : 하나의 테이블을 두 개 이상의 테이블로 분할한다
3. 테이블 수평분할 : 하나의 테이블에 있는 값을 기준으로 테이블을 분할한다
4. 테이블 병합
   - 1대1 관계의 테이블을 하나의 테이블로 병합하여 성능을 향상
   - 1대N 관계의 테이블을 병합하여 성능을 향상 (하지만 중복이 많이 발생)
   - 슈퍼 타입과 서브 타입 관계가 발생하면 테이블을 통합하여 성능을 향상

<br/>

**슈퍼 타입 및 서브 타입 변환 방법**

- OneToOne Type : 슈퍼 타입과 서브 타입을 개별 테이블로 도출한다 (조인이 많이 발생하고 관리가 어렵다)
- Plus Type : 슈퍼 타입과 서브 타입 테이블로 도출한다 (조인이 발생하고 관리가 어렵다)
- Single Type : 슈퍼 타입과 서브 타입을 하나의 테이블로 도출한다 (조인 성능이 좋고 관리가 편리하지만, I/O 성능이 나쁘다)

<br/>

**분산 데이터베이스**

- 물리적으로 떨어진 데이터베이스에 네트워크로 연결하여 단일 데이터베이스 이미지를 보여 주고 분산된 작업 처리를 수행하는 데이터베이스를 분산 데이터베이스라고 한다.
- 데이터베이스는 투명성을 제공해야 한다

<br/>

**분산 데이터베이스 투명성의 종류**

- 분할 투명성 : 고객은 하나의 논리적 릴레이션이 여러 단편으로 분할되어 각 단편의 사본이 여러 시스템에 저장되어 있음을 인식할 필요가 없다
- 위치 투명성 : 고객은 사용하려는 데이터의 저장 장소를 명시할 필요가 없다, 고객은 어느 위치에 있더라도 동일한 명령을 사용하여 데이터에 접근할 수 있어야 한다
- 지역 사상 투명성 : 지역 DBMS와 물리적 데이터베이스 사이의 사상이 보장됨에 따라 각 지역 시스템 이름과 무관한 이름이 사용 가능하다
- 중복 투명성 : 데이터베이스 객체가 여러 시스템에 중복되어 존재함에도 고객과는 무관하게 데이터의 일관성이 유지된다
- 장애 투명성 : 데이터베이스가 분산되어 있는 각 지역의 시스템이나 통신망에 이상이 발생해도, 데이터의 무결성은 보장된다
- 병행 투명성 : 여러 고객의 응용 프로그램이 동시에 분산 데이터베이스에 대한 트랜잭션을 수행하는 경우에도 결과에 이상이 없다

<br/>

**분산 데이터베이스 설계 방식**

- 상향식 설계방식 : 지역 스키마 작성 후 향후 전역 스키마를 작성하여 분산 데이터베이스를 구축한다
- 하향식 설계방식 : 전역 스키마 작성 후 해당 지역 스키마를 작성하여 분산 데이터베이스를 구축한다

<br/>

**분산 데이터베이스 장점과 단점**

| 장점                                                         | 단점                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 데이터베이스 신뢰성과 가용성이 높다                          | 데이터베이스가 여러 네트워크를 통해서 분리되어 있기 때문에 관리와 통제가 어렵다 |
| 분산 데이터베이스가 병렬 처리를 수행하기 때문에 빠른 응답이 가능하다 | 보안관리가 어렵다                                            |
| 분산 데이터베이스를 추가하여 시스템 용량 확장이 쉽다         | 데이터 무결성 관리가 어렵다                                  |
|                                                              | 데이터베이스 설계가 복잡하다                                 |