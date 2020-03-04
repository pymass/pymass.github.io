---
published: true
layout: single
title: "정보처리기사 필기 정리 - 2과목"
category: Certificate
toc: true
use_math: true
---

정보처리기사 2과목은 **<u>소프트웨어 개발</u>**이다.



# 1. 데이터 입출력 구현



## 1-1. 논리 데이터저장소 확인



### 자료구조

- 선형구조
  - 리스트
    - 선형리스트
    - 연결리스트
  - 스택
  - 큐
  - 데크
- 비선형구조
  - 트리
  - 그래프



- 이진 트리의 순회 방법

  - 전위순회 : **중**간 노드 방문 → 왼쪽 서브 트리 → 오른쪽 서브 트리 

  - 중위순회 :  왼쪽 서브 트리 → **중**간 노드 방문 → 오른쪽 서브 트리

  - 후위순회 : 왼쪽 서브 트리 → 오른쪽 서브 트리 → **중**간 노드 방문

    

### 논리 데이터모델 개요

- 시스템 개발 절차

<table style="border-collapse:collapse;border-spacing:0" class="tg"><tr><th style="font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;text-align:center;vertical-align:top">구분</th><th style="font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;text-align:center;vertical-align:top">데이터 관점</th><th style="font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;text-align:center;vertical-align:top">프로세스 관점</th></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;text-align:center;vertical-align:top">요구분석</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;text-align:center;vertical-align:top" colspan="2">비즈니스 요구 사항</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;text-align:center;vertical-align:top">전략수립</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;text-align:center;vertical-align:top" colspan="2">개념 모델링</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;text-align:center;vertical-align:top">분석</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;text-align:center;vertical-align:top">논리 데이터 모델링</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;text-align:center;vertical-align:top">분석 모델링</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;text-align:center;vertical-align:top">설계</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;text-align:center;vertical-align:top">물리 데이터 모델링</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;text-align:center;vertical-align:top">설계 모델링</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;text-align:center;vertical-align:top">개발</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;text-align:center;vertical-align:top">데이터베이스 구축</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;text-align:center;vertical-align:top">애플리케이션 개발</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;text-align:center;vertical-align:top">운영 시스템</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;text-align:center;vertical-align:top">데이터베이스</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;text-align:center;vertical-align:top">애플리케이션</td></tr></table>



- 데이터 모델링 절차
  - 개념 데이터 모델링
    - 중요 개념을 구분
    - 핵심 개체(Entity) 도출
  - 논리 데이터 모델링
    - 각 개념을 구체화
    - ERD-RDB모델 사상
    - 상세 속성 정의
    - 정규화 등
  - 물리 데이터 모델링
    - 개체, 인덱스 등 생성
    - DB 개체 정의
    - 테이블 및 인덱스 등 설계



- 데이터 정규화

  |        형태        |           조건            |
  | :----------------: | :-----------------------: |
  |      비정규형      |             -             |
  |     제1정규화      |       중복속성 제거       |
  |     제2정규화      |      부분종속성 제거      |
  |     제3정규화      |      이행종속성 제거      |
  | 보이스/코드 정규형 | 후보키가 아닌 결정자 제거 |
  |     제4정규화      |      다치종속성 제거      |
  |     제5정규화      |      조인종속성 제거      |



### 논리 데이터모델 검증

- 논리 데이터저장소 확인절차
  - 엔터티 및 속성 확인
  - 관계 확인
  - 데이터 흐름 확인
  - 데이터 접근 권한 확인
  - 데이터 백업정책 및 분산구조 확인



## 1-2. 물리 데이터저장소 설계



### 물리 데이터모델 설계

- 데이터베이스 스키마 종류
  - 외부 스키마 : 프로그래머나 사용자 입장에서 데이터베이스의 모습으로 조직의 일부분을 정의
  - 개념 스키마 : 데이터베이스 구조를 논리적으로 정의
  - 내부 스키마 : 전체 데이터베이스의 물리적 저장 형태



- 물리 데이터 모델링 (논리 데이터 모델 → 물리 데이터 모델)
  - 단위 엔터티를 테이블로 변환
  - 속성을 칼럼으로 변환
  - UID를 기본키로 변환
  - 관계를 외래키로 변환
  - 칼럼 유형과 길의 정의
  - 데이터 처리 범위와 빈도수를 분석하여 반정규화 고려



### 물리 데이터저장소 구성

- 오브젝트는 디스크 구성 설계를 통해 구성
  - 테이블 제약 조건 : 참조 무결성 관리
  - 인덱스 : 빠른 검색 속도를 위해 설계 (칼럼의 분포도가 10 ~ 15% 이내)
  - 뷰 : 가상 테이블을 설계하여 사용성 높임
  - 클러스터 : 하나 혹은 그 이상의 테이블 설계
  - 파티션 : 대용량 DB에서 성능 저하를 막고 관리를 용이하게 하기 위해 설계
    - 범위분할 : 지정한 열의 값을 기준으로 분할
    - 해시분할 : 해시 함수에 따라 데이터를 분할
    - 조합분할 : 범위분할 + 해시분할



### ORM(Object-Relational Mapping) 프레임워크

- ORM 절차
  - 클래스를 테이블로 변환
  - 속성은 칼럼으로 변환
  - 클래스간 관계는 관계형 테이블 간의 관계로 변환



### 트랜잭션 인터페이스

- 데이터베이스 트랜잭션 특징
  - 원자성
  - 일관성
  - 독립성
  - 영속성



## 1-3. 데이터 조작 프로시저 작성



### 데이터 조작 프로시저 개발

- SQL 분류
  - DDL(데이터 정의어)
  - DML(데이터 조작어)
  - DCL(데이터 제어어)



- 데이터 조작 프로시저 개발하기
  - 데이터 저장소에 연결한다
  - 데이터 저장소를 정의한다
  - 데이터 조작 프로시저를 작성한다
  - 데이터 검색 프로시저를 작성한다
  - 절차형 데이터 조작 프로시저를 작성한다



### 데이터 조작 프로시저 테스트

SQL*Plus는 SQL을 DBMS 서버에 전송하여 처리할 수 있도록 하는 Oracle에서 제공하는 도구



- PL/SQL 저장 객체 테스트
  - Stored Function
  - Stored Procedure
  - Stored Package
  - Trigger



## 1-4. 데이터 조작 프로시저 최적화



### 데이터 조작 프로시저 성능 개선

- 성능 분석 도구
  - 리소스 및 성능 모니터링 : APM(Application Performance Management)
  - SQL 성능 측정 : TKPROF, EXPLAIN PLAN



- SQL 성능개선 순서
  - 문제 있는 SQL 식별
  - 옵티마이저 통계 확인
  - 실행 계획 검토
  - SQL문 재구성
  - 인덱스 재구성
  - 실행 계획 유지 관리



### 소스코드 인스펙션

- SQL 코드 인스펙션 대상
  - 사용되지 않는 변수
  - 사용되지 않는 서브쿼리
  - Null 값과 비교하는 프로시저 소스
  - 과거의 데이터타입 사용



- SQL 코드 인스펙션 절차
  - 계획
  - 개관
  - 준비
  - 검사
  - 재작업
  - 추적



# 2. 통합 구현



## 2-1. 모듈 구현



### 단위모듈 구현

- 단위모듈 구현시 고려사항
  - 응집도는 높이고, 결합도는 낮춤
  - 공통모듈을 먼저 구현
  - 항상 예외처리 로직을 고려하여 구현



### 단위모듈 테스트

- 단위모듈 테스트 방법
  - 화이트박스 테스트
  - 메소드 기반 테스트
  - 화면 기반 테스트
  - 스텁과 드라이버 활용 테스트



- 소스코드 커버리지 유형
  - 구문 커버리지
  - 결정 커버리지
  - 조건 커버리지
  - 조건/결정 커버리지
  - 변경조건/결정 커버리지
  - 다중조건 커버리지



- 단위테스트 자동화 도구를 활용한 디버깅
  - JUnit : Java 기반의 단위 모듈 테스트 자동화 도구
  - CppUnit : C++ 언어 기반의 단위 테스트 자동화 도구
  - unittest : Python에서 단위 테스트를 수행하기 위한 자동화 도구



## 2-2. 통합 구현 관리



### IDE 도구

- IDE 도구의 기능
  - 개발 환경 지원
  - 컴파일 및 디버깅 기능 제공
  - 외부 연계모듈과 통합 기능 제공



### 협업도구

- 협업도구 기능
  - 개발자간 커뮤니케이션
  - 일정 및 이슈 공유
  - 개발자간 집단지성 활용



### 형상관리 도구

- 형상관리절차
  - 형상 식별
  - 변경 제어
  - 형상 감사
  - 형상 기록



- 소프트웨어 형상관리 도구
  - CVS
  - SVN
  - Git



# 3. 제품소프트웨어 패키징



## 3-1. 제품소프트웨어 패키징



### 애플리케이션 패키징

- 애플리케이션 패키징 특징
  - 사용자 중심 진행
  - 모듈화
  - 신규/변경 이력 확인
  - 버전 관리 및 릴리즈 노트를 통해 지속적 관리
  - 범용 환경에서 사용 가능



- 애플리케이션 패키징 순서
  - 기능 식별
  - 모듈화
  - 빌드 진행
  - 사용자 환경 분석
  - 패키징 적용 시험
  - 패키징 변경 개선



- 릴리즈 노트 작성 순서
  - 모듈 식별
  - 릴리즈 정보 확인
  - 릴리즈 노트 개요 작성
  - 영향도 체크
  - 정식 릴리즈노트 작성
  - 추가 개선항목 식별



### 애플리케이션 배포 도구

- 패키징 도구 활용 시 고려 사항
  - 암호화/보안 고려
  - 이기종 연동 고려
  - 복잡성 및 비효율성 문제 고려
  - 적합한 암호화 알고리즘 적용



### 애플리케이션 모니터링 도구

- 애플리케이션 모니터링 도구 기능
  - 애플리케이션 변경관리
  - 애플리케이션 성능관리
  - 애플리케이션 정적분석
  - 애플리케이션 동적분석



## 3-2. 제품소프트웨어 매뉴얼 작성



### 제품소프트웨어 매뉴얼 작성

- 제품 소프트웨어 설치 매뉴얼 기본 사항
  - 제품소프트웨어 개요
  - 설치 관련 파일
  - 설치 절차
  - 설치 아이콘
  - 프로그램 삭제
  - 설치 환경
  - 설치 버전 및 작성자
  - FAQ



- 제품 소프트웨어 설치 매뉴얼 작성 순서
  - 기능 식별
  - UI분류
  - 설치 / 백업 파일 확인
  - Uninstall 절차 확인
  - 이상 Case 확인
  - 최종 매뉴얼 적용



- 제품 소프트웨어 사용자 매뉴얼 기본 사항
  - 제품 소프트웨어 개요
  - 제품 소프트웨어 사용
  - 제품 소프트웨어 관리
  - 모델, 버전별 특징
  - 기능, I/F의 특징
  - 제품 소프트웨어 구동환경



### 국제 표준 제품 품질 특성

- 소프트웨어 제품 품질 관련 국제 표준
  - ISO/IEC 9126
  - ISO/IEC 14598
  - ISO/IEC 12119
  - ISO/IEC 25000



- 소프트웨어 프로세스 품질 관련 국제 표준
  - ISO/IEC 9000
  - ISO/IEC 12207
  - ISO/IEC 15504
  - ISO/IEC 15288
  - CMMi



## 3-3. 제품소프트웨어 버전관리



### 소프트웨어 버전 관리 도구

- 소프트웨어 버전관리 도구 유형
  - 공유 폴더 방식 : RCS, SCCS
  - 클라이언트/서버 방식 : CVS, SVN
  - 분산 저장소 방식 : Git, Bitkeeper



### 빌드 자동화 도구

- 젠킨스 : 온라인 빌드 자동화 도구
- 그래들 : 안드로이드 환경에 적합한 도구



# 4. 애플리케이션 테스트 관리



## 4-1. 애플리케이션 테스트케이스 설계



### 애플리케이션 테스트 케이스 작성

- 소프트웨어 테스트의 필요성
  - 오류 발견 관점
  - 오류 예방 관점
  - 품질 향상 관점



- 소프트웨어 테스트 산출물
  - 테스트 계획서
  - 테스트 케이스
  - 테스트 시나리오
  - 테스트 결과서



- 정적 테스트
  - 기술적 검토기법
    - 개별 검토
    - 동료 검토
    - 검토 회의
  - 관리적 검토기법
    - 검사
    - 감사



- 동적 테스트
  - 명세 기반 테스트(블랙박스 테스트)
  - 구조 기반 테스트(화이트박스 테스트)



- V 모델과 테스트 단계

  | 요구사항 분석 |  인수 테스트  |
  | :-----------: | :-----------: |
  | 기능명세 분석 | 시스템 테스트 |
  |     설계      |  통합 테스트  |
  |     개발      |  단위 테스트  |



### 애플리케이션 테스트 시나리오 작성

- 테스트 환경 구축시 유의점
  - 테스트 환경의 분리
  - 가상 머신 기반의 서버나 클라우드 환경의 이용
  - 네트워크 분할과 공유디스크 관리
  - 연동 시스템의 테스트 환경



## 4-2. 애플리케이션 통합 테스트



### 애플리케이션 통합 테스트 수행

- 통합 테스트 수행 방법
  - 상향식 통합 테스트(Top Down)
  - 하향식 통합 테스트(Bottom Up)
  - 회귀 테스팅 : 반복 테스트



- 테스트 자동화 도구 유형
  - 정적 분석 도구
  - 테스트 실행 도구
  - 성능 테스트 도구
  - 테스트 통제 도구
  - 테스트 장치
  - 테스트 자동화 도구



### 애플리케이션 테스트 결과 분석

- 결함 추적 관리 활동
  - 단위 테스트
  - 통합 테스트
  - 시스템 테스트
  - 운영 테스트



## 4-3. 애플리케이션 성능 개선



### 알고리즘

- 정렬 알고리즘
  - 선택 정렬
  - 삽입 정렬
  - 버블 정렬



- 그래프 알고리즘
  - 깊이우선탐색(DFS) : 스택 자료구조 사용
  - 너비우선탐색(BFS) : 큐 자료구조 사용



- 탐색 알고리즘
  - 선형 탐색
  - 이진 탐색



### 애플리케이션 성능 개선

- 클린 코드의 작성 원칙
  - 가독성
  - 단순성
  - 의존성
  - 중복성
  - 추상화



- 소스코드 최적화 기법
  - 클래스 분할 배치 기법
  - 느슨한 결합(Loosely Coupled) 기법
  - 코딩 형식 기법
  - 주석문 사용을 습관화



- 소스코드 품질분석 도구
  - 정적 분석 도구
    - Pmd
    - cppcheck
    - SonarQube
    - checkstyle
  - 동적 분석 도구
    - Avalanche
    - valgrind



# 5. 인터페이스 구현



## 5-1. 인터페이스 설계 확인



### 인터페이스 기능 확인

- 인터페이스 기능 확인 방법
  - 인터페이스 설계서
  - 정적, 동적 모형
  - 데이터 명세 정의
  - 내부, 외부 모듈 연계 방법
    - EAI(Enterprise Application Integration)
    - ESB(Enterprise Service Bus)



### 데이터 표준 확인

인터페이스 구현 시 데이터 표준을 준수하여 구현



## 5-2. 인터페이스 기능 구현



### 인터페이스 보안

- 스니핑 : 패킷을 중간에 채감
- 스푸핑 : IP 주소나 e-메일 주소를 바꾸어 해킹



- 인터페이스 보안 기능 적용
  - 네트워크 구간 보안 기능 적용
  - 애플리케이션에 보안 기능 적용
  - 데이터베이스에 보안 기능 적용



- 데이터베이스 암호화 알고리즘
  - 대칭키
  - 해시
  - 비대칭키



### 소프트웨어 연계 테스트

송신 시스템에서 중계 시스템을 거쳐서 수신 시스템까지 연계하였을 경우 데이터의 정합성 및 데이터 전송 여부를 테스트



## 5-3. 인터페이스 구현 검증



### 인터페이스 구현 검증

- 인터페이스 구현 검증 도구
  - xUnit
  - STAF
  - FitNesse
  - NTAF : Naver에서 개발
  - Selenium
  - watir



- 인터페이스 구현 검증에 필요한 설계 산출물
  - 인터페이스 명세서(정의서)
  - 인터페이스 단위 및 통합 테스트 설계서



### 인터페이스 오류 처리 확인 및 보고서 작성

- 인터페이스 오류 처리 방법
  - 사용자 화면에서 오류를 발생
  - 인터페이스 오류 로그 생성
  - 인터페이스 관련 테이블에 오류 사항 기록