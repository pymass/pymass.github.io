---
published: true
layout: single
title: "정보처리기사 필기 정리 - 4과목"
category: Certificate
toc: true
use_math: true
---

코로나 때문에 시험이 4월 25일로 미루어져 이제서야 포스팅을 올린다.



정보처리기사 4과목은 **<u>프로그래밍 언어 활용</u>**이다.



# 1. 서버프로그램 구현



## 1-1. 개발환경 구축

### 개발환경 구축

- 응용 소프트웨어 개발을 위해 개발 프로젝트를 이해하고 소프트웨어 및 하드웨어 장비를 구축하는 것
- 개발 하드웨어 환경
  - 클라이언트
  - 서버
    - 웹 서버
    - 웹 애플리케이션 서버
    - 데이터베이스 서버
    - 파일 서버
- 개발 소프트웨어 환경
  - 시스템 소프트웨어
    - 운영체제(OS, Operation System)
    - JVM(Java Virtual Machine)
    - Web Server
    - WAS(Web Application Server)
    - DBMS(Database Management System)
  - 개발 소프트웨어
    - 요구사항 관리 도구
    - 설계/모델링 도구
    - 구현 도구
    - 빌드 도구
    - 테스트 도구
    - 형상 관리 도구
- 개발환경 구축 순서
  - 목표 시스템의 환경 및 요구사항 분석
  - 개발언어 선정
  - 통합 개발환경 선정
  - 프로그램의 배포 및 라이브러리 관리를 위한 빌드 도구 선정
  - 개발 인원을 고려한 형상관리 도구를 선정
  - 프로젝트 검증에 적합한 테스트 도구를 선정



### 서버 개발 프레임워크

- 서버 프로그램 개발 시 다양한 네트워크 설정, 요청 및 응답처리, 아키텍처 모델 구현 등을 손쉡게 처리할 수 있도록 클래스나 인터페이스를 제공하는 소프트웨어
- 서버 개발 프레임워크의 종류
  - Spring(JAVA)
  - Node.js(JavaScript)
  - Django(Python)
  - Codeigiter(PHP)
  - Ruby on Rails(Ruby)



## 1-2. 공통 모듈 구현

### 재사용

- 목표 시스템의 개발 시간 및 비용 절감을 위하여 검증된 기능을 파악하고 재구성하여 시스템에 응용하기 위한 최적화 작업

- 재사용 범위에 따른 분류

  - 함수와 객체 재사용 : Function, Class 단위
  - 컴포넌트 재사용 : Component 단위
  - 애플리케이션 재사용

  

### 모듈화

- 소프트웨어 개발 작업을 실제로 개발할 수 있는 작은 단위로 나누는 것

- 모듈화 측정 척도

  - 응집도
  - 결합도

- 모듈 간의 좋은 관계 : 응집도는 높게, 결합도는 낮게

  

### 응집도

- 인터페이스의 요청을 처리함에 있어서 공통 모듈 내의 클래스들 간에 얼마나 유기적으로 협업하여 처리하는 가에 관한 정도

- 응집도 유형

  - 기능적(Functional Cohension)

  - 순차적(Sequential Cohension)

  - 통신적(Communication Cohension)

  - 절차적(Procedual Cohension)

  - 시간적(Temporal Cohension)

  - 논리적(Logical Cohension)

  - 우연적(Coincidental Cohension)

    

### 결합도

- 어떤 모듈이 다른 모듈에 의존하는 정도
- 결합도 유형
  - 내용(Content Coupling)
  - 공통(Common Coupling)
  - 외부(External Coupling)
  - 제어(Control Coupling)
  - 스탬프(Stamp Coupling)
  - 자료(Data Coupling)



## 1-3. 서버 프로그램 구현

### 보안 취약성 식별

- 소프트웨어 개발 보안은 소프트웨어 개발 과정에서 발생할 수 있는 보안 취약점을 최소화하고, 사이버 보안 위협에 대응할 수 있는 안전한 소프트웨어를 개발하기 위한 일련의 보안 활동
- 소프트웨어 개발 보안 점검 항목
  - 입력 데이터 검증 및 표현
  - 보안 기능
  - 시간 및 상태
  - 에러 처리
  - 코드 오류
  - 캡슐화
  - API 오용



### API

- API는 응용 프로그램 개발 시 운영체제나 프로그래밍 언어 등에 있는 라이브러리를 이용할 수 있도록 규칙 등을 정해 놓은 인터페이스
- API의 종류
  - Windows API
  - 단일 유닉스 규격(SUS)
  - Java API
  - 웹 API



## 1-4. 배치 프로그램 구현



### 배치 프로그램

- 배치 프로그램은 사용자와의 상호 작용 없이 일련의 작업들을 작업 단위로 묶어 정기적으로 반복 수행하거나 정해진 규칙에 따라 일괄 처리하는 것
- 배치 프로그램의 필수 요소
  - 대용량 데이터
  - 자동화
  - 견고함
  - 안정성
  - 성능
- 배치 스케줄러는 일괄 처리(Batch Processing)를 위해 주기적으로 발생하거나 반복적으로 발생하는 작업을 지원하는 도구
- 배치 스케줄러의 종류
  - Spring Batch
  - Quartz



# 2. 프로그래밍 언어 활용



## 2-1. 기본문법 활용

### 데이터 타입

- 변수에 저장될 데이터의 형식을 나타내는 것
- 데이터 타입의 유형
  - 불린 타입
  - 문자 타입
  - 문자열 타입
  - 정수 타입
  - 부동 소수점 타입
  - 배열 타입



### 변수

- 변수는 저장하고자 하는 어떠한 값이 있을 때, 그 값을 주기억 장치에 기억하기 위한 공간을 의미
- 변수는 저장하는 값에 따라 정수형, 실수형, 문자형 등으로 구분
- C언어 변수명 설정 규칙
  - 영문자, 숫자, 밑줄(_)의 사용이 가능
  - 첫 글자는 영문자나 '_'로 시작, 숫자는 사용불가
  - 변수명에는 공백 사용 불가
  - 영문자는 대소문자를 구분함
  - 변수명은 제어문, 자료형 등 예약어(do, for, while, char, double 등) 사용 불가



### 연산자

- 연산자는 프로그램 실행을 위해 연산을 표현하는 기호
- 연산자의 종류는 산술 연산자, 관계 연산자, 비트 연산자, 시프트 연산자, 논리 연산자 등이 있음
- 연산 표기법의 종류
  - 전위(Prefix) 표기법
  - 중위(Infix) 표기법
  - 후위(Postfix) 표기법



## 2-2. 언어특성 활용

### 절차적 프로그래밍 언어

- 일련의 처리 절차를 수행하는 프로시저를 구현하며, 정해진 문법에 따라 순서대로 기술하는 언어
- 장점 : 실행 속도가 빠르며, 모듈 구성이 용이한 구조적 프로그래밍
- 단점 : 프로그램 분석이 어렵고, 유지 보수나 코드 수정이 어려움
- 종류 : C, ALGOL, COBOL, FORTRAN



### 객체지향 프로그래밍 언어

- 프로시저보다는 명령과 데이터로 구성된 객체를 중심으로 하는 프로그래밍 기법
- 장점 : 재사용성, 소프트웨어 개발 및 유지보수가 용이
- 단점 : 구현 시 처리 시간의 지연
- 특징 : 캡슐화, 정보은닉, 추상화, 상속성, 다형성
- 종류 : C#, JAVA, C++, Smalltalk



### 스크립트 언어

- HTML 문서 안에 직접 프로그래밍 언어를 삽입하여 사용되며, 기계어로 컴파일 되지 않고 별도의 번역기가 소스를 분석하여 동작하는 언어
- 장점 : 컴파일 없이 바로 실행하며 소스 코드를 빠르게 수정 가능
- 단점 : 코드를 읽고 해석해야 하므로 실행속도가 느림
- 클라이언트용 스크립트 언어 : JavaScript
- 서버용 스크립트 언어 : ASP, JSP, PHP, Python



### 선언형 언어

- 명령형 언어가 문제를 해결하기 위한 방법을 기술한다면 선언형 언어는 프로그램이 수행해야 할 문제를 기술하는 언어
- 종류 : LISP, PROLOG, Haskell, SQL, HTMl, XML



## 2-3. 라이브러리 활용

### 라이브러리

- 프로그램을 효율적으로 개발할 수 있도록 자주 사용하는 함수나 데이터들을 미리 만들어 모아 놓은 집합체
- 일반적으로 도움말, 설치 파일, 샘플 코드 등을 제공
- 라이브러리 종류 : 표준 라이브러리, 외부 라이브러리
- C언어의 대표적인 표준 라이브러리
  - stdio.h : 데이터 입출력
  - string.h : 문자열 처리
  - math.h : 수학 함수
  - stdlib.h : 자료형 변환, 난수 발생, 메모리 할당
  - time.h : 시간 처리
- JAVA언어의 대표적인 표준 라이브러리
  - java.lang : 기본 인터페이스, 자료형
  - java.util : 날짜 처리, 난수 발생, 복잡한 문자열 처리
  - java.io : 파일 입출력
  - java.net : 네트워크 관련
  - java.awt : 사용자 인터페이스



### 데이터 입출력

- 키보드로 입력 받아 화면으로 출력할 때 사용하는 함수 또는 클래스와 메소드
- C언어의 데이터 표준 입출력 함수
  - scanf()
  - getchar()
  - gets()
  - printf()
  - putchar()
  - puts()
- JAVA 언어의 데이터 표준 입출력
  - 입력 관련 클래스 : Scanner
  - 서식 지정 출력 : System.out.printf()



### 예외 처리

- 프로그램의 정상적인 실행을 방해하는 조건이나 상태를 뜻하는 예외가 발생했을 때 해당 문제에 대한 처리 루틴을 수행하도록 하는 것
- JAVA에서의 예외 처리 : try~catch 구문을 이용한 예외 처리



### 프로토타입

- 프로그래밍 언어에서 프로토타입이랑 함수 원형(Prototype)이라는 의미로 컴파일러에게 사용될 함수에 대한 정보를 미리 알리는 것
- C언어에서는 함수가 호출되기 전에 함수가 미리 정의된 경우에는 프로토타입을 정의하지 않아도 됨
- 프로토타입에 정의된 반환 형식은 함수 정의에 지정된 반환 형식과 반드시 일치하여야 함



# 3. 응용 SW 기초 기술 활용



## 3-1. 운영체제 기초 활용

### 운영체제 종류

- 운영체제는 컴퓨터 시스템의 자원들을 효율적으로 관리하며, 사용자가 컴퓨터를 편리하고 효과적으로 사용할 수 있도록 인터페이스를 제공하는 시스템 소프트웨어
- 운영체제의 종류
  - Windows
  - UNIX
  - LINUX
  - MacOS
  - MS-DOS



### 메모리 관리

- 기억장치의 계층 구조
  - 레지스터
  - 캐시 기억장치
  - 주기억장치
  - 보조기억장치
- 기억장치 관리 전략
  - Fetch(반입) : 언제 적재?
  - Placement(배치) : 어디에 위치?
  - Replacement(교체) : 어느 영역 교체?
- 주기억장치 할당 기법
  - 연속 할당
  - 분산 할당
- 가상기억장치의 구현 기법
  - 페이징 : 동일한 크기로 나눔
  - 세그먼테이션 : 다양한 크기의 논리적 단위로 나눔
- 페이지 교체 알고리즘 종류
  - FIFO : 선입선출
  - LRU : 최근 오래 사용 않은 페이지 교체
  - NUR : 최근 사용 않은 페이지 교체
  - OPT : 최적 교체
  - LFU : 사용 빈도가 가장 작은 페이지
  - SCR : 2차 기회 교체
- 가상기억장치의 기타 관리
  - Locality
  - Working Set
  - Thrashing



### 프로세스 스케줄링

- 프로세스 : 프로세서(처리기, CPU)에 의해 처리되는 사용자 프로그램
- 스레드 : 프로세스 내에서의 작업 단위로서 시스템의 여러 자원을 할당받아 실행하는 프로그램의 단위
- 스케줄링 : 프로세스가 생성되어 실행될 때 필요한 시스템의 여러 자원을 해당 프로세스에게 할당하는 작업
- 비선점 스케줄링의 종류
  - FIFO(FCFS)
  - SJF
  - HRN
- 선점 스케줄링의 종류
  - RR
  - SRT
  - MLQ
  - MFQ



### 환경변수

- 시스템 소프트웨어의 동작에 영향을 미치는 동적인 값들의 모임
- 윈도우 환경변수
  - %APPDATA%
  - %COMSPEC%
  - %PATH%
  - %USERNAME%
  - %PROGRAMFILES%
  - %SYSTEMDRIVE%
- 유닉스 환경변수
  - $DISPLAY
  - $HOME
  - $LANG
  - $PATH
  - $PWD
  - $TERM
  - $USER



### shell script

- CLI(Command Line Interface)는 키보드로 명령어를 직접 입력하여 작업을 수행하는 사용자 인터페이스
- UNIX의 기본 명령어
  - cat : 파일 내용을 화면에 표시
  - chdir : 디렉토리의 위치 변경
  - chmod : 파일의 사용 허가(권한) 지정
  - chown : 소유자 변경
  - cp : 파일 복사
  - getpid : 자신의 프로세스 아이디를 얻음
  - ls : 현재 디렉토리 내의 파일 목록을 표시
  - rm : 파일 삭제



## 3-2. 네트워크 기초 활용

### 인터넷 구성의 개념

- 인터넷은 TCP/IP 프로토콜을 기반으로 전 세계 수많은 컴퓨터와 네트워크들이 연결된 광범위한 컴퓨터 통신망
- 네트워크 장비의 종류
  - 게이트웨이(Gateway)
  - 라우터(Router)
  - 리피터(Repeater)
  - 허브(Hub)
  - 랜 카드(NIC)
  - 브리지(Bridge)
  - 스위치(Switch)



### 네트워크 7계층

- OSI 7계층은 국제 표준화 기구인 ISO에서 다른 시스템간 통신을 위해 네트워크 구조를 제시한 기본 모델
  - 7계층 : 응용 계층
  - 6계층 : 표현 계층
  - 5계층 : 세션 계층
  - 4계층 : 전송 계층
  - 3계층 : 네트워크 계층
  - 2계층 : 데이터링크 계층
  - 1계층 : 물리 계층



### 네트워크 프로토콜

- 프로토콜은 서로 다른 기기들 간의 데이터 교환을 원할하게 수행할 수 있도록 표준화 시켜 놓은 통신 규약
- TCP/IP 4계층
  - 4계층 : 응용 계층
  - 3계층 : 전송 계층
  - 2계층 : 인터넷 계층
  - 1계층 : 네트워크 엑세스 계층
- 프로토콜의 기본 요소 : 구문, 의미, 시간
- 라우팅 프로토콜
  - RIP : 거리 벡터 알고리즘
  - IGRP
  - OSPF : 링크 상태 알고리즘
  - BGP



### IP

- IP는 OSI 7계층의 네트워크 계층에서 호스트의 주소지정과 패킷 분할 및 조립 기능을 담당



### TCP/UDP

- TCP는 OSI 7계층의 전송 계층에서 논리적인 1:1 가상 회선을 지원하고 CRC 체크와 재전송 기능을 통해 신뢰성 있는 연결형 서비스를 제공
- TCP 사용 서비스
  - FTP
  - Telnet
  - Http
  - SMTP
  - POP
  - IMAP
- UDP는 비연결성이고 신뢰성이 없는 데이터 전송
- UDP 사용 서비스
  - SNMP
  - DNS
  - TFTP
  - NFS
  - NETBIOS
  - 인터넷 게임



## 3-3. 기본 개발환경 구축

### 웹서버

- 웹서버는 웹 브라우저 클라이언트로부터 HTTP request를 받아 HTML과 같은 정적인 contents를 제공하는 프로그램과 해당 애플리케이션 서버가 설치된 컴퓨터
- 웹 서버 종류
  - Apache
  - Nginx
  - IIS
  - GWS
- WAS서버는 DB 조회나 다양한 로직 처리를 요구하는 동적인 contents를 제공하기 위한 Apllication Server
- WAS 서버 종류
  - Tomcat
  - Undertow
  - JEUS
  - Weblogic
  - Websphere



### DB서버

- 사용자, 다른 애플리케이션, 데이터베이스와 상호 작용하여 데이터를 저장하고 분석하기 위한 컴퓨터 소프트웨어
- DB서버 종류
  - Oracle
  - DB2
  - Microsoft SQL Server
  - MySQL
  - MongoDB
- DB서버 고려사항
  - 가용성
  - 성능
  - 기술지원
  - 상호호환성
  - 구축비용



### 패키지

- 패키지 방식 개발은 여러 성공사례의 노하우를 기반으로 만들어진 개발된 제품을 이용하여 시스템을 구축하는 방식
- 패키지 방식 개발의 장점
  - 국제 및 산업계 표준으로 정착된 비즈니스 프로세스 적용
  - 품질이 검증된 안정적인 시스템 구축 가능
  - 개발 기간의 단축으로 비용절감 효과
- 패키지 방식 개발의 단점
  - 사용자 요구사항에 대한 대처가 쉽지 않음
  - 사용자(고객)의 프로세스 개선의 저항발생