---
published: true
layout: single
title: "Cargle 프로젝트 후기"
category: Project
comments: true
toc: true
toc_sticky: true
use_math: true
---

두 번째로 진행했던 프로젝트는 Cargle이다. (Car와 google의 합성어)

멀티캠퍼스 '빅데이터 활용 AI 설계' 이론 교육이 다 끝나고 종합 실습과정에서 진행한 프로젝트이다.



## 기획 배경 및 목표

자동차를 처음 구매하는 사람들은 차량의 종류가 많다보니 어떤 차를 사야할지 모르는 경우가 많다.  이를 해소하고자 자동차 리뷰를 자연어 처리로 분석하여 긍정 리뷰와 부정 리뷰 중 어느 것이 더 많은지 분석하여 자동차 추천 서비스를 구축한다.



## 프로젝트 수행 과정

1. 데이터 수집 : Selenium, BeautifulSoup 라이브러리를 이용하여 자동차 관련 16개 사이트에서 자동차 시승후기 및 리뷰를 수집 (수집 시 정규표현식을 활용하여 1차 전처리)

   

2. 전처리 : 정규표현식을 활용하여 시승후기 본문에 포함된 기자 이름, 메일 주소, 이미지 캡션, html 코드 삭제 

   

3. 라벨링

   - 브랜드와 모델명을 네이버에 등록된 명칭으로 라벨링

   - 각 리뷰를 수집해온 사이트의 출처 라벨링

   - 네이버에 차종 별 등록된 금액(최소금액과 최대금액) 크롤링하여 라벨링

     

4. 감성사전 구축

   - 군산대학교의KNU 감성사전과 서울대학교의 KOSAC 감성사전을 통합

   - 사전에 포함되지 않았으나 본문에 포함된 자동차 관련 단어를 감성사전에 추가

     

5. 자연어 처리

   - Konlpy 라이브러리의 Komoran 형태소 분석기를 활용하여 리뷰 본문에 형태소 태그를 부착

   - 1-gram, 2-gram, 3-gram 형태로 변환

   - 한글, 알파벳, 문장 부호를 제외한 특수문자 제거

     

6. 감성 점수 계산

   - 자연어 처리한 데이터를 구축한 감성사전에 대입하여 수치화

   - 글의 길이에 따른 영향력을 최소화하기 위해 감성 점수를 전체 단어 개수로 나누어 계산

     

7. 검색 기능 구현

   - 원하는 금액대를 입력하면 해당 금액대에 속하는 브랜드, 모델명, 최소 가격, 최대 가격, 감성 점수를 출력
   - 원하는 브랜드를 입력하면 해당 브랜드의 모델명, 최소 가격, 최대 가격, 감성 점수를 출력  



## 후기

지금까지는 주어진 데이터를 분석하였다면, 종합 프로젝트는 데이터를 직접 구해서 분석하고 서비스를 구현하는 작업이었다. 데이터 수집부터 전처리, 데이터 분석, 기능 구현까지 프로젝트의 전과정을 참여하며 실력이 많이 늘었다. 또한 보고서 및 PPT를 제작하며 문서 작성 능력도 길렀다.

프로젝트를 하며 아쉬웠던 점은 한국어 감성사전의 완성도이다. KNU와 KOSAC 감성사전을 통합하여 사용했지만 실제 감성분석 결과를 보면 분석이 되지 않은 글자가 많다. 때문에 자동차 관련 단어를 추가하느라 고생했다. 나중에 완벽한 한국어 감성사전이 구축되면 좋을 것 같다.