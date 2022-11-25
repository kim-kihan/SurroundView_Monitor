# 주차 보조 시스템 SVM 개발

## 개발환경

- OS : Windows 10 Pro
- Language : Python(3.8.13)
- Code Editor : Visual Studio Code (1.70.0)
- Library
    - openCV (4.6.0)

## 팀원

김기한, 김이랑, 김필재, 박주현, 임영선, 임진현(팀장)

## 프로젝트 기간

2022.10.11 ~ 2022.11.21 (7주)

## 개요

많은 운전자들이 자동차 주차 시 사각지대로 인해 숙달되기 전까지는 어려움을 겪는다. 이는 자동차의 크기가 클수록 정도가 심하다.

위의 문제점을 해결하기 위해서 SVM(Surround View Monitor) 시스템을 개발하여 차량의 전후좌우 4방향 카메라 영상을 합성하여 마치 위에서 내려다 보는 듯한 Top View 영상을 제공하여 주차 및 주행 시 사고를 크게 줄일 수 있도록 한다.

사진으로 보면 아래와 같다.

<img src="https://user-images.githubusercontent.com/19484971/202943753-d0798bd1-287d-4f07-8091-a57fc0f0edc7.png" width=300>
<img src="https://user-images.githubusercontent.com/19484971/202943835-5b512df4-aefb-4451-acc5-c977cd6de323.png" width=310>

## 이용 대상자

- 차량 이용자

## 사용 장비 및 설치

- [마트카트](https://yestore.kr/shop/item.php?it_id=1426567216)

<img src="https://user-images.githubusercontent.com/19484971/202383691-cbfe1e29-a2dc-4116-a4ad-d3014939b53b.png" width=400>

- [카메라](https://www.coupang.com/vp/products/6595685374?itemId=14901263581&vendorItemId=82139995937&q=%EC%96%B4%EC%95%88%EC%B9%B4%EB%A9%94%EB%9D%BC+usb&itemsCount=36&searchId=2a2f3dea45a94ec59734a168e85c6537&rank=2&isAddedCart=)

<img src="https://user-images.githubusercontent.com/19484971/203259896-15f01710-4c54-47ba-ab3b-9c45517a73e5.png" width=200>

- 카메라 설치위치와 각도

<img src="https://user-images.githubusercontent.com/19484971/203239711-6229d112-633e-446c-adbe-fd906a1ca9f8.png" width=400>

<img src="https://user-images.githubusercontent.com/19484971/203261100-d28591d0-7703-438d-8c89-f2c647dc81f1.png" width=400>

<img src="https://user-images.githubusercontent.com/19484971/203241175-f45d5bea-fa6b-4aa1-bba8-983ff5b42270.png" width=300>

- 카메라 4대 연결

<img src="https://user-images.githubusercontent.com/19484971/203247200-2268395d-ba5b-4eb9-8d6f-686f36b07608.jpg" width=400>

## 주요 기능

- 화각 180°이상 카메라 이미지 보정 기능 (왜곡보정 및 이미지정렬)
- Top View 영상 이미지 (2D) 제공
- 합성 경계 영역 처리
- 합성 이미지 밝기/색상 보정 기능

## 주요 기술

- 왜곡보정을 위한 카메라 캘리브레이션(Camera Calibration)


- TopView 변환을 위한 이미지 투영(Imaging Geometry)
- 이미지 투영과 합성을 위한 동차좌표 (Homogeneous coordinates)

<img src="https://user-images.githubusercontent.com/19484971/202954698-00de85b2-040d-45dc-846d-6a6ccb92a298.png" width=600>

> 호모그래피(homography) : 한 평면을 다른 평면에 투영(projection)시켰을 때 투영된 대응점들 사이의 일정한 변환관계

![image](https://user-images.githubusercontent.com/19484971/202955513-674f6fd6-0a0c-4a98-85af-e3f37ec3b0b8.png)
![image](https://user-images.githubusercontent.com/19484971/202955529-fcfe90e4-ef50-498f-9df1-65ce66827a90.png)


- 이미지 합성 경계면 블랜딩 처리 (색상, 명도 보정)

<img src="https://user-images.githubusercontent.com/19484971/202955210-bc560959-7a5b-4fc2-bfb6-efe00a989ec6.png" width=600>

> RGB 행렬 평균값을 활용하여 색상 보정

<img src="https://user-images.githubusercontent.com/19484971/202955228-9185c4dd-9cfe-4ef9-a3d6-46cc6acf6f90.png" width=600>

> HSV 행렬 산술평균 값을 적용하여 명도 조절

- 영상 속도 향상을 위한 멀티 프로세싱

<img src="https://user-images.githubusercontent.com/19484971/202961245-487441ab-5789-491f-8458-bb34e278639a.png" width=600>

<img src="https://user-images.githubusercontent.com/19484971/202960833-23b7c56b-b57d-4c3e-8dd0-81340a9f75e4.png" width=600>

> 5~7 프레임에서 19~20 프레임으로 성능 향상

## 추후 개발

- 자동차를 중심으로 다양한 시점에서 주변을 확인할 수 있는 [Bowl View 영상](./gif/)(3D SVM) 제공

## 결과물

<img src="https://user-images.githubusercontent.com/19484971/202616735-6f23192e-6178-40eb-91fc-a91374316e08.png" width=300>

<img src="./gif/output.gif" width=600>

## 기존 제품과 비교

<img src="https://user-images.githubusercontent.com/19484971/202961156-5bc5ac89-b355-453f-a87a-67fb10fe6bbe.png" width=600>
