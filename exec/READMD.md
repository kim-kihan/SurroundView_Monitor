# 포팅메뉴얼

## 2D Surround View Monitor

- 카메라 내부 파라미터 추출
- 카메라 왜곡 보정
- 카메라 homography 적용 및 이미지 합성
- TopView를 통한 2D SVM 제작

### 개발환경

- OS : Windows 10 Pro
- Language : Python(3.8.13)
- Code Editor : Visual Studio Code (1.70.0)
- Library
    - openCV (4.6.0)

python 3.5.3에서도 잘 작동되었다.

### 설치 방법

1. Visual Studio Code, Python 설치

[Visual Studio Code](https://code.visualstudio.com/)는 공식 홈페이지에서 추천하는 버전으로 다운로드 받으면 충분할 것이다.

Python은 [anaconda](https://www.anaconda.com/)를 활용해서 설치해도 되고 [python 공식 홈페이지](https://www.python.org/downloads/)에서 적당한 버전을 설치하면 된다. 

해당 프로젝트는 `Python 3.8.13`을 기준으로 만들었기 때문에 해당 버전을 강력히 추천한다.

2. Visual Studio Code Extension 설치

Visual Studio Code에서 `Extension` 에서 `Python` 설치

![image](https://user-images.githubusercontent.com/19484971/202385665-70638691-85eb-49a1-a2e0-6ff157c90a23.png)

3. openCV 설치

커맨드 창에서 `pip install opencv-python`을 입력하여 openCV를 설치한다.

만약 정상적으로 설치되지 않았다면 `python -m pip install --upgrade pip`을 입력하여 pip를 업데이트한 후 다시 oepncv 설치 명령어를 입력한다.

이후 아래의 코드가 잘 작동되면 설치가 잘 된 것이다.

```
# OpenCV 패키지 임포트
import cv2

# 패키지 설치 위치 확인
print(cv2.__file__)

# 패키지 버전 확인
print(cv2.__version__)
```

4. 실행

전후좌우 카메라에 맞는 숫자를 [메인 파일](../SurroundViewMonitor%202D/main.py)에 입력해준다. 아래의 코드에서 숫자를 바꾸어주면 된다. 

```
cap1 = cv2.VideoCapture(2) # 전방 카메라
cap2 = cv2.VideoCapture(5) # 좌측 카메라
cap3 = cv2.VideoCapture(4) # 우측 카메라
cap4 = cv2.VideoCapture(1) # 후방 카메라
```

보통 인식이 안되는 것은 USB 포트를 뺏다 끼거나 `cv2.VideoCapture(0)`의 숫자를 바꾸어보면서 확인하거나 직접 카메라 테스트 파일을 넣어 확인하면서 진행하자.

만약 예시 이미지로만 확인하는 것이라면 `WORK_PROCESS` 함수 내의 아래의 `cv2.imread`들의 주석을 지우고 `cap2.read()`줄을 주석 처리해주면 된다.

`cv2.imread`에는 원하는 이미지의 경로를 입력해준다. 현재는 프로젝트에서 기본으로 제공하는 예시 이미지로 지정되어있다.

```
# frame = cv2.imread('front.png')
ret1, frame = cap1.read()

# frame = cv2.imread('left.png')
ret2, frame = cap2.read()

# frame = cv2.imread('right.png')
ret2, frame = cap3.read()

# frame = cv2.imread('back.png')
ret3, frame = cap4.read()

실행은 매우 오래걸리기 때문에 길면 15분까지도 기다려주어야 한다. 초기 계산만이 오래 걸리는 것이기 때문에 프로그램 속도와는 연관이 없으니 에러가 생기지 않는 이상 중단시키지 말자. 

카메라의 순서가 맞지 않다면 전,후,좌,우가 다르게 출력될 수 있다. 4개의 사진을 이상하게 이어붙인 듯한 이미지가 보인다면 카메라 세팅을 다시 진행하자.

### 결과물

<img src="https://user-images.githubusercontent.com/19484971/202616735-6f23192e-6178-40eb-91fc-a91374316e08.png" width=300>

#### 개발 과정의 이미지들

<img src="https://user-images.githubusercontent.com/19484971/202380608-13c4bd98-325a-44a8-ae4c-6e7c90f9c4c4.png" width=300>

<img src="https://user-images.githubusercontent.com/19484971/202380765-056145a0-20c7-43f3-a2b9-ad1e777a9359.png" width=500>

<img src="https://user-images.githubusercontent.com/19484971/202380881-145c6286-b8bb-433e-b30e-2144a3e03e94.png" width=500>