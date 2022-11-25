# 포팅메뉴얼 (임시)

## 3D

- Bowl 모델을 통한 3D SVM 제작

### 환경 및 라이브러리

- OS : Windows 10 Pro
- Language : C++
- Visual Studio 2017
- openCV : 3.4.15
- openGL
    - glew : 2.1.0 win32
    - glfw : 3.3.8 WIN64
    - glm : 0.9.8.5
    - freeglut-MSVC : 3.0.0-1.mp for MSVC
    - glut : 3.7.6 Win32
- Magick++ : ImageMagick-7.1.0-52-Q16-HDRI

### 설치방법

1. Visual Studio 다운로드

[공식홈페이지](https://learn.microsoft.com/ko-kr/visualstudio/releasenotes/vs2017-relnotes-history)에서 Visual Studio installer 2017버전을 다운로드를 받는다.

<img src="https://user-images.githubusercontent.com/19484971/202615663-33288eda-3ddc-4e32-a820-f9b01cde2923.png" width=500>

현재 2022.11.18일 기준 최신의 2022버전은 해당 프로젝트에서 필요한 SDK나 구성요소를 설치할 수가 없어서 따로 설치를 해주어야 하고 2022버전을 사용한다고 하더라도 플렛폼 도구집합을 `v141`으로 사용해주어야 하기 때문에 2017버전으로 다운로드 해주어야 한다.

2. C++에 대한 필요한 라이브러리를 다운로드

어떤 라이브러리가 필수인지 확실하지는 않지만, 적어도 `C++를 사용한 데스크톱 개발`의 `Windows 8.1 SDK 및 UCRT SDK`, `x86 및 x64용 Visual C++ MFC`는 필수적이다. 

`Windows 10 SDK(10.0.17134.0)`의 경우에는 다른 `Windows 10 SDK`도 작동되는 것 같으므로 하나만 설치하면 될 것이다.

단, 다른 윈도우 버전에서는 확인을 못했으므로 다른 버전이라면 진행이 원활하지 않을 수도 있다. 

<img src="https://user-images.githubusercontent.com/19484971/201578453-b53ba018-f2f5-4bc8-915c-286530e0cb3d.png" width=500>

<img src="https://user-images.githubusercontent.com/19484971/201579769-8f1207eb-501f-4d69-b9a3-a44175ef8b46.png" width=700>

<img src="https://user-images.githubusercontent.com/19484971/201579981-0a828e06-84bb-423c-ac90-7892acaed66c.png" width=400>

2. 솔루션 속성 확인

다른 버전의 Visual Studio로 진행해도 되는 것 같으나 플렛폼 도구집합은 꼭 `v141`로 해주어야 한다. 

또한 `Debug`나 `Release`로 해주는 것은 상관이 없으나, 플렛폼은 꼭 `x64`로 설정해주어야 한다.

<img src="https://user-images.githubusercontent.com/19484971/201580882-5b17af1c-1440-4045-8287-e2ae2fad060b.png" width=500>
<img src="https://user-images.githubusercontent.com/19484971/201581201-55f24c67-1a13-4008-a937-d2bbcae1d0ab.png" width=300>

3. 실행

실행해서 아래와 같은 이미지를 본다면 성공한 것이다!

-- 실행시 이미지 추가 요망--

### 에러

1. `image.cpp`

`image.cpp` 파일이 없다고 에러가 생길 수 있다. 이런 경우 `SurroundViewMonitor\include\Magick++`에 있는 `image.cpp`이 있는 폴더를 참조경로로 넣어주면 된다. 

해당 파일은 [ImageMagick-Windows 깃헙](https://github.com/ImageMagick/ImageMagick-Windows)을 클론하고 파일 내부에 있는 `CloneRepositories.cmd`을 더블클릭하면 자동으로 설치되는 라이브러리 중 `ImageMagick-Windows-main\ImageMagick\Magick++\lib`에서 복사한 것이다.

