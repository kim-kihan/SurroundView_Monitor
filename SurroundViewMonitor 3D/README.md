## 3D Surround View Monitor 

- OS : Windows 10 Pro
- Visual Studio 2017
- openCV : 3.4.15
- openGL
    - glew : 2.1.0 win32
    - glfw : 3.3.8 WIN64
    - glm : 0.9.8.5
    - freeglut-MSVC : 3.0.0-1.mp for MSVC
    - glut : 3.7.6 Win32
- Magick++ : ImageMagick-7.1.0-52-Q16-HDRI

### 작동방식

1. SVM의 2D 파트에서 받은 어안렌즈의 전후좌우 왜곡보정 이미지를 가져온다.
2. 위의 이미지들을 Bowl 모델에 맞게 왜곡을 적용시킨다.
3. 왜곡을 다시 적용한 이미지들를 합성하여 하나의 이미지를 만든다.

<img src="https://user-images.githubusercontent.com/19484971/203186728-0587f2eb-3851-470f-be16-6784d3c54b49.png" width=300>

4. 이미지를 적절하게 합성한다.

<img src="https://user-images.githubusercontent.com/19484971/203186736-f68443c8-9a7d-40fe-ba98-42d52300b4bc.png" width=300>

5. 합성하여 만든 이미지를 Texture로 만들어 Bowl 모델에 입힌다.

<img src="https://user-images.githubusercontent.com/19484971/203187271-00b2fce6-4036-44ba-93ca-256c28eacae6.png" width=300>

### Bowl 모델

[resource](./resource/) 폴더에서 `Blender`로 만든 `bowl.obj`를 확인할 수 있다. 

바닥은 거의 평평하고 특정 부분부터 각도가 크게 꺽여 그릇과 같은 모양을 가지고 있어 Bowl 모델이라고 불린다.

<img src="https://user-images.githubusercontent.com/19484971/202378959-3f79bb95-56b7-4e75-bd4a-0bf523079329.png" width=400>
> Windows 의 3D 뷰어로 본 Bowl 모델

### Magick++ 라이브러리 설치

우선 [ImageMagick-7.1.0-52-Q16-HDRI](https://imagemagick.org/script/download.php#windows)를 설치하였다. visual studio 내부에 라이브러리를 내장해서 사용할 예정이었기에 C와 C++ 헤더와 라이브러리를 같이 설치해주었다.

<img src="https://user-images.githubusercontent.com/19484971/201234945-10a88017-3c42-4eaa-8bbc-e794e3a5f895.png" width=600>

환경변수 자동으로 설정해주는 체크박스도 잊지않고 챙겨주었다.

<img src="https://user-images.githubusercontent.com/19484971/201234729-f18c1865-aa52-4555-9b9f-ededc6f02630.png" width=300>

설치가 잘 되었는지 확인해보라는 글대로 cmd 창에

```
magick logo: logo.gif
magick identify logo.gif
magick logo.gif win:
```
를 입력했더니 허리 좋은 호그와트 할아버지가 지팡이들고 흔들고 있는 것을 볼 수 있었다.

<img src="https://user-images.githubusercontent.com/19484971/201255935-a1813866-f6ef-42c9-9810-96f5aa586d24.png" width=500>

<img src="https://user-images.githubusercontent.com/19484971/201256222-4441cf56-00c9-4300-98b2-e1a601cd250c.png" width=600>

다운로드받은 라이브러리를 visual studio 2017의 솔루션 폴더에 내장시켜주었다. 이미 openGL과 openCV 라이브러리를 내장한 상태라 솔루션의 설정을 크게 바꾸어줄 필요는 없었다.

아래는 필자의 솔루션 속성

<img src="https://user-images.githubusercontent.com/19484971/201235464-1a2125f2-be4b-400d-96d4-2d83680c4a01.png" width=600>

<img src="https://user-images.githubusercontent.com/19484971/201235748-990f7d46-b742-49a0-b38d-c9c190618e53.png" width=600>

<img src="https://user-images.githubusercontent.com/19484971/201235800-50b543d5-6e7d-4bbb-ad56-f8d092b253f6.png" width=600>

<img src="https://user-images.githubusercontent.com/19484971/201235886-cd3e9916-330e-401e-a619-c05edce0e01d.png" width=600>

[공식 홈페이지 설치 가이드](https://imagemagick.org/script/magick++.php#install)에서 아래와 같이 `InitializeMagick`에 `DLL`이 설치된 폴더 경로를 넣어주었다.

<img src="https://user-images.githubusercontent.com/19484971/201236559-b8034fa8-ab26-4586-8a41-c27d233fe080.png" width=600>

### Image 에러

그런데.. `예외 발생(0x00007FF942C6C86D(CORE_RL_Magick++_.dll), SurroundViewMonitor.exe): 0xC0000005: 0xFFFFFFFFFFFFFFFF 위치를 읽는 동안 액세스 위반이 발생했습니다..` 라는 에러 문구와 함께 에러가 생겼다. 참고로 위의 `InitializeMagick` 초기화를 하지 않으면 `CORE_RL_Magick++_.dll` 대신 `ntill.dll`이었나..? 에러가 생긴다.

해결방법은 해당 부분의 최하단에 있다.

<img src="https://user-images.githubusercontent.com/19484971/201238202-d972d4a8-ce6f-40ee-a1e2-5e3e9f858317.png" width=600>

1. 환경변수 확인

`Magick++`를 설치한 폴더로 잘.. 설정 되어있다. 설치할 때 체크박스를 잘 신경 썼기 때문에 자동으로 들어가있다.

<img src="https://user-images.githubusercontent.com/19484971/201238559-f19f9599-625f-4a8b-bf31-59513e891ff3.png" width=600>

2. DLL 파일

처음에는 dll 파일을 인식 못한다고 생각하고 구글 검색을 하였을 때 나오는 위치들..

`깃랩 레포지토리 메인 폴더\SurroundViewMonitor`, `C:\Windows\System32`, `C:\Windows\SysWOW64`에..

`CORE_RL_Magick++_.dll`, `CORE_RL_MagickCore_.dll`, `CORE_RL_MagickWand_.dll`을 넣어주었지만 결과는 같았다.

<img src="https://user-images.githubusercontent.com/19484971/201239227-cfc1b926-3b68-49ef-b7f4-3a73613dbbeb.png" width=300>

3. 경로

경로를 잘못 설정했다고 생각하여 구글링해서 진행해보았다.

`C:/img/front.png`, `C:\\img\\front.png`, `C:\img\front.png` (인식 못함) 을 시도했지만, 모두 같은 에러를 볼 수 있었다.

4. 리부트

당연히 Visual Studio와 컴퓨터를 완전히 껐다가 켜보는 방법도 시도하였다. 문제없이 에러를 볼 수 있다.

5. 다른 버전

팀원이신 이랑님이 7버전이 아닌 6버전으로 테스트를 해주셨다. 문제없이 같은 에러를 볼 수 있다.

6. 윈도우 업데이트

[한 곳](https://www.exefiles.com/en/dll/core-rl-magick-dll/)에서는 윈도우 업데이트를 진행해보라고 하여서 했더니 변화가 없었다.

#### 해결방법

지웠다가 재설치를 진행하는 등.. 명확하지 않지만, 아래의 방법으로 해결했다고 생각한다.

1. 우선 Visual Studio에서 C++에 대한 필요한 라이브러리를 다운로드 받는다.

어떤 라이브러리가 필수인지 잘 모르겠지만 적어도 2017 installer 기준으로 `C++를 사용한 데스크톱 개발`의 `Windows 8.1 SDK 및 UCRT SDK`, `x86 및 x64용 Visual C++ MFC`는 에러가 생겨서 설치했던 것으로 기억한다. `Windows 10 SDK(10.0.17134.0)`의 경우에는 다른 `Windows 10 SDK`도 작동되는 것 같으므로 하나만 설치하면 될 것 같다..

단, 다른 윈도우 버전에서는 확인을 못했으므로 다른 버전이라면 진행이 원활하지 않을 수도 있겠다. 또한, 2022 installer의 경우 위의 라이브러리 설치 자체가 없는데 구글링해서 따로 다운로드 받아야 한다.

<img src="https://user-images.githubusercontent.com/19484971/201578453-b53ba018-f2f5-4bc8-915c-286530e0cb3d.png" width=500>
<img src="https://user-images.githubusercontent.com/19484971/201579769-8f1207eb-501f-4d69-b9a3-a44175ef8b46.png" width=700>
<img src="https://user-images.githubusercontent.com/19484971/201579981-0a828e06-84bb-423c-ac90-7892acaed66c.png" width=400>

2. 솔루션 속성

다른 버전의 Visual Studio로 진행해도 되는 것 같으나 플렛폼 도구집합은 꼭 v141로 해주어야 한다. 
또한 `Debug`나 `Release`로 해주는 것은 상관이 없으나, 플렛폼은 꼭 `x64`로 설정해주어야 한다.

<img src="https://user-images.githubusercontent.com/19484971/201580882-5b17af1c-1440-4045-8287-e2ae2fad060b.png" width=500>
<img src="https://user-images.githubusercontent.com/19484971/201581201-55f24c67-1a13-4008-a937-d2bbcae1d0ab.png" width=300>

3. `CORE_DB`

[ImageMagick-Windows 깃헙](https://github.com/ImageMagick/ImageMagick-Windows)을 클론하고 파일 내부에 있는 `CloneRepositories.cmd`을 더블클릭하면 자동으로 다운로드한 후, [한 블로그](https://paragonofjoke.tistory.com/307)의 글을 따라 진행하여 `CORE_DB`파일들을 얻어서 솔루션 메인파일에 추가해주었다.

그리고 `image.cpp` 파일이 없다고 에러가 생길 수 있다. 이런 경우 `SurroundViewMonitor\include\Magick++`에 있는 `image.cpp`이 있는 폴더를 참조경로로 넣어주면 된다. 해당 파일은 `ImageMagick-Windows-main\ImageMagick\Magick++\lib`에서 복사한 것이다.

### 결과물

<img src="https://user-images.githubusercontent.com/19484971/202388480-86515d5b-276c-4928-bef3-36460c28c27b.gif" width=500>

### 추후 개발

<img src="https://user-images.githubusercontent.com/19484971/202389493-19032c5e-d7d9-4118-a063-40c2759f8673.png" width=500>

### 시행착오

<img src="https://user-images.githubusercontent.com/19484971/202388930-afa650ef-f5f2-460b-87b0-c736cc18900a.png" width=500>

<img src="https://user-images.githubusercontent.com/19484971/202390353-f81d82aa-d075-49e8-9b8d-0750c246b6c1.png" width=500>

### 어려웠던 점

1. magick++ 와 C++

C++ 언어 자체가 생소하기도 하고 라이브러리를 사용하기 위해서 환경설정을 적절하게 해주는 것이 너무 어려웠다. 위의 글들로 충분히 이해하였다고 생각하고 내용은 생략한다.

2. 파라미터

C++과 Python에서 사용하는 자료형과 함수, 파라미터가 크게 다른데, 이러한 점으로 생소한 자료형과 함수를 사용해야 했어서 간단한 변환작업인데도 시간이 오래 걸린 경우가 있다.

대표적으로 아래와 같은 문제가 있었다.

 - `Magick::Image`의 픽셀 값을 가져오는 코드
    - C언어를 기반으로 하고 있어서 픽셀의 위치를 수리적으로 직접 계산하여 접근해야 한다.
    - [이곳](https://stackoverflow.com/questions/47781396/imagemagick-c-version-7-modify-pixel-value-in-blank-image)을 참고하였다.
    - [공식 홈페이지](https://www.imagemagick.org/Magick++/Image++.html#Raw%20Image%20Pixel%20Access)의 글도 참고하였다.
    - 추후에 `Magick::Image`를 `cv::Mat`으로 변환하여 사용하였다.
- `Magick::Image`의 픽셀 데이터 순서
    - 놀랍게도 RGB가 아니라 BGR
    - [스택오버플로우](https://stackoverflow.com/questions/7899108/opencv-get-pixel-channel-value-from-mat-image)를 보니 외국인 친구들도 햇갈려하는 것 같다. 
- `Mat` 초기화
    - `CV_8UC3`라는 특수한 openCV 자료형을 사용하였다. 의미는 unsigned char 3채널(B,G,R)
    - 처음에는 `Mat::zero` 함수를 사용하였으나, zero함수를 사용하면 Scalar(0)으로 고정되어 3차원 배열이 불가능하였고 오랫동안 해매었다.
    - 두 블로그를 참고하였는데 하나는 [여기](https://cafepurple.tistory.com/42) 다른 곳은 [여기](https://3001ssw.tistory.com/172)
    - `Mat`을 초기화 할 때 다른 함수와는 다르게 높이 파라미터를 먼저 넣고 다음으로 너비 파라미터를 넣어주었는데, 다른 함수와 혼동되었다.
    <img src="https://user-images.githubusercontent.com/19484971/201603987-6dc19abe-e587-4de4-8992-587b6b5b9f11.png" width=500>
- `cv::copyTo()` 
    - 이미지 합성을 위해서 찾은 함수
    - 참고한 [블로그](https://m.blog.naver.com/hwidong0102/221771828880)
    - `mask error` 가 생겨서 찾아보니 [스택오버플로우](https://m.blog.naver.com/hwidong0102/221771828880)에서 말하길 큰 이미지 객체에서 함수를 사용하여 작은 이미지 객체를 파라미터로 넣는 것이 아니라, 작은 이미지 객체에서 함수를 부르는 것 이었다;
- dll 파일 인식
    - `glew32.dll`과 `glu32.dll`을 `C:\Windows\SysWOW64`에 넣어서 해결하였다.
    - 문제는 시스템 환경 변수나 OS 등의 이유로 인식하는 폴더 위치가 다르거나 설치된 파일이 다를 수 있어서 다른 dll 파일 인식에러가 날 수 있다. 구글에서 해당 dll 파일을 검색해서 `C:\Windows\SysWOW64` 혹은 `C:\Windows\System32`에 넣어주어야 한다.

### 참고

- [openGL 노션](https://www.notion.so/3D-OpenGL-578443569f6947fab3b34078455225b6)
- [노션 openGL 튜토리얼 정리](https://www.notion.so/openGL-8deeed9e075d4bd49afa64066b6f092e)
    - [openGL 튜토리얼](http://www.opengl-tutorial.org/)
- [카메라 이론](https://www.notion.so/Coordinate-System-791c56e52b124823823861ec4145dca9)
    - [다크 프로그래머](https://darkpgmr.tistory.com/category/%EC%98%81%EC%83%81%EC%B2%98%EB%A6%AC?page=1)