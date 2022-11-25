import cv2
import sys
import glob

img_names = glob.glob('pjh/data/stitch_test/*.png')

# 불러온 영상을 imgs에 저장
imgs = []
for name in img_names:
    print(name)
    img = cv2.imread(name)
    
    if img is None:
        print('Image load failed!')
        sys.exit()
        
    imgs.append(img)

# 객체 생성
stitcher = cv2.Stitcher_create()

# 이미지 스티칭
status, dst = stitcher.stitch(imgs)

if status != cv2.Stitcher_OK:
    print('Stitch failed!')
    sys.exit()
    
# 결과 영상 저장
cv2.imshow('output.png', dst)

# 출력 영상이 화면보다 커질 가능성이 있어 WINDOW_NORMAL 지정
cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
cv2.imshow('dst',dst)
cv2.waitKey()
cv2.destroyAllWindows()