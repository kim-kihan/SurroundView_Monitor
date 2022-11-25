import cv2
src = cv2.imread('./data/triangle/extrinsic.png')

# 이미지를 자른다.
dst = src[0:720, 1280*3:1280*4].copy()

cv2.imshow('source', src)
cv2.imshow('cut image', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('./data/triangle/extrinsic4.png', dst)

