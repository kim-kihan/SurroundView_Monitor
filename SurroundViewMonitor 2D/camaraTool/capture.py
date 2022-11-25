import cv2
import datetime

cap = cv2.VideoCapture(1)
# cap2 = cv2.VideoCapture(2)
print(datetime.datetime.now())
num = 1
while True:
    ret, img = cap.read()
    # ret2, img2 = cap2.read()
    cv2.imshow('camera', img)
    # cv2.imshow('camera2', img2)
    if cv2.waitKey(1) == ord('c'):
        print('captured_' + str(num))
        img_captured = cv2.imwrite('captured_' + str(num) +'.png', img)
        num += 1
    
    if cv2.waitKey(1) == ord('q'):
        img_captured = cv2.imwrite
        break

cap.release()
cv2.destroyAllWindows()

