import cv2
import numpy as np

def undistort_left(img, ratio):
    DIM=(1280, 720)
    K=np.array([[486.43710381577273, 0.0, 643.0021325671074], [0.0, 485.584911786959, 402.9808925210084], [0.0, 0.0, 1.0]])
    D=np.array([[-0.06338733272909226], [-0.007861033496168955], [0.005073683389947028], [-0.0010639404289377306]])

    ## new_K 설정 
    new_K = K.copy()
    new_K[0,0]=K[0,0]/ratio
    new_K[1,1]=K[1,1]/ratio

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)

    # print(map1, map2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def undistort(img, ratio):
    DIM=(1280, 720)
    K=np.array([[455.8515274977241, 0.0, 655.7621645964248], [0.0, 455.08604281075947, 367.3548823943176], [0.0, 0.0, 1.0]])
    D=np.array([[-0.02077978156022359], [-0.02434621475644252], [0.009725498728069807], [-0.0018108318059442028]])
    

    ## new_K 설정 
    new_K = K.copy()
    new_K[0,0]=K[0,0]/ratio
    new_K[1,1]=K[1,1]/ratio

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)



    # print(map1, map2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)
cap3 = cv2.VideoCapture(3)
cap4 = cv2.VideoCapture(4)

cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap3.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap4.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap4.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

print('width1 :%d, height1 : %d' % (cap1.get(3), cap1.get(4)))
print('width2 :%d, height2 : %d' % (cap2.get(3), cap2.get(4)))
print('width3 :%d, height3 : %d' % (cap3.get(3), cap3.get(4)))
print('width4 :%d, height4 : %d' % (cap4.get(3), cap4.get(4)))

num = 1
while(True):

    ret1, frame_1 = cap1.read()    # Read 결과와 frame
    ret2, frame_2 = cap2.read()
    ret3, frame_3 = cap3.read()
    ret4, frame_4 = cap4.read()

    if(ret1) :
        # cam1 = undi_top(frame1, 'right')         
        cv2.imshow('frame_1', frame_1) 
        undistorted_front = undistort(frame_1, 1.5)
        # cv2.imshow('frame_1', undistorted_img1)     

        pass

    if(ret2) :
        # cam2 = undi_top(frame2, 'front')   
        # cv2.imshow('frame_2_front', cam2)
        cv2.imshow('frame_2', frame_2) 
        undistorted_back = undistort(frame_2, 1.5)
        # cv2.imshow('frame_2', undistorted_img2) 
        pass

    if(ret3) :
        # cam3 = undi_top(frame3, 'back')     
        # cv2.imshow('frame_3_back', cam3)
        cv2.imshow('frame_3', frame_3) 
        undistorted_right = undistort(frame_3, 1.5)
        # cv2.imshow('frame_3', undistorted_img3) 
        pass

    if(ret4) :
        cv2.imshow('frame_4', frame_4) 
        undistorted_left = undistort_left(frame_4, 1.5)

        
    
    key = cv2.waitKey(1)

    if key == ord('z'):
        cv2.imwrite('front' + str(num) + '.png', frame_1)
        cv2.imwrite('back' + str(num) + '.png', frame_2)
        cv2.imwrite('right' + str(num) + '.png', frame_3)
        cv2.imwrite('left' + str(num) + '.png', frame_4)
        cv2.imwrite('undistorted_front' + str(num) + '.png', undistorted_front)
        cv2.imwrite('undistorted_back' + str(num) + '.png', undistorted_back)
        cv2.imwrite('undistorted_right' + str(num) + '.png', undistorted_right)
        cv2.imwrite('undistorted_left' + str(num) + '.png', undistorted_left)
        print('all_frame + undi_' + str(num) + ' =============================== ') 
        num += 1

    elif key == ord('c'):
        cv2.destroyAllWindows()
