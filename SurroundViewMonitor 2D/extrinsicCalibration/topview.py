import cv2
import numpy as np
import glob
# from undistort import *
from cv2 import FONT_HERSHEY_COMPLEX



def topview(img_original, side):    
        
    chessboardx=4
    chessboardy=4 
    CHECKERBOARD = (chessboardx,chessboardy)
    img = img_original
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    print(ret)
    corner1x = corners[0][0][0]
    corner1y = corners[0][0][1]
    corner2x = corners[chessboardx-1][0][0]
    corner2y = corners[chessboardx-1][0][1]
    corner3x = corners[(chessboardx)*(chessboardy-1)][0][0]
    corner3y = corners[(chessboardx)*(chessboardy-1)][0][1]
    corner4x = corners[(chessboardx)*(chessboardy)-1][0][0]
    corner4y = corners[(chessboardx)*(chessboardy)-1][0][1]

    
    pts = np.array([[corner1x, corner1y], [corner2x, corner2y], [corner3x, corner3y], [corner4x, corner4y]], dtype=np.float32)
    idx=0
    for pt in pts:
        idx+=1
        cv2.circle(img, tuple(pt.astype(np.int)), 1, (0,0,255), 5)
        cv2.putText(img,str(idx), tuple(pt.astype(np.int)),FONT_HERSHEY_COMPLEX,1,(0,0,255))
    # compute IPM matrix and apply it

    

    if side == 'front':

        point1y=215
        point1x=295
        point2y=215
        point2x=345
        point3y=265
        point3x=295
        point4y=265
        point4x=345 

        # ipm_pts = np.array([[448,609], [580,609], [580,741], [448,741]], dtype=np.float32)
        ipm_pts = np.array([[point1x,point1y], [point2x,point2y], [point3x,point3y], [point4x,point4y]], dtype=np.float32)
        ipm_matrix = cv2.getPerspectiveTransform(pts, ipm_pts)


        size = (640, 480)
        # angle = cv2.ROTATE_90_CLOCKWISE
        ipm = cv2.warpPerspective(img, ipm_matrix, size)
        # ipm = cv2.rotate(ipm, angle) 

    elif side == 'back':
        point1x=300
        point1y=532
        point2x=220
        point2y=532
        point3x=300
        point3y=452
        point4x=220 
        point4y=452

        # ipm_pts = np.array([[448,609], [580,609], [580,741], [448,741]], dtype=np.float32)
        print(pts)
        ipm_pts = np.array([[point1x,point1y], [point2x,point2y], [point3x,point3y], [point4x,point4y]], dtype=np.float32)
        ipm_matrix = cv2.getPerspectiveTransform(pts, ipm_pts)

        size = (520, 596)
        # angle = cv2.ROTATE_90_COUNTERCLOCKWISE
        ipm = cv2.warpPerspective(img, ipm_matrix, size)
        # ipm = cv2.rotate(ipm, angle) 

    elif side == 'right':
        
        point1x=215
        point1y=295
        point2x=215
        point2y=345
        point3x=265
        point3y=295
        point4x=265
        point4y=345  



        ipm_pts = np.array([[point3x,point3y], [point4x,point4y], [point1x,point1y], [point2x,point2y]], dtype=np.float32)  
        # ipm_pts = np.array([[480/2,0], [215,29], [480/2,640/2], [0,640/2]], dtype=np.float32)
        ipm_matrix = cv2.getPerspectiveTransform(pts, ipm_pts)
        
        size = (480, 640)
        # angle = cv2.ROTATE_180
        ipm = cv2.warpPerspective(img, ipm_matrix, size)
        # ipm = cv2.rotate(ipm, angle) 

    elif side == 'left' :
        point1x=215
        point1y=295
        point2x=215
        point2y=345
        point3x=265
        point3y=295
        point4x=265
        point4y=345 

        # ipm_pts = np.array([[448,609], [580,609], [580,741], [448,741]], dtype=np.float32)
        ipm_pts = np.array([[point3x,point3y], [point4x,point4y], [point1x,point1y], [point2x,point2y]], dtype=np.float32)
        ipm_matrix = cv2.getPerspectiveTransform(pts, ipm_pts)
        
        size = (480, 640)
        # angle = cv2.ROTATE_180
        ipm = cv2.warpPerspective(img, ipm_matrix, size)
        # ipm = cv2.rotate(ipm, angle) 

    print('=======================pts=========================')
    print(pts)

    print('=======================ipm_pts=======================')
    print(ipm_pts)    
    
    print('================== ipm_matrix ==============')
    print(ipm_matrix)

    return ipm

# C:\Users\multicampus\Desktop\S07P31D108\back_top_undi.png
# =====================================================================

side = 'back'
# fname = 'pjh/1031/cap_undi_done/' + side + '/' + side + '_cap_undistorted.png'

# fname = 'pjh/1029/topview_undi_done_v2/' + side + '_top_undi_v2.png'
fname = 'C:/Users/multicampus/Desktop/S07P31D108/back_undistorted.png'
img = cv2.imread(fname)

# print(str(img))
# undistorted_img = undistort(img, K, D, DIM) 
# cv2.imwrite('left_undi.png', undistorted_img)

topview_img = topview(img, side)
# cv2.imshow('left_undi', undistorted_img)
cv2.imshow(side + '_top', topview_img) 
# cv2.imwrite(side + '_undi_top_v3.png', topview_img)
cv2.imshow(side + '_ori', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 왜곡보정 할 때, 
