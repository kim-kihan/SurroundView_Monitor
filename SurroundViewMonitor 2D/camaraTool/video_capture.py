from multiprocessing import Process, Manager

import cv2
import numpy as np
import time


# 1. 초기값 세팅
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
BEV_WIDTH = 1020
BEV_HEIGHT = 1128
CAR_WIDTH = 155
CAR_HEIGHT = 315
FOCAL_SCALE = 0.65
SIZE_SCALE = 1


# 2. 왜곡 계수 및 intrinsic matrix 생성
DIM=(1280, 720)
LEFT_K=np.array([[486.43710381577273, 0.0, 643.0021325671074], [0.0, 485.584911786959, 402.9808925210084], [0.0, 0.0, 1.0]])
LEFT_D=np.array([[-0.06338733272909226], [-0.007861033496168955], [0.005073683389947028], [-0.0010639404289377306]])
K=np.array([[455.8515274977241, 0.0, 655.7621645964248], [0.0, 455.08604281075947, 367.3548823943176], [0.0, 0.0, 1.0]])
D=np.array([[-0.02077978156022359], [-0.02434621475644252], [0.009725498728069807], [-0.0018108318059442028]])

new_LEFT_K = LEFT_K.copy()
new_LEFT_K[0,0]=LEFT_K[0,0]/1.5
new_LEFT_K[1,1]=LEFT_K[1,1]/1.5
left_map1, left_map2 = cv2.fisheye.initUndistortRectifyMap(LEFT_K, LEFT_D, np.eye(3), new_LEFT_K, DIM, cv2.CV_16SC2)

new_K = K.copy()
new_K[0,0]=K[0,0]/1.5
new_K[1,1]=K[1,1]/1.5
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)



# 3. 호모그래피 초기화
front_homography = np.array([
    [3.8290818190655296, 5.66226108412413, -1961.3463578215583], 
    [0.04676494495778877, 7.593794538002965, -1398.2503661266026], 
    [9.936840064368021e-05, 0.011048826673722773, 1.0]])

back_homography = np.array([
    [-3.6014363756395578, 4.36751919471232, 2793.845376423345], 
    [-0.16343764199990043, 3.070397421396325, 2357.357976286147],
    [-0.00020012270934899256, 0.008545448669492679, 1.0]])

right_homography = np.array([
    [-0.08221005850690113, 1.6235034060147169, 2032.459375714023], 
    [2.945881257892563, 3.582400762214573, -1439.2015581787357], 
    [-0.00011329233289173604, 0.0068701395422870026, 0.9999999999999999]])

left_homography = np.array([
    [0.1022385737675907, 5.770995782571725, -1228.0525962849354], 
    [-3.1690352115113147, 3.516499167979982, 2659.2521230613215], 
    [0.0002888891947907289, 0.006800266459274919, 1.0]])



# 4. 차량 설치 공간 init
def padding(img,width,height):
    H = img.shape[0]
    W = img.shape[1]
    top = (height - H) // 2 
    bottom = (height - H) // 2 
    if top + bottom + H < height:
        bottom += 1
    left = (width - W) // 2 
    right = (width - W) // 2 
    if left + right + W < width:
        right += 1
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value = (0,0,0)) 
                             #copyMakeBorder 함수는 이미지를 액자 형태로 만들 때 사용할 수 있습니다. 이미지에 가장자리가 추가
    return img
car = cv2.imread('C:/Users/multicampus/Desktop/porche.png')
car = cv2.resize(car,(320,450))
car = padding(car, BEV_WIDTH, BEV_HEIGHT)



# 5. 카메라 객체 생성
cap1 = cv2.VideoCapture(1)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap1.set(cv2.CAP_PROP_BUFFERSIZE, 2)

cap2 = cv2.VideoCapture(2)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap2.set(cv2.CAP_PROP_BUFFERSIZE, 2)

cap3 = cv2.VideoCapture(3)
cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap3.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap3.set(cv2.CAP_PROP_BUFFERSIZE, 2)

cap4 = cv2.VideoCapture(4)
cap4.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap4.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap4.set(cv2.CAP_PROP_BUFFERSIZE, 2)



# 6. blending 마스킹
class BlendMask:
    def __init__(self,name):
        mf = self.get_mask('front')
        mb = self.get_mask('back')
        ml = self.get_mask('left')
        mr = self.get_mask('right')
        self.get_lines()
        if name == 'front':
            mf = self.get_blend_mask(mf, ml, self.lineFL, self.lineLF)
            mf = self.get_blend_mask(mf, mr, self.lineFR, self.lineRF)
            self.mask = mf
        if name == 'back':
            mb = self.get_blend_mask(mb, ml, self.lineBL, self.lineLB)
            mb = self.get_blend_mask(mb, mr, self.lineBR, self.lineRB)
            self.mask = mb
        if name == 'left':
            ml = self.get_blend_mask(ml, mf, self.lineLF, self.lineFL)
            ml = self.get_blend_mask(ml, mb, self.lineLB, self.lineBL)
            self.mask = ml
        if name == 'right':
            mr = self.get_blend_mask(mr, mf, self.lineRF, self.lineFR)
            mr = self.get_blend_mask(mr, mb, self.lineRB, self.lineBR)
            self.mask = mr
        self.weight = np.repeat(self.mask[:, :, np.newaxis], 3, axis=2) / 255.0
        self.weight = self.weight.astype(np.float32)
        
    def get_points(self, name):  # Bird Eye View를 위한 point 값 (변환 좌표)
        if name == 'front':
            points = np.array([
                [0, 0],
                [BEV_WIDTH, 0], 
                [BEV_WIDTH, BEV_HEIGHT/5], 
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                [0, BEV_HEIGHT/5], 
            ]).astype(np.int32)
        elif name == 'back':
            points = np.array([
                [0, BEV_HEIGHT],
                [BEV_WIDTH, BEV_HEIGHT],
                [BEV_WIDTH, BEV_HEIGHT - BEV_HEIGHT/5],
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2], 
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2],
                [0, BEV_HEIGHT - BEV_HEIGHT/5],
            ]).astype(np.int32)
        elif name == 'left':
            points = np.array([
                [0, 0],
                [0, BEV_HEIGHT], 
                [BEV_WIDTH/5, BEV_HEIGHT], 
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2],
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                [BEV_WIDTH/5, 0]
            ]).astype(np.int32)
        elif name == 'right':
            points = np.array([
                [BEV_WIDTH, 0],
                [BEV_WIDTH, BEV_HEIGHT], 
                [BEV_WIDTH - BEV_WIDTH/5, BEV_HEIGHT],
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2], 
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                [BEV_WIDTH - BEV_WIDTH/5, 0]
            ]).astype(np.int32)
        else:
            raise Exception("name should be front/back/left/right")
        return points
    
    def get_mask(self, name): 
        mask = np.zeros((BEV_HEIGHT,BEV_WIDTH), dtype=np.uint8) # 마스크 생성, Bird Eye View 높이, 너비만큼 배열 생성
        points = self.get_points(name)
        return cv2.fillPoly(mask, [points], 255)
    
    def get_lines(self):
        self.lineFL = np.array([
                        [0, BEV_HEIGHT/5], 
                        [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                    ]).astype(np.int32)
        self.lineFR = np.array([
                        [BEV_WIDTH, BEV_HEIGHT/5], 
                        [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                    ]).astype(np.int32)
        self.lineBL = np.array([
                        [0, BEV_HEIGHT - BEV_HEIGHT/5], 
                        [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2],
                    ]).astype(np.int32)
        self.lineBR = np.array([
                        [BEV_WIDTH, BEV_HEIGHT - BEV_HEIGHT/5], 
                        [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2],
                    ]).astype(np.int32)
        self.lineLF = np.array([
                        [BEV_WIDTH/5, 0],
                        [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2]
                    ]).astype(np.int32)
        self.lineLB = np.array([
                        [BEV_WIDTH/5, BEV_HEIGHT],
                        [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2]
                    ]).astype(np.int32)
        self.lineRF = np.array([
                        [BEV_WIDTH - BEV_WIDTH/5, 0],
                        [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2]
                    ]).astype(np.int32)
        self.lineRB = np.array([
                        [BEV_WIDTH - BEV_WIDTH/5, BEV_HEIGHT],
                        [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2]
                    ]).astype(np.int32)
        
    def get_blend_mask(self, maskA, maskB, lineA, lineB): #maskA 값을 합성하는 함수 
        overlap = cv2.bitwise_and(maskA, maskB) # mask 영역에서 서로 공통으로 겹치는 부분 출력
        indices = np.where(overlap != 0)


        for y, x in zip(*indices):
            distA = cv2.pointPolygonTest(np.array(lineA), (x.astype(np.int16), y.astype(np.int16)), True)
            distB = cv2.pointPolygonTest(np.array(lineB), (x.astype(np.int16), y.astype(np.int16)),  True)
            # 이미지에서 해당 Point가 Contour의 어디에 위치해 있는지

            # 1. contour : Contour Points들을 인자로 받는다.
            # 2. pt : Contour에 테스트할 Point를 인자로 받는다.
            # 3. measureDist : boolean 데이터 타입. 
            maskA[y, x] = distA**2 / (distA**2 + distB**2 + 1e-6) * 255
        return maskA
    
    def __call__(self, img):
        return (img * self.weight).astype(np.uint8)   

# 6-1. 마스킹 객체 생성
mask_front = BlendMask('front')
mask_left = BlendMask('left')
mask_right = BlendMask('right')
mask_back = BlendMask('back')



# 7. 멀티 프로세싱 task 1  (영상획득 + 왜곡보정 + 시점변환 + 명도조절 + 합성마스킹)
def WORK_PROCESS(d, id):
    videoFileName_1 = 'front_ori.avi'
    videoFileName_2 = 'left_ori.avi'
    videoFileName_3 = 'right_ori.avi'
    videoFileName_4 = 'back_ori.avi'

    w = round(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)) # width
    h = round(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)) #height
    fps = cap1.get(cv2.CAP_PROP_FPS) #frame per second
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') #fourcc
    # delay = round(1000/fps) #set interval between frame


    if id == 1: # 1번(front) 화면
        while True:
            frame = cv2.imread('C:/Users/multicampus/Desktop/S07P31D108/2d/data/extrinsic/front.png')
            # frame = cv2.imread('front.png')
            front_ori = cv2.VideoWriter(videoFileName_1, fourcc, fps, (w,h))
            # ret1, frame = cap1.read()

            front_ori.write(frame)
            
            tmp = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            tmp = cv2.warpPerspective(tmp, front_homography, (1020, 1128))
            
            # luminance balancing
            hsv = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)
            hf, sf, vf = cv2.split(hsv)

            V_f = np.mean(vf)
            d['V_f'] = V_f

            V_mean = (d['V_f'] + d['V_b'] + d['V_l'] +d['V_r']) * .4
            vf = cv2.add(vf,(V_mean - V_f))
            tmp = cv2.merge([hf,sf,vf])

            tmp = cv2.cvtColor(tmp, cv2.COLOR_HSV2BGR)

            # masking
            tmp = mask_front(tmp)
            d['front'] = tmp

            if cv2.waitKey(1) == ord('c'): #wait 10ms until user input 'esc'
                front_ori.release() 
                break     

    elif id == 2: # 2번(left) 화면
        while True:
            left_ori = cv2.VideoWriter(videoFileName_2, fourcc, fps, (w,h))
            
            frame = cv2.imread('C:/Users/multicampus/Desktop/S07P31D108/2d/data/extrinsic/left.png')
            # frame = cv2.imread('left.png')
            # ret1, frame = cap2.read()
            left_ori.write(frame)
            tmp = cv2.remap(frame, left_map1, left_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            tmp = cv2.warpPerspective(tmp, left_homography, (1020, 1128))

            # luminance balancing
            hsv = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)
            hl, sl, vl = cv2.split(hsv)

            V_l = np.mean(vl)
            d['V_l'] = V_l

            V_mean = (d['V_f'] + d['V_b'] + d['V_l'] +d['V_r']) * .4
            vl = cv2.add(vl,(V_mean - V_l))
            tmp = cv2.merge([hl,sl,vl])

            tmp = cv2.cvtColor(tmp, cv2.COLOR_HSV2BGR)

            # masking
            tmp = mask_left(tmp)
            d['left'] = tmp
            # cv2.imshow(f"{id}", tmp)

            if cv2.waitKey(1) == ord('c'): #wait 10ms until user input 'esc'
                left_ori.release() 
                break

    elif id == 3: # 3번(right) 화면
        while True:
            right_ori = cv2.VideoWriter(videoFileName_3, fourcc, fps, (w,h))
            frame = cv2.imread('C:/Users/multicampus/Desktop/S07P31D108/2d/data/extrinsic/right.png')
            # frame = cv2.imread('right.png')
            # ret1, frame = cap3.read()

            right_ori.write(frame)
            tmp = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            tmp = cv2.warpPerspective(tmp, right_homography, (1020, 1128))

            # luminance balancing
            hsv = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)
            hr, sr, vr = cv2.split(hsv)

            V_r = np.mean(vr)
            d['V_r'] = V_r

            V_mean = (d['V_f'] + d['V_b'] + d['V_l'] +d['V_r']) * .4
            vr = cv2.add(vr,(V_mean - V_r))
            tmp = cv2.merge([hr,sr,vr])

            tmp = cv2.cvtColor(tmp, cv2.COLOR_HSV2BGR)

            # masking
            tmp = mask_right(tmp)
            d['right'] = tmp
            # cv2.imshow(f"{id}", tmp)

            if cv2.waitKey(1) == ord('c'): #wait 10ms until user input 'esc'
                right_ori.release() 
                break     

    elif id == 4: # 4번(back) 화면
        while True:
            back_ori = cv2.VideoWriter(videoFileName_4, fourcc, fps, (w,h))
            frame = cv2.imread('C:/Users/multicampus/Desktop/S07P31D108/2d/data/extrinsic/back.png')
            # frame = cv2.imread('back.png')
            # ret1, frame = cap4.read()

            back_ori.write(frame)
            tmp = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            tmp = cv2.warpPerspective(tmp, back_homography, (1020, 1128))

            # luminance balancing
            tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)
            hb, sb, vb = cv2.split(tmp)

            V_b = np.mean(vb)
            d['V_b'] = V_b
            V_mean = (d['V_f'] + d['V_b'] + d['V_l'] +d['V_r']) * .4
            vb = cv2.add(vb,(V_mean - V_b))
            tmp = cv2.merge([hb,sb,vb])

            tmp = cv2.cvtColor(tmp, cv2.COLOR_HSV2BGR)

            # masking
            tmp = mask_back(tmp)
            d['back'] = tmp
            # cv2.imshow(f"{id}", tmp)

            if cv2.waitKey(1) == ord('c'): #wait 10ms until user input 'esc'
                # back_ori.release() 
                break    
            




# 8. 멀티 프로세싱 task 2  (합성 + 색 조절)
def svm(d, id):

    videoFileName = 'SVM.avi'
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') #fourcc
    svm_out = cv2.VideoWriter(videoFileName, fourcc, 20, (670,752))

    while True:
        if len(d['front']) < 1 or len(d['left']) < 1 or len(d['right']) < 1 or len(d['back']) < 1:
                print(len(d['front']), len(d['left']) , len(d['right']) , len(d['back']))
                continue
        else:
            print("passssssssssssssss")
            while True:
                time1 = time.time()

                # 1) 4방향 작업물 합성
                surround1 = cv2.add(d["front"], d["back"])
                surround2 = cv2.add(d["left"], d["right"])
                surround = cv2.add(surround1, surround2)
                
                # 2) 색 balancing
                b, g, r = cv2.split(surround)
                B = np.mean(b)
                G = np.mean(g)
                R = np.mean(r)
                K = (R + G + B) * .3
                Kb = K / B
                Kg = K / G
                Kr = K / R
                cv2.addWeighted(b, Kb, 0, 0, 0, b)
                cv2.addWeighted(g, Kg, 0, 0, 0, g)
                cv2.addWeighted(r, Kr, 0, 0, 0, r)
                surround = cv2.merge([b,g,r])

                # 3) 차량 이미지 합성
                surround = cv2.add(surround,car)
                surround = cv2.resize(surround,(670,752))

                # 4) 결과 출력
                cv2.imshow('surround', surround)
                svm_out.write(surround)

                if cv2.waitKey(1) == ord('c'): #wait 10ms until user input 'esc'
                    svm_out.release() 
                    break

                time2 = time.time()
                print("SVM 실행 FPS", 1 / (time2 - time1))



if __name__ == '__main__':
    # 1) 프로세스 간 공유 메모리 설정
    manager = Manager()
    d = manager.dict()

    # 2) 저장소 1 : 영상 처리 공유용
    d['front'] = np.array([])
    d['left'] = np.array([])
    d['right'] = np.array([])
    d['back'] = np.array([])

    # 3) 저장소 2 : 명도 평균값 공유용
    d["V_f"] = 1
    d["V_b"] = 1
    d["V_l"] = 1
    d["V_r"] = 1

    # 4) 프로세스 분기 설정
    process1 = Process(target=WORK_PROCESS, args=(d, 1))
    process2 = Process(target=WORK_PROCESS, args=(d, 2))
    process3 = Process(target=WORK_PROCESS, args=(d, 3))
    process4 = Process(target=WORK_PROCESS, args=(d, 4))
    process5 = Process(target=svm, args=(d, 5))

    # 5) 실행 명령
    process1.start()
    process2.start()
    process3.start()
    process4.start()
    process5.start()

    print("All processes have been started")

    process1.join()
    process2.join()
    process3.join()
    process4.join()
    process5.join()