import cv2
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Generate Surrounding Camera Bird Eye View")
parser.add_argument('-fw', '--FRAME_WIDTH', default=1280, type=int, help='Camera Frame Width')      # 원본 이미지 길이
parser.add_argument('-fh', '--FRAME_HEIGHT', default=720, type=int, help='Camera Frame Height')    # 원본 이미지 높이
parser.add_argument('-bew', '--BEV_WIDTH', default= 340, type=int, help='BEV Frame Width')       # 탑뷰 이미지 길이
parser.add_argument('-beh', '--BEV_HEIGHT', default= 376, type=int, help='BEV Frame Height')     # 탑뷰 이미지 높이
parser.add_argument('-cw', '--CAR_WIDTH', default=104.75, type=int, help='Car Frame Width')        # 차량 이미지 길이
parser.add_argument('-ch', '--CAR_HEIGHT', default=400, type=int, help='Car Frame Height')      # 차량 이미지 높이
parser.add_argument('-fs', '--FOCAL_SCALE', default=0.65, type=float, help='Camera Undistort Focal Scale')     # 카메라 왜곡되지 않은 초점 스케일
parser.add_argument('-ss', '--SIZE_SCALE', default=1, type=float, help='Camera Undistort Size Scale')       # 카메라 왜곡되지 않은 크기 스케일
parser.add_argument('-blend','--BLEND_FLAG', default=False, type=bool, help='Blend BEV Image (Ture/False)')
parser.add_argument('-balance','--BALANCE_FLAG', default=False, type=bool, help='Balance BEV Image (Ture/False)')
args = parser.parse_args()

FRAME_WIDTH = args.FRAME_WIDTH
FRAME_HEIGHT = args.FRAME_HEIGHT
BEV_WIDTH = args.BEV_WIDTH
BEV_HEIGHT = args.BEV_HEIGHT
CAR_WIDTH = args.CAR_WIDTH
CAR_HEIGHT = args.CAR_HEIGHT
FOCAL_SCALE = args.FOCAL_SCALE
SIZE_SCALE = args.SIZE_SCALE


parser2 = argparse.ArgumentParser(description="Homography from Source to Destination Image")
parser2.add_argument('-bw','--BORAD_WIDTH', default=14, type=int, help='Chess Board Width (corners number)')
parser2.add_argument('-bh','--BORAD_HEIGHT', default=5, type=int, help='Chess Board Height (corners number)')
parser2.add_argument('-size','--SCALED_SIZE', default=10, type=int, help='Scaled Chess Board Square Size (image pixel)')
parser2.add_argument('-subpix_s','--SUBPIX_REGION_SRC', default=3, type=int, help='Corners Subpix Region of img_src')
parser2.add_argument('-subpix_d','--SUBPIX_REGION_DST', default=3, type=int, help='Corners Subpix Region of img_dst')
parser2.add_argument('-store_path', '--STORE_PATH', default='./data/', type=str, help='Path to Store Centerd/Scaled Images')
args2 = parser2.parse_args()

class CenterImage:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.param = {'tl': None, 'br': None, 'current_pos': None,'complete': False}
        self.display = "CLICK image center and press Y/N to validate, ESC to stay original"

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            img = self.raw_frame.copy()
            param['current_pos'] = (x, y)
            if param['tl'] is None:
                param['tl'] = param['current_pos'] 
        if event == cv2.EVENT_MOUSEMOVE and param['tl'] is not None and not param['complete']:
            img = self.raw_frame.copy()
            param['current_pos'] = (x, y)
            cv2.rectangle(img, param['tl'], param['current_pos'], (0, 0, 255))
            cv2.imshow(self.display, img)
        if event == cv2.EVENT_LBUTTONUP and param['tl'] is not None:
            img = self.raw_frame.copy()
            param['br'] = (x, y)
            param['complete'] = True
            cv2.rectangle(img, param['tl'], param['br'], (0, 0, 255))
            cv2.imshow(self.display, img)
            self.x = (param['tl'][0] + param['br'][0] ) // 2
            self.y = (param['tl'][1] + param['br'][1] ) // 2
            text = " %d,%d? (y/n)" % (self.x, self.y)
            cv2.circle(img, (self.x, self.y), 1, (0, 0, 255), thickness = 2)
            cv2.putText(img, text, (self.x, self.y), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 0, 0), thickness = 1)
            cv2.imshow(self.display, img)
        self.param = param
        
    def translate(self, img):
        shift_x = img.shape[1] // 2 - self.x
        shift_y = img.shape[0] // 2 - self.y
        M = np.float32([[1,0,shift_x],[0,1,shift_y]])
        img_dst = cv2.warpAffine(img,M,img.shape[1::-1])
        return img_dst
        
    def __call__(self, raw_frame):   
        self.raw_frame = raw_frame
        cv2.namedWindow(self.display, flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(self.display, self.mouse, self.param)
        while True:
            cv2.imshow(self.display, self.raw_frame)
            key = cv2.waitKey(0)
            if key in (ord("y"), ord("Y")):
                break
            elif key in (ord("n"), ord("N")):
                self.x = 0
                self.y = 0
                self.param['tl'] = None
                self.param['br'] = None
                self.param['current_pos'] = None
                self.param['complete'] = None
            elif key == 27: 
                self.x = 0
                self.y = 0
                break
        cv2.destroyAllWindows()
        if not (self.x == 0 and self.y == 0):
            return self.translate(self.raw_frame)
        else:
            return self.raw_frame

class ScaleImage:
    def __init__(self, corners):        
        self.calc_dist(corners)
        print('scale image from {} to {}'.format(self.dist_square,args2.SCALED_SIZE))
        self.scale_factor = args2.SCALED_SIZE / self.dist_square
        
    def calc_dist(self, corners):
        dist_total = 0
        for i in range(args2.BORAD_HEIGHT):
            dist = cv2.norm(corners[i * args2.BORAD_WIDTH,:], corners[(i+1) * args2.BORAD_WIDTH-1,:], cv2.NORM_L2)
            dist_total += dist / (args2.BORAD_WIDTH - 1)
        self.dist_square = dist_total / args2.BORAD_HEIGHT

    def padding(self, img, width, height):
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
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value = (0,0,0))  

    def center_crop(self, img, width, height):
        H = img.shape[0]
        W = img.shape[1]
        top = (H - height) // 2
        bottom = (H - height) // 2 + height
        left = (W - width) // 2
        right = (W - width) // 2 + width
        return img[top:bottom, left:right]          
    
    def __call__(self, raw_frame):
        width = raw_frame.shape[1]
        height = raw_frame.shape[0]
        raw_frame = cv2.resize(raw_frame, (0,0), fx=self.scale_factor, fy=self.scale_factor)  # 图像缩放
        if self.scale_factor < 1:
            raw_frame = self.padding(raw_frame, width, height)
        else:                     
            raw_frame = self.center_crop(raw_frame, width, height)
        return raw_frame

class ExCalibrator():
    def __init__(self):
        self.src_corners_total = np.empty([0,1,2]) #임의 배열 생성
        self.dst_corners_total = np.empty([0,1,2])

    @staticmethod
    def get_args2():
        return args2

    def imgPreprocess(self, img, center, scale):
        if center:
            centerImg = CenterImage() #4
            img = centerImg(img)    #5
        if scale:
            ok, corners = self.get_corners(img, subpix = args2.SUBPIX_REGION_DST) #6
            if not ok:
                raise Exception("failed to find corners in destination image")
            scaleImg = ScaleImage(corners)
            img = scaleImg(img)
        cv2.imshow("Preprocessed Image", img)
        cv2.waitKey(0)
        return img
        
    def get_corners(self, img, subpix, draw=False):
        ok, corners = cv2.findChessboardCorners(img, (args2.BORAD_WIDTH, args2.BORAD_HEIGHT),
                      flags = cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok: 
            print(2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners = cv2.cornerSubPix(gray, corners, (subpix, subpix), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        if draw:
            print(3)
            cv2.drawChessboardCorners(img, (args2.BORAD_WIDTH, args2.BORAD_HEIGHT), corners, ok)

        cv2.namedWindow("corner View", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("corner View", img)
        return ok, corners
    
    def warp(self):
        src_warp = cv2.warpPerspective(self.src_img, self.homography, 
                                       (340, 376)) 
        return src_warp
        
    def __call__(self, src_img):
        ok, src_corners = self.get_corners(src_img, subpix = args2.SUBPIX_REGION_SRC, draw=True)
        if not ok:
            raise Exception("failed to find corners in source image")
        
        dst_corners = np.array([
        [[293, 124]], [[293, 134]], [[293, 144]], [[293, 154]], [[293, 164]], [[293, 174]], [[293, 184]], [[293, 194]], [[293, 204]], [[293, 214]], [[293, 224]], [[293, 234]], [[293, 244]], [[293, 254]],
        [[283, 124]], [[283, 134]], [[283, 144]], [[283, 154]], [[283, 164]], [[283, 174]], [[283, 184]], [[283, 194]], [[283, 204]], [[283, 214]], [[283, 224]], [[283, 234]], [[283, 244]], [[283, 254]],
        [[273, 124]], [[273, 134]], [[273, 144]], [[273, 154]], [[273, 164]], [[273, 174]], [[273, 184]], [[273, 194]], [[273, 204]], [[273, 214]], [[273, 224]], [[273, 234]], [[273, 244]], [[273, 254]],
        [[263, 124]], [[263, 134]], [[263, 144]], [[263, 154]], [[263, 164]], [[263, 174]], [[263, 184]], [[263, 194]], [[263, 204]], [[263, 214]], [[263, 224]], [[263, 234]], [[263, 244]], [[263, 254]],
        [[253, 124]], [[253, 134]], [[253, 144]], [[253, 154]], [[253, 164]], [[253, 174]], [[253, 184]], [[253, 194]], [[253, 204]], [[253, 214]], [[253, 224]], [[253, 234]], [[253, 244]], [[253, 254]],
        ])


        self.dst_corners_total = np.append(self.dst_corners_total, dst_corners, axis = 0)
        self.src_corners_total = np.append(self.src_corners_total, src_corners, axis = 0)
        print(self.dst_corners_total)
        self.homography, _ = cv2.findHomography(self.src_corners_total, self.dst_corners_total,method = cv2.RANSAC)
        self.src_img = src_img
        return self.homography    

def get_images(PATH, NAME):
    filePath = [os.path.join(PATH, x) for x in os.listdir(PATH) # os.path.join = path 값 붙여줌
                                                                # os.listdir(PATH) = 해당 Path 내 모든 파일과 디렉토리 리스트 반환
                if any(x.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
                # 그 중 이미지 파일 꺼내옴
               ]
    filenames = [filename for filename in filePath if NAME in filename] # 이미지 파일 중 해당 이름의 파일이 있다면 저장
    if len(filenames) == 0:
        raise Exception("from {} read images failed".format(PATH))
    return filenames


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

def color_balance(image): # 색 밸런싱하는 함수
    b, g, r = cv2.split(image)
    B = np.mean(b)
    G = np.mean(g)
    R = np.mean(r)
    K = (R + G + B) / 3
    Kb = K / B
    Kg = K / G
    Kr = K / R
    cv2.addWeighted(b, Kb, 0, 0, 0, b)
    cv2.addWeighted(g, Kg, 0, 0, 0, g)
    cv2.addWeighted(r, Kr, 0, 0, 0, r)
    return cv2.merge([b,g,r])

def luminance_balance(images): # 이미지의 HSV를 통일해주는 함수
    [front,back,left,right] = [cv2.cvtColor(image,cv2.COLOR_BGR2HSV)  
                               for image in images]
                               # -> RGB 색상 이미지를 H(Hue, 색조), S(Saturation, 채도), V(Value, 명도) HSV 이미지로 변형
    hf, sf, vf = cv2.split(front)   # 멀티 채널 Matrix를 여러 개의 싱글 채널 Matrix로 바꿔준다.
    hb, sb, vb = cv2.split(back)    # H,S,V 각각으로 분해된 값이다.
    hl, sl, vl = cv2.split(left)
    hr, sr, vr = cv2.split(right)
    V_f = np.mean(vf) # 주어진 배열의 산술 평균을 반환
    V_b = np.mean(vb)
    V_l = np.mean(vl)
    V_r = np.mean(vr)
    V_mean = (V_f + V_b + V_l +V_r) / 4
    vf = cv2.add(vf,(V_mean - V_f)) # V_mean - V_f = 전체 명도 평균 - FRONT 명도 평균 의 값과 front의 명도를 더함
    vb = cv2.add(vb,(V_mean - V_b)) # 이렇게 더해서 모든 bird Eye View 이미지의 명도 값을 평균적으로 변환
    vl = cv2.add(vl,(V_mean - V_l))
    vr = cv2.add(vr,(V_mean - V_r))
    front = cv2.merge([hf,sf,vf]) # 여러 개의 싱글 채널 Matrix를 멀티 채널 Matrix로 바꿔준다. split의 반대
    back = cv2.merge([hb,sb,vb])
    left = cv2.merge([hl,sl,vl])
    right = cv2.merge([hr,sr,vr])
    images = [front,back,left,right]
    images = [cv2.cvtColor(image,cv2.COLOR_HSV2BGR) for image in images]

    return images

class Mask:
    def __init__(self, name):
        self.mask = self.get_mask(name) #11-1
        
    def get_points(self, name): # Bird Eye View를 위한 point 값 (변환 좌표)
        if name == 'front':
            points = np.array([
                [0, 0],
                [BEV_WIDTH, 0], 
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2],
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2]
            ]).astype(np.int32)
        elif name == 'back':
            points = np.array([
                [0, BEV_HEIGHT],
                [BEV_WIDTH, BEV_HEIGHT],
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2], 
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2]
            ]).astype(np.int32)
        elif name == 'left':
            points = np.array([
                [0, 0],
                [0, BEV_HEIGHT], 
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2],
                [(BEV_WIDTH-CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2]
            ]).astype(np.int32)
        elif name == 'right':
            points = np.array([
                [BEV_WIDTH, 0],
                [BEV_WIDTH, BEV_HEIGHT], 
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT+CAR_HEIGHT)/2], 
                [(BEV_WIDTH+CAR_WIDTH)/2, (BEV_HEIGHT-CAR_HEIGHT)/2]
            ]).astype(np.int32)
        else:
            raise Exception("name should be front/back/left/right")
        return points
    
    def get_mask(self, name):
        mask = np.zeros((BEV_HEIGHT,BEV_WIDTH), dtype=np.uint8) # 마스크 생성, Bird Eye View 높이, 너비만큼 배열 생성
        points = self.get_points(name) # 12-1

        img = cv2.fillPoly(mask, [points], 255)
        cv2.namedWindow("raw_frame", flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("raw_frame", img)
        cv2.waitKey(0)

        return cv2.fillPoly(mask, [points], 255)
        # cv2.fillPoly = 채워진 다각형을 그립니다. pts에 다각형 배열 값을 여러 개 입력할 수도 있습니다. 255 = color
        # fillPoly()에 다각형 좌표 배열을 여러 개 적용한 경우 겹치는 부분이 사라집니다.
    
    def __call__(self, img):
        return cv2.bitwise_and(img, img, mask=self.mask) # mask 영역에서 서로 공통으로 겹치는 부분 출력

class BlendMask:
    def __init__(self,name):
        mf = self.get_mask('front') #11-2
        mb = self.get_mask('back')
        ml = self.get_mask('left')
        mr = self.get_mask('right')
        self.get_lines()    #12-2-1
        if name == 'front':
            mf = self.get_blend_mask(mf, ml, self.lineFL, self.lineLF) #12-2-2
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
                [0, BEV_HEIGHT - BEV_HEIGHT/.5],
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
        points = self.get_points(name)  # 12-2
        return cv2.fillPoly(mask, [points], 255)    # cv2.fillPoly = 채워진 다각형을 그립니다. pts에 다각형 배열 값을 여러 개 입력할 수도 있습니다. 255 = color
        # fillPoly()에 다각형 좌표 배열을 여러 개 적용한 경우 겹치는 부분이 사라집니다.
    
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

    #zip() 함수는 여러 개의 순회 가능한(iterable) 객체를 인자로 받고, 
    # 각 객체가 담고 있는 원소를 터플의 형태로 차례로 접근할 수 있는 반복자(iterator)를 반환합니다. 
    # 설명이 좀 어렵게 들릴 수도 있는데요. 간단한 예제를 보면 이해가 쉬우실 겁니다.
    #>>> numbers = [1, 2, 3]
    #>>> letters = ["A", "B", "C"]
    #>>> for pair in zip(numbers, letters):
    #...     print(pair)
    #...
    #(1, 'A')
    #(2, 'B')
    #(3, 'C')

        for y, x in zip(*indices):
            distA = cv2.pointPolygonTest(np.array(lineA), (x, y), True)
            distB = cv2.pointPolygonTest(np.array(lineB), (x, y), True)
            # 이미지에서 해당 Point가 Contour의 어디에 위치해 있는지 확인하는 함수이다.

            # 1. contour : Contour Points들을 인자로 받는다.
            # 2. pt : Contour에 테스트할 Point를 인자로 받는다.
            # 3. measureDist : boolean 데이터 타입. 
            maskA[y, x] = distA**2 / (distA**2 + distB**2 + 1e-6) * 255
        return maskA
    
    def __call__(self, img):
        return (img * self.weight).astype(np.uint8)    
    
class BevGenerator:
    def __init__(self, blend=args.BLEND_FLAG, balance=args.BALANCE_FLAG): #2
        self.init_args()
        self.blend = blend # Bird Eye View 이미지 혼합상태 판단 boolean
        self.balance = balance # Bird Eye View 이미지 균형상태 판단 boolean
        if not self.blend:
            self.masks = [Mask('front'), Mask('back'),  # 10-1
                          Mask('left'), Mask('right')]
        else:
            self.masks = [BlendMask('front'), BlendMask('back'), #10-2
                      BlendMask('left'), BlendMask('right')]

    @staticmethod
    def get_args():
        return args

    def init_args(self):
        global FRAME_WIDTH, FRAME_HEIGHT, BEV_WIDTH, BEV_HEIGHT
        global CAR_WIDTH, CAR_HEIGHT, FOCAL_SCALE, SIZE_SCALE
        FRAME_WIDTH = args.FRAME_WIDTH
        FRAME_HEIGHT = args.FRAME_HEIGHT
        BEV_WIDTH = args.BEV_WIDTH
        BEV_HEIGHT = args.BEV_HEIGHT
        CAR_WIDTH = args.CAR_WIDTH
        CAR_HEIGHT = args.CAR_HEIGHT
        FOCAL_SCALE = args.FOCAL_SCALE
        SIZE_SCALE = args.SIZE_SCALE

    def __call__(self, front, back, left, right, car = None):
        images = [front,back,left,right]

        if self.balance:
            images = luminance_balance(images)  #14
        images = [mask(img) #15
                  for img, mask in zip(images, self.masks)]
        surround = cv2.add(images[0],images[1]) #이미지를 합침
        surround = cv2.add(surround,images[2])
        surround = cv2.add(surround,images[3])
        if self.balance:
            surround = color_balance(surround) #16
        if car is not None:
            surround = cv2.add(surround,car)
        return surround

def runExCalib():
    print("Extrinsic Calibration ......")
    exCalib = ExCalibrator()                         

    src_raw = cv2.imread('C:\SSAFY\python\images\surroundView\img_src_right.png')
    homography = exCalib(src_raw)
    print("Homography Matrix is:")
    print(homography.tolist())

    src_warp = exCalib.warp()                        

    cv2.namedWindow("Source View", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Source View", src_warp)
    cv2.imwrite("C:\SSAFY\python\images\surroundView\\right.jpg", src_warp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def runBEV():
    print("Generating Surround BEV ......")
    front = cv2.imread('C:\SSAFY\python\images\surroundView\\front.jpg')
    back = cv2.imread('C:\SSAFY\python\images\surroundView\\back.jpg')
    left = cv2.imread('C:\SSAFY\python\images\surroundView\\left.jpg')
    right = cv2.imread('C:\SSAFY\python\images\surroundView\\right.jpg')

    args = BevGenerator.get_args()            
    args.CAR_WIDTH = 0
    args.CAR_HEIGHT = 0                 

    bev = BevGenerator(blend=False, balance=True)    
    surround = bev(front, back, left, right)         

    cv2.namedWindow('surround', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow('surround', surround)
    cv2.imwrite("C:\SSAFY\python\images\surroundView\\surround3.jpg", surround)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

def main():
    runExCalib()
    runBEV()

if __name__ == '__main__':
    main()