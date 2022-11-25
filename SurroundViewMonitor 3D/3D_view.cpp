// Include standard headers
#include <iostream>
#include <stdio.h>
//#include <stdlib.h>
#include <vector>
//#include <cmath>
#include <thread>
#include <time.h>

// Include GLEWc
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
GLFWwindow* window;

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

#include <common/shader.hpp>
#include <common/texture.hpp>
#include <common/controls.hpp>
#include <common/objloader.hpp>
//cv
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <typeinfo>  

using namespace cv;
using namespace std;
using std::thread;

// wand
//#include <MagickWand/MagickWand.h>
#include <Magick++.h>

//using namespace Magick;

/*
VideoCapture cap1(0);
VideoCapture cap2(1);
Mat img_frame1;
Mat img_frame2;
*/

Mat cart_image;
Mat bowlImg;
//size_t  w, h;

float k1 = 0.2;
float k2 = 0.8;
float k3 = 20;

float rows = 765;
float cols = 765;

bool bowlInit = false;

cv::Mat mapX(rows, cols, CV_32F);
cv::Mat mapY(rows, cols, CV_32F);

// 변수 선언(global)
int FRAME_WIDTH = 1280;
int FRAME_HEIGHT = 720;
int BEV_WIDTH = 1020;
int BEV_HEIGHT = 1128;
int CAR_WIDTH = 155;
int CAR_HEIGHT = 315;
float FOCAL_SCALE = 0.65;
float SIZE_SCALE = 1;

// 시간확인 디버깅용
clock_t start = clock();
clock_t endtime;


// 함수 선언 (내부 코드 선언은 main 함수 아래로)
Mat padding(Mat img, int width, int height);
void color_balance();
void read_frame(int index);

Mat* luminance_balance(Mat* images);
Mat weight[4];
Mat *images = new Mat[4];

Mat resizing_car;
cv::Mat car = cv::imread("porche.png");

// 0. 영상 획득 1
//cv::VideoCapture cap_0(0, CAP_DSHOW);
//cap_0.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
//cap_0.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
cv::VideoCapture cap_1(1, CAP_DSHOW);
cv::VideoCapture cap_2(2, CAP_DSHOW);
cv::VideoCapture cap_3(3, CAP_DSHOW);
cv::VideoCapture cap_4(4, CAP_DSHOW);
cv::Mat img0, img1, img2, img3, img4;
cv::Mat undistort_front, undistort_back, undistort_left, undistort_right;
cv::Mat top_front, top_back, top_left, top_right;

cv::Mat K = cv::Mat(3, 3, cv::DataType<double>::type);
cv::Mat D = cv::Mat(4, 1, cv::DataType<double>::type);
cv::Mat new_K = cv::Mat(3, 3, cv::DataType<double>::type);
cv::Mat map1, map2;

cv::Mat LEFT_K = cv::Mat(3, 3, cv::DataType<double>::type);
cv::Mat LEFT_D = cv::Mat(4, 1, cv::DataType<double>::type);
cv::Mat E = cv::Mat::eye(3, 3, cv::DataType<double>::type);
cv::Mat new_LEFT_K = cv::Mat(3, 3, cv::DataType<double>::type);
cv::Size size1 = { 1280, 720 };
cv::Mat left_map1, left_map2;

cv::Mat front_homography = cv::Mat(3, 3, cv::DataType<double>::type);
cv::Mat back_homography = cv::Mat(3, 3, cv::DataType<double>::type);
cv::Mat right_homography = cv::Mat(3, 3, cv::DataType<double>::type);
cv::Mat left_homography = cv::Mat(3, 3, cv::DataType<double>::type);
vector<thread> workers;
vector<thread> workers_surround;

Mat surround;


void read_surround();

void remap_warp(int i);

void getBowlImg(Mat &cameraImg, int mode);

class BlendMask {
public:
	Mat lineFL = cv::Mat(2, 2, cv::DataType<int>::type);
	Mat lineFR = cv::Mat(2, 2, cv::DataType<int>::type);
	Mat lineBL = cv::Mat(2, 2, cv::DataType<int>::type);
	Mat lineBR = cv::Mat(2, 2, cv::DataType<int>::type);
	Mat lineLF = cv::Mat(2, 2, cv::DataType<int>::type);
	Mat lineLB = cv::Mat(2, 2, cv::DataType<int>::type);
	Mat lineRF = cv::Mat(2, 2, cv::DataType<int>::type);
	Mat lineRB = cv::Mat(2, 2, cv::DataType<int>::type);

	Mat mf, mb, ml, mr, mask;

	BlendMask() {
	}

	BlendMask(string name) {
		mf = get_mask("front");
		mb = get_mask("back");
		ml = get_mask("left");
		mr = get_mask("right");
		get_lines();
		int index = -1;
		if (name == "front") {
			mf = get_blend_mask(mf, ml, lineFL, lineLF);
			mf = get_blend_mask(mf, mr, lineFR, lineRF);
			mask = mf;
			index = 0;
		}
		else if (name == "back") {
			mb = get_blend_mask(mb, ml, lineBL, lineLB);
			mb = get_blend_mask(mb, mr, lineBR, lineRB);
			mask = mb;
			index = 1;
		}
		else if (name == "left") {
			ml = get_blend_mask(ml, mf, lineLF, lineFL);
			ml = get_blend_mask(ml, mb, lineLB, lineBL);
			mask = ml;
			index = 2;
		}
		else if (name == "right") {
			mr = get_blend_mask(mr, mf, lineRF, lineFR);
			mr = get_blend_mask(mr, mb, lineRB, lineBR);
			mask = mr;
			index = 3;
		}

		int size1[] = { mask.rows ,mask.cols };
		weight[index] = Mat::zeros(2, size1, CV_32FC3);

		for (int i = 0; i < mask.rows; i++) {
			for (int j = 0; j < mask.cols; j++) {

				Vec3f& p1 = weight[index].at<Vec3f>(i, j);
				p1 = mask.at<Vec3f>(i, j) / 255.0;
			}
		}



	}


	Mat get_mask(string name) {
		Mat mask = Mat::zeros(BEV_HEIGHT, BEV_WIDTH, CV_32FC3);
		Point rook_points[1][6];
		if (name == "front") {
			rook_points[0][0] = Point(0, 0);
			rook_points[0][1] = Point(BEV_WIDTH, 0);
			rook_points[0][2] = Point(BEV_WIDTH, BEV_HEIGHT / 5);
			rook_points[0][3] = Point((BEV_WIDTH + CAR_WIDTH) / 2, (BEV_HEIGHT - CAR_HEIGHT) / 2);
			rook_points[0][4] = Point((BEV_WIDTH - CAR_WIDTH) / 2, (BEV_HEIGHT - CAR_HEIGHT) / 2);
			rook_points[0][5] = Point(0, BEV_HEIGHT / 5);
		}
		else if (name == "back") {
			rook_points[0][0] = Point(0, BEV_HEIGHT);
			rook_points[0][1] = Point(BEV_WIDTH, BEV_HEIGHT);
			rook_points[0][2] = Point(BEV_WIDTH, BEV_HEIGHT - BEV_HEIGHT / 5);
			rook_points[0][3] = Point((BEV_WIDTH + CAR_WIDTH) / 2, (BEV_HEIGHT + CAR_HEIGHT) / 2);
			rook_points[0][4] = Point((BEV_WIDTH - CAR_WIDTH) / 2, (BEV_HEIGHT + CAR_HEIGHT) / 2);
			rook_points[0][5] = Point(0, BEV_HEIGHT - BEV_HEIGHT / 5);
		}
		else if (name == "left") {
			rook_points[0][0] = Point(0, 0);
			rook_points[0][1] = Point(0, BEV_HEIGHT);
			rook_points[0][2] = Point(BEV_WIDTH / 5, BEV_HEIGHT);
			rook_points[0][3] = Point((BEV_WIDTH - CAR_WIDTH) / 2, (BEV_HEIGHT + CAR_HEIGHT) / 2);
			rook_points[0][4] = Point((BEV_WIDTH - CAR_WIDTH) / 2, (BEV_HEIGHT - CAR_HEIGHT) / 2);
			rook_points[0][5] = Point(BEV_WIDTH / 5, 0);
		}
		else if (name == "right") {
			rook_points[0][0] = Point(BEV_WIDTH, 0);
			rook_points[0][1] = Point(BEV_WIDTH, BEV_HEIGHT);
			rook_points[0][2] = Point(BEV_WIDTH - BEV_WIDTH / 5, BEV_HEIGHT);
			rook_points[0][3] = Point((BEV_WIDTH + CAR_WIDTH) / 2, (BEV_HEIGHT + CAR_HEIGHT) / 2);
			rook_points[0][4] = Point((BEV_WIDTH + CAR_WIDTH) / 2, (BEV_HEIGHT - CAR_HEIGHT) / 2);
			rook_points[0][5] = Point(BEV_WIDTH - BEV_WIDTH / 5, 0);
		}


		const Point* ppt[1] = { rook_points[0] };

		int npt[] = { 6 };
		fillPoly(mask, ppt, npt, 1, Scalar(255, 255, 255));

		return mask;
	}

	void get_lines() {

		lineFL.at<int>(0, 0) = 0;
		lineFL.at<int>(0, 1) = BEV_HEIGHT / 5;
		lineFL.at<int>(1, 0) = (BEV_WIDTH - CAR_WIDTH) / 2;
		lineFL.at<int>(1, 1) = (BEV_HEIGHT - CAR_HEIGHT) / 2;

		lineFR.at<int>(0, 0) = BEV_WIDTH;
		lineFR.at<int>(0, 1) = BEV_HEIGHT / 5;
		lineFR.at<int>(1, 0) = (BEV_WIDTH + CAR_WIDTH) / 2;
		lineFR.at<int>(1, 1) = (BEV_HEIGHT - CAR_HEIGHT) / 2;

		lineBL.at<int>(0, 0) = 0;
		lineBL.at<int>(0, 1) = BEV_HEIGHT - BEV_HEIGHT / 5;
		lineBL.at<int>(1, 0) = (BEV_WIDTH - CAR_WIDTH) / 2;
		lineBL.at<int>(1, 1) = (BEV_HEIGHT + CAR_HEIGHT) / 2;

		lineBR.at<int>(0, 0) = BEV_WIDTH;
		lineBR.at<int>(0, 1) = BEV_HEIGHT - BEV_HEIGHT / 5;
		lineBR.at<int>(1, 0) = (BEV_WIDTH + CAR_WIDTH) / 2;
		lineBR.at<int>(1, 1) = (BEV_HEIGHT + CAR_HEIGHT) / 2;

		lineLF.at<int>(0, 0) = BEV_WIDTH / 5;
		lineLF.at<int>(0, 1) = 0;
		lineLF.at<int>(1, 0) = (BEV_WIDTH - CAR_WIDTH) / 2;
		lineLF.at<int>(1, 1) = (BEV_HEIGHT - CAR_HEIGHT) / 2;

		lineLB.at<int>(0, 0) = BEV_WIDTH / 5;
		lineLB.at<int>(0, 1) = BEV_HEIGHT;
		lineLB.at<int>(1, 0) = (BEV_WIDTH - CAR_WIDTH) / 2;
		lineLB.at<int>(1, 1) = (BEV_HEIGHT + CAR_HEIGHT) / 2;

		lineRF.at<int>(0, 0) = BEV_WIDTH - BEV_WIDTH / 5;
		lineRF.at<int>(0, 1) = 0;
		lineRF.at<int>(1, 0) = (BEV_WIDTH + CAR_WIDTH) / 2;
		lineRF.at<int>(1, 1) = (BEV_HEIGHT - CAR_HEIGHT) / 2;
		lineRB.at<int>(0, 0) = BEV_WIDTH - BEV_WIDTH / 5;
		lineRB.at<int>(0, 1) = BEV_HEIGHT;
		lineRB.at<int>(1, 0) = (BEV_WIDTH + CAR_WIDTH) / 2;
		lineRB.at<int>(1, 1) = (BEV_HEIGHT + CAR_HEIGHT) / 2;
	}

	Mat get_blend_mask(Mat maskA, Mat maskB, Mat lineA, Mat lineB) {
		Mat overlap;
		cv::bitwise_and(maskA, maskB, overlap);
		waitKey(1);
		Point *indices = new Point[overlap.rows*overlap.cols];

		int idx = 0;
		Vec3f tmp;

		for (int x = 0; x < overlap.rows; x++) {
			for (int y = 0; y < overlap.cols; y++) {
				tmp = overlap.at<Vec3f>(Point(y, x));

				if (tmp != Vec3f()) {
					indices[idx++] = Point(y, x);

				}
			}
		}
		for (int i = 0; i < idx; i++) {
			float distA = cv::pointPolygonTest(lineA, Point(Point(indices[i]).x, Point(indices[i]).y), true);
			float distB = cv::pointPolygonTest(lineB, Point(Point(indices[i]).x, Point(indices[i]).y), true);

			maskA.at<Vec3f>(Point(indices[i])) = Vec3f(pow(distA, 2) / (pow(distA, 2) + pow(distB, 2) + 1e-6) * 255, pow(distA, 2) / (pow(distA, 2) + pow(distB, 2) + 1e-6) * 255, pow(distA, 2) / (pow(distA, 2) + pow(distB, 2) + 1e-6) * 255);

		}


		delete[] indices;

		return maskA;
	}

	/*Mat returning(Mat img) {
		img.convertTo(img,CV_32FC3);
		Mat a;
		cv::multiply(img, weight, a);

		return a;
	}*/
};
void multiply_weight(int i) {
	images[i].convertTo(images[i], CV_32FC3);
	//Mat a;
	//덮어쓰기 되는지 확인 필요
	cv::multiply(images[i], weight[i], images[i]);
}
class BevGenerator {
public:
	BlendMask masks[4];

	clock_t start;
	clock_t end1;
	clock_t end2;
	clock_t end3;

	BevGenerator() {
		/*	masks[0] = BlendMask("front");
			masks[1] = BlendMask("back");
			masks[2] = BlendMask("left");
			masks[3] = BlendMask("right");*/
		BlendMask("front");
		BlendMask("back");
		BlendMask("left");
		BlendMask("right");


	}
	thread t[4];
	Mat returning(Mat front, Mat back, Mat left, Mat right, Mat car) {

		images[0] = front;
		images[1] = back;
		images[2] = left;
		images[3] = right;

		//start = clock();

		images = luminance_balance(images);

		//end1 = clock();

		//for (int i = 0; i < 4; i++) {
		//	//images[i] = masks[i].returning(images[i]);
		//	multiply_weight(i);
		//}

		////쓰레드
		for (int i = 0; i < 4; i++) {
			//images[i] = masks[i].returning(images[i]);
			t[i] = thread(multiply_weight, i);
		}
		for (int i = 0; i < 4; i++)
			t[i].join();
		//imshow("mask" + to_string(0), images[0]);
		//imshow("mask" + to_string(1) , images[1]);
		//imshow("mask" + to_string(2) , images[2]);
		//imshow("mask" + to_string(3) , images[3]);
		//imwrite("mask" + to_string(0)+".jpg", images[0]);
		//imwrite("mask" + to_string(1) + ".jpg", images[1]);
		//imwrite("mask" + to_string(2) + ".jpg", images[2]);
		//imwrite("mask" + to_string(3) + ".jpg", images[3]);
		//waitKey(1);

		//end2 = clock();

		cv::add(images[0], images[1], surround);
		cv::add(surround, images[2], surround);
		cv::add(surround, images[3], surround);

		thread t1 = thread(color_balance);
		t1.join();

		cv::add(surround, car, surround);

		//end3 = clock();

		//cout << "time1 : " << (double)end1 - start << endl;
		//cout << "time2 : " << (double)end2 - end1 << endl;
		//cout << "time3 : " << (double)end3 - end2 << endl;


		return surround;
	}

};

BevGenerator bev = BevGenerator();
Mat surround1;

int main(void)
{
	start = clock();
	// C:/Users/multicampus/Desktop/ssafy/self_project/gitlab/SurroundViewMonitor
	Magick::InitializeMagick("");

	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		getchar();
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Open a window and create its OpenGL context
	window = glfwCreateWindow(1024, 768, "Tutorial 07 - Model Loading", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		getchar();
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return -1;
	}

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	// Hide the mouse and enable unlimited mouvement
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// Set the mouse at the center of the screen
	glfwPollEvents();
	glfwSetCursorPos(window, 1024 / 2, 768 / 2);

	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);

	// Cull triangles which normal is not towards the camera
	glEnable(GL_CULL_FACE);
	// 안쪽면만 그림
	glCullFace(GL_FRONT);

	// 호모그래피 배열 초기화

	front_homography.at<double>(0, 0) = 3.8290818190655296;
	front_homography.at<double>(0, 1) = 5.66226108412413;
	front_homography.at<double>(0, 2) = -1961.3463578215583;
	front_homography.at<double>(1, 0) = 0.04676494495778877;
	front_homography.at<double>(1, 1) = 7.593794538002965;
	front_homography.at<double>(1, 2) = -1398.2503661266026;
	front_homography.at<double>(2, 0) = 9.936840064368021e-05;
	front_homography.at<double>(2, 1) = 0.011048826673722773;
	front_homography.at<double>(2, 2) = 1.0;


	back_homography.at<double>(0, 0) = -3.6014363756395578;
	back_homography.at<double>(0, 1) = 4.36751919471232;
	back_homography.at<double>(0, 2) = 2793.845376423345;
	back_homography.at<double>(1, 0) = -0.16343764199990043;
	back_homography.at<double>(1, 1) = 3.070397421396325;
	back_homography.at<double>(1, 2) = 2357.357976286147;
	back_homography.at<double>(2, 0) = -0.00020012270934899256;
	back_homography.at<double>(2, 1) = 0.008545448669492679;
	back_homography.at<double>(2, 2) = 1.0;


	right_homography.at<double>(0, 0) = -0.08221005850690113;
	right_homography.at<double>(0, 1) = 1.6235034060147169;
	right_homography.at<double>(0, 2) = 2032.459375714023;
	right_homography.at<double>(1, 0) = 2.945881257892563;
	right_homography.at<double>(1, 1) = 3.582400762214573;
	right_homography.at<double>(1, 2) = -1439.2015581787357;
	right_homography.at<double>(2, 0) = -0.00011329233289173604;
	right_homography.at<double>(2, 1) = 0.0068701395422870026;
	right_homography.at<double>(2, 2) = 0.9999999999999999;


	left_homography.at<double>(0, 0) = 0.1022385737675907;
	left_homography.at<double>(0, 1) = 5.770995782571725;
	left_homography.at<double>(0, 2) = -1228.0525962849354;
	left_homography.at<double>(1, 0) = -3.1690352115113147;
	left_homography.at<double>(1, 1) = 3.516499167979982;
	left_homography.at<double>(1, 2) = 2659.2521230613215;
	left_homography.at<double>(2, 0) = 0.0002888891947907289;
	left_homography.at<double>(2, 1) = 0.006800266459274919;
	left_homography.at<double>(2, 2) = 1.0;

	// left fish-eye 왜곡 RectifyMap


	LEFT_K.at<double>(0, 0) = 486.43710381577273;
	LEFT_K.at<double>(0, 1) = 0.0;
	LEFT_K.at<double>(0, 2) = 643.0021325671074;
	LEFT_K.at<double>(1, 0) = 0.0;
	LEFT_K.at<double>(1, 1) = 485.584911786959;
	LEFT_K.at<double>(1, 2) = 402.9808925210084;
	LEFT_K.at<double>(2, 0) = 0.0;
	LEFT_K.at<double>(2, 1) = 0.0;
	LEFT_K.at<double>(2, 2) = 1.0;

	LEFT_D.at<double>(0, 0) = -0.06338733272909226;
	LEFT_D.at<double>(1, 0) = -0.007861033496168955;
	LEFT_D.at<double>(2, 0) = 0.005073683389947028;
	LEFT_D.at<double>(3, 0) = -0.0010639404289377306;

	new_LEFT_K.at<double>(0, 0) = 486.43710381577273 / 1.5;
	new_LEFT_K.at<double>(0, 1) = 0.0;
	new_LEFT_K.at<double>(0, 2) = 643.0021325671074;
	new_LEFT_K.at<double>(1, 0) = 0.0;
	new_LEFT_K.at<double>(1, 1) = 485.584911786959 / 1.5;
	new_LEFT_K.at<double>(1, 2) = 402.9808925210084;
	new_LEFT_K.at<double>(2, 0) = 0.0;
	new_LEFT_K.at<double>(2, 1) = 0.0;
	new_LEFT_K.at<double>(2, 2) = 1.0;

	cv::fisheye::initUndistortRectifyMap(LEFT_K, LEFT_D, E, new_LEFT_K, size1, CV_16SC2, left_map1, left_map2);


	// other fish-eye 왜곡 RectifyMap


	K.at<double>(0, 0) = 455.8515274977241;
	K.at<double>(0, 1) = 0.0;
	K.at<double>(0, 2) = 655.7621645964248;
	K.at<double>(1, 0) = 0.0;
	K.at<double>(1, 1) = 455.08604281075947;
	K.at<double>(1, 2) = 367.3548823943176;
	K.at<double>(2, 0) = 0.0;
	K.at<double>(2, 1) = 0.0;
	K.at<double>(2, 2) = 1.0;

	D.at<double>(0, 0) = -0.02077978156022359;
	D.at<double>(1, 0) = -0.02434621475644252;
	D.at<double>(2, 0) = 0.009725498728069807;
	D.at<double>(3, 0) = -0.0018108318059442028;

	new_K.at<double>(0, 0) = 455.8515274977241 / 1.5;
	new_K.at<double>(0, 1) = 0.0;
	new_K.at<double>(0, 2) = 655.7621645964248;
	new_K.at<double>(1, 0) = 0.0;
	new_K.at<double>(1, 1) = 455.08604281075947 / 1.5;
	new_K.at<double>(1, 2) = 367.3548823943176;
	new_K.at<double>(2, 0) = 0.0;
	new_K.at<double>(2, 1) = 0.0;
	new_K.at<double>(2, 2) = 1.0;

	cv::fisheye::initUndistortRectifyMap(K, D, E, new_K, size1, CV_16SC2, map1, map2);

	// 차량 생성

	cv::resize(car, resizing_car, cv::Size(320, 450));
	car = padding(resizing_car, BEV_WIDTH, BEV_HEIGHT);
	car.convertTo(car, CV_32FC3);

	// 영상 사이즈 선언
	cap_1.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	cap_1.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
	cap_2.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	cap_2.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
	cap_3.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	cap_3.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
	cap_4.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	cap_4.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

	// openGL
	GLuint VertexArrayID[2];
	glGenVertexArrays(2, VertexArrayID);

	// Create and compile our GLSL program from the shaders
	GLuint programID = LoadShaders("TransformVertexShader.vertexshader", "TextureFragmentShader.fragmentshader");
	// Get a handle for our "MVP" uniform
	GLuint MatrixID = glGetUniformLocation(programID, "MVP");
	// Load the texture
	GLuint Texture = loadDDS("./resource/uvmap.DDS");
	// Get a handle for our "myTextureSampler" uniform
	GLuint TextureID = glGetUniformLocation(programID, "myTextureSampler");

	// 카트 유니폼
	GLuint Cart_MatrixID = glGetUniformLocation(programID, "Cart_MVP");
	GLuint Cart_Texture = loadDDS("./resource/sample.DDS");
	GLuint Cart_TextureID = glGetUniformLocation(programID, "Cart_myTextureSampler");


	// Read our .obj file
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec2> uvs;
	std::vector<glm::vec3> normals; // Won't be used at the moment.
	bool res = loadOBJ("./resource/bowl.obj", vertices, uvs, normals);

	// 카트 오브젝트
	std::vector<glm::vec3> cart_vertices;
	std::vector<glm::vec2> cart_uvs;
	std::vector<glm::vec3> cart_normals;
	bool res2 = loadOBJ("./resource/cart.obj", cart_vertices, cart_uvs, cart_normals);

	// Load it into a VBO
	GLuint vertexbuffer[2];

	// 새로운 버퍼 생성 glGenBuffers(버퍼 개수, 이름 저장 공간)
	glGenBuffers(2, vertexbuffer);
	// 버퍼에 타겟을 할당(타겟, 버퍼)
	// GL_ARRAY_BUFFER -> 정점에 대한 데이터를 생성한 후 버퍼에 넣는다
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer[0]);
	// 실제 버퍼에 데이터 넣기 (타켓, 사이즈, 실제넣을 데이터 주소값, usage)
	// GL_STATIC_DRAW -> 데이터가 저장되면 변경되지 않음
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer[1]);
	glBufferData(GL_ARRAY_BUFFER, cart_vertices.size() * sizeof(glm::vec3), &cart_vertices[0], GL_STATIC_DRAW);


	GLuint uvbuffer;
	glGenBuffers(1, &uvbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(glm::vec2), &uvs[0], GL_STATIC_DRAW);

	clock_t start;
	clock_t end1;
	clock_t end2;
	clock_t end3;
	thread t1;
	thread t[4];
	thread t2[4];

	for (int i = 0; i < 4; i++)
		t[i] = thread(read_frame, i);
	for (int i = 0; i < 4; i++)
		t[i].join();
	for (int i = 0; i < 4; i++)
		t2[i] = thread(remap_warp, i);
	//cout << "2 : " << workers.size() << endl;
	for (int i = 0; i < 4; i++)
		t2[i].join();
	for (int i = 0; i < 4; i++)
		t[i] = thread(read_frame, i);

	t1 = thread(read_surround);

	// bowl thread
	thread bowl_t[4];

	do {
		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Use our shader
		glUseProgram(programID);

		// Compute the MVP matrix from keyboard and mouse input
		computeMatricesFromInputs();
		glm::mat4 ProjectionMatrix = getProjectionMatrix();
		glm::mat4 ViewMatrix = getViewMatrix();
		glm::mat4 ModelMatrix = glm::mat4(1.0);
		glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

		// Send our transformation to the currently bound shader, 
		// in the "MVP" uniform
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

		// Bind our texture in Texture Unit 0
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, Texture);
		// Set our "myTextureSampler" sampler to use Texture Unit 0
		glUniform1i(TextureID, 0);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		
		// bowl view
		imshow("getkey", img1);
		if (cv::waitKey(10) == 13) {	// cv::waitKey(10) == 13
			start = clock();
			for (int i = 0; i < 4; i++)
				t[i].join();

			//cout << "1 : "<<workers.size() << endl;
			for (int i = 0; i < 4; i++)
				t2[i] = thread(remap_warp, i);
			//cout << "2 : " << workers.size() << endl;
			for (int i = 0; i < 4; i++)
				t2[i].join();
			for (int i = 0; i < 4; i++)
				t[i] = thread(read_frame, i);
			//cout << "3 : " << workers.size() << endl;
			//end1 = clock();
			// 5. 5개 합성
			t1.join();
			t1 = thread(read_surround);

			cv::imshow("surround1212", surround1);
			/*cv::imwrite("surround22.jpg", surround);

			// 테스트용 (원하는 data를 2번째 인자에 입력하세요)
			/*cv::imshow("1", undistort_front);
			cv::imshow("2", undistort_back);
			cv::imshow("3", undistort_left);
			cv::imshow("4", undistort_back);*/

			//cout << "time1 : "<<(double)(end1 - start) << endl;
			//cout << "time2 : " << (double)(end2 - end1) << endl;


			//end2 = clock();
			//cout << "time2 : " << (double)(end2 - start) << endl;
			//cv::waitKey(1);

			//endtime = clock();
			//cout << "start : " << double(endtime - start) / CLOCKS_PER_SEC << endl;

			cv::Mat img_list[4] = { undistort_back, undistort_left, undistort_right, undistort_front };

			for (int i = 0; i < 4; i++) {
				bowl_t[i] = thread(getBowlImg, std::ref(img_list[i]), i);
			}
			for (int i = 0; i < 4; i++) {
				bowl_t[i].join();
			}

			endtime = clock();
			cout << "end : " << double(endtime - start) / CLOCKS_PER_SEC << endl;

			Mat bowlTemp;
			cv::resize(bowlImg, bowlTemp, cv::Size(1000, 1000));
			imshow("bowlImg", bowlTemp);
		}


		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bowlImg.cols, bowlImg.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, bowlImg.ptr());


		glBindVertexArray(VertexArrayID[0]);
		// 1rst attribute buffer : vertices
		// 위에 생성한 index 버퍼를 활성화
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer[0]);
		// 저장한 데이터의 속성 정보 지정
		glVertexAttribPointer(
			0,                  // attribute
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

		// Draw the triangle !
		glBindVertexArray(VertexArrayID[0]);
		glDrawArrays(GL_TRIANGLES, 0, vertices.size());


		// 2nd attribute buffer : UVs
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
		glVertexAttribPointer(
			1,                                // attribute
			2,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);

		glBindTexture(GL_TEXTURE_2D, Cart_Texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, cart_image.cols, cart_image.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, cart_image.data);

		glBindVertexArray(VertexArrayID[1]);
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer[1]);
		glVertexAttribPointer(
			0,                  // attribute
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);


		glBindVertexArray(VertexArrayID[1]);
		glDrawArrays(GL_TRIANGLES, 0, cart_vertices.size());

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

		//endtime = clock();
		//cout << "gl end : " << double(endtime - start) / CLOCKS_PER_SEC << endl;

	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
		glfwWindowShouldClose(window) == 0);

	// Cleanup VBO and shader
	glDeleteBuffers(1, vertexbuffer);
	glDeleteBuffers(1, &uvbuffer);
	glDeleteProgram(programID);
	glDeleteTextures(1, &Texture);
	glDeleteVertexArrays(1, VertexArrayID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();

	return 0;
}

void read_surround() {
	surround1 = bev.returning(top_front, top_back, top_left, top_right, car);
	cv::resize(surround1, surround1, cv::Size(670, 752));
	surround1.convertTo(surround1, CV_8UC3);
}
void read_frame(int index) {
	switch (index) {
	case 0:
		//cap_2 >> img1;
		img1 = imread("front2.png");
		break;
	case 1:
		//cap_4 >> img2;
		img2 = imread("back2.png");
		break;
	case 2:
		//cap_1 >> img3;
		img3 = imread("left2.png");
		break;
	case 3:
		//cap_3 >> img4;
		img4 = imread("right2.png");
		break;

	}
}

void remap_warp(int index) {

	switch (index) {
	case 0:
		// 2. 왜곡 보정 (output : undistort_front, undistort_back, undistort_left, undistort_right)
		cv::remap(img1, undistort_front, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
		// 3. 탑뷰 전환 (output : top_front, top_back, top_left, top_right)
		cv::warpPerspective(undistort_front, top_front, front_homography, cv::Size(1020, 1128));
		break;
	case 1:
		// 2. 왜곡 보정 (output : undistort_front, undistort_back, undistort_left, undistort_right)
		cv::remap(img2, undistort_back, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
		// 3. 탑뷰 전환 (output : top_front, top_back, top_left, top_right)
		cv::warpPerspective(undistort_back, top_back, back_homography, cv::Size(1020, 1128));
		break;
	case 2:
		cv::remap(img3, undistort_left, left_map1, left_map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
		cv::warpPerspective(undistort_left, top_left, left_homography, cv::Size(1020, 1128));
		break;
	case 3:
		cv::remap(img4, undistort_right, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
		cv::warpPerspective(undistort_right, top_right, right_homography, cv::Size(1020, 1128));
		break;

	}
}

Mat padding(Mat img, int width, int height) {
	Mat dst_constant;
	int H = 450;
	int W = 320;

	int top = (height - H) / 2;
	int bottom = (height - H) / 2;
	if (top + bottom + H < height) { bottom += 1; };

	int left = (width - W) / 2;
	int right = (width - W) / 2;
	if (left + right + W < width) { right += 1; };

	cv::copyMakeBorder(img, dst_constant, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

	return dst_constant;
}

void color_balance() {
	Mat bgr[3];
	cv::split(surround, bgr);

	Scalar scalar = mean(surround);
	double B = scalar[0];
	double G = scalar[1];
	double R = scalar[2];

	double K = (R + G + B) / 3;
	double Kb = K / B;
	double Kg = K / G;
	double Kr = K / R;
	cv::addWeighted(bgr[0], Kb, 0, 0, 0, bgr[0]);
	cv::addWeighted(bgr[1], Kg, 0, 0, 0, bgr[1]);
	cv::addWeighted(bgr[2], Kr, 0, 0, 0, bgr[2]);

	cv::merge(bgr, 3, surround);

}

Mat front1, back1, left1, right1;
Mat split_front[3], split_back[3], split_left[3], split_right[3];
Scalar V_f, V_b, V_l, V_r, V_mean, vf, vb, vl, vr;
Mat Front1, Back1, Left1, Right1;
Mat *answer = new Mat[4];

void luminance_split(int index) {
	switch (index) {
	case 0:
		cv::cvtColor(images[0], front1, COLOR_BGR2HSV);
		cv::split(front1, split_front);
		V_f = cv::mean(split_front[2]);
		break;
	case 1:
		cv::cvtColor(images[1], back1, COLOR_BGR2HSV);
		cv::split(back1, split_back);
		V_b = cv::mean(split_back[2]);
		break;
	case 2:
		cv::cvtColor(images[2], left1, COLOR_BGR2HSV);
		cv::split(left1, split_left);
		V_l = cv::mean(split_left[2]);
		break;
	case 3:
		cv::cvtColor(images[3], right1, COLOR_BGR2HSV);
		cv::split(right1, split_right);
		V_r = cv::mean(split_right[2]);
		break;
	}
}

void luminance_marge(int index) {
	switch (index) {
	case 0:
		cv::add(vf, (V_mean - V_f), vf);
		cv::merge(split_front, 3, Front1);
		cv::cvtColor(Front1, answer[0], COLOR_HSV2BGR);
		break;
	case 1:
		cv::add(vb, (V_mean - V_b), vb);
		cv::merge(split_back, 3, Back1);
		cv::cvtColor(Back1, answer[1], COLOR_HSV2BGR);
		break;
	case 2:
		cv::add(vl, (V_mean - V_l), vl);
		cv::merge(split_left, 3, Left1);
		cv::cvtColor(Left1, answer[2], COLOR_HSV2BGR);
		break;
	case 3:
		cv::add(vr, (V_mean - V_r), vr);
		cv::merge(split_right, 3, Right1);
		cv::cvtColor(Right1, answer[3], COLOR_HSV2BGR);
		break;
	}
}

Mat* luminance_balance(Mat* images) {
	thread t[4];

	for (int i = 0; i < 4; i++)
		t[i] = thread(luminance_split, i);
	//cout << "2 : " << workers.size() << endl;
	for (int i = 0; i < 4; i++)
		t[i].join();

	V_mean = (V_f + V_b + V_l + V_r) / 4;

	for (int i = 0; i < 4; i++)
		t[i] = thread(luminance_marge, i);
	//cout << "2 : " << workers.size() << endl;
	for (int i = 0; i < 4; i++)
		t[i].join();

	return answer;
}


void getBowlImg(Mat &cameraImg, int mode) {
	endtime = clock();
	cout << "getBowlImg start : " << double(endtime - start) / CLOCKS_PER_SEC << endl;

	double degree;	// 이미지를 회전시킬 정도 
	// Mat 이미지 Image로 변환
	Magick::Image img(cameraImg.cols, cameraImg.rows, "BGR", Magick::CharPixel, (char *)cameraImg.data);
	endtime = clock();
	cout << "mat to image transform : " << double(endtime - start) / CLOCKS_PER_SEC << endl;

	// 파이썬의 "arc" 그냥 왜곡을 하는 방법 중 하나
	MagickCore::DistortMethod method = Magick::ArcDistortion;

	if (mode == 0){
		degree = 315;
	}else if (mode == 1) {
		degree = 45 + 5;
	}else if (mode == 2) {
		degree = 225 - 5;
	}else{
		degree = 135;
	}

	// 아래의 파라미터에 따라서 왜곡 진행
	double listOfArguments[2] = { 90, degree };
	img.distort(method, 2, listOfArguments);

	endtime = clock();
	cout << "distort : " << double(endtime - start) / CLOCKS_PER_SEC << endl;

	// bowlImg 초기화
	if (!bowlInit) {
		// 이미지 크기
		//w = img.columns(), h = img.rows();
		// zero함수를 사용하면 Scalar(0)으로 고정되어 3차원 배열이 불가능하다.
		//bowlImg = Mat(img.columns() * 1.30, img.rows() * 1.30, CV_8UC3, Scalar(0, 0, 0));
		
		// thread로 인하여 몇 번째 img로 초기화 될지 몰라, img1 기준의 가로, 세로 고정 비로 고정
		bowlImg = Mat(1176 * 1.30, 1176 * 1.30, CV_8UC3, Scalar(0, 0, 0));
		bowlInit = true;
	}

	// bowlView 변수에 왜곡을 넣은 이미지 붙여넣기

	// x가 높이, y가 가로 길이
	// 순서대로 이미지가 차지하는 폭을 의미한다.
	// 전방이미지는 가로 0 ~ top_center, 세로 0 ~ left_center,
	// 우측이미지는 가로 top_center ~ bowlImg.col, 세로 0 ~ right_center,
	// 좌측이미지는 가로 0 ~ bottom_center, 세로 left_center ~  bowlImg.col,
	// 후측이미지는 가로 bottom_center ~ bowlImg.col, 세로 right_center ~  bowlImg.col 을 차지한다.
	int left_center = bowlImg.rows * 0.5;
	int right_center = bowlImg.rows * 0.52;
	int top_center = bowlImg.cols * 0.5;
	int bottom_center = bowlImg.cols * 0.52;

	// 왜곡을 만든 Image 를 다시 Mat으로 변환
	Mat temp = Mat(img.rows(), img.columns(), CV_8UC3, Scalar(255, 255, 255));
	img.write(0, 0, img.columns(), img.rows(), "BGR", Magick::CharPixel, temp.data);

	//endtime = clock();
	//cout << "image to mat : " << double(endtime - start) / CLOCKS_PER_SEC << endl;

	if (mode == 0) {
		cout << "height : " << img.rows() << ", width : " << img.columns() << endl;
		//imshow("front", temp);
		int img_x_move = 70;
		int img_y_move = 70;
		float resize = 1;

		Mat imageROI = bowlImg(Rect(0, 0, top_center, left_center));
		Mat resized;
		cv::resize(temp, resized, Size(temp.cols * resize, temp.rows * resize));
		resized = resized(Rect(img_y_move, img_x_move, top_center, left_center));
		Mat gray;
		cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
		
		resized.copyTo(imageROI, gray);
	}
	else if (mode == 1) {
		//imshow("right", temp);
		int img_x_move = 81;
		int img_y_move = 285;
		float resize = 1;

		int height = right_center;
		int width = bowlImg.cols - top_center;

		Mat imageROI = bowlImg(Rect(top_center, 0, width, height));
		Mat resized;
		cv::resize(temp, resized, Size(temp.cols * resize, temp.rows * resize));
		resized = resized(Rect(img_y_move, img_x_move, width, height));
		Mat gray;
		cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);

		resized.copyTo(imageROI, gray);
	}
	else if (mode == 2) {
		//imshow("left", temp);
		int img_x_move = 288;
		int img_y_move = 53;
		float resize = 1;

		int height = bowlImg.rows - left_center;
		int width = bottom_center;

		Mat imageROI = bowlImg(Rect(0, left_center, width, height));
		Mat resized;
		cv::resize(temp, resized, Size(temp.cols * resize, temp.rows * resize));
		resized = resized(Rect(img_y_move, img_x_move, width, height));
		Mat gray;
		cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);

		resized.copyTo(imageROI, gray);
	}
	else if(mode == 3) {
		//imshow("back", temp);
		int img_x_move = 225;
		int img_y_move = 215;
		float resize = 0.84;

		int height = bowlImg.rows - right_center;
		int width = bowlImg.cols - bottom_center;

		Mat imageROI = bowlImg(Rect(bottom_center, right_center, width, height));
		Mat resized;
		cv::resize(temp, resized, Size(temp.cols * resize, temp.rows * resize));
		resized = resized(Rect(img_y_move, img_x_move, width, height));
		Mat gray;
		cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);

		resized.copyTo(imageROI, gray);
	}
}