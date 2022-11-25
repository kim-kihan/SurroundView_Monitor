// Include GLFW
#include <GLFW/glfw3.h>
extern GLFWwindow* window; // The "extern" keyword here is to access the variable "window" declared in tutorialXXX.cpp. This is a hack to keep the tutorials simple. Please avoid this.

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

#include "controls.hpp"

glm::mat4 ViewMatrix;
glm::mat4 ProjectionMatrix;

glm::mat4 getViewMatrix() {
	return ViewMatrix;
}
glm::mat4 getProjectionMatrix() {
	return ProjectionMatrix;
}


// 시점 시작 위치
glm::vec3 position = glm::vec3(0, 1, 3);
// Initial horizontal angle : toward -Z
float horizontalAngle = 3.14f;
// Initial vertical angle : none
float verticalAngle = 0.0f;
// Initial Field of View
float initialFoV = 30.0f;

float speed = 3.0f; // 3 units / second
float mouseSpeed = 0.005f;


void computeMatricesFromInputs() {

	// glfwGetTime is called only once, the first time this function is called
	static double lastTime = glfwGetTime();

	// Compute time difference between current and last frame
	double currentTime = glfwGetTime();
	float deltaTime = float(currentTime - lastTime);


	// Get mouse position
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	// 창 크기에 맞게 시점을 만들기 위해 적용
	int stageWidth, stageHeight;
	glfwGetWindowSize(window, &stageWidth, &stageHeight);

	// Reset mouse position for next frame
	glfwSetCursorPos(window, stageWidth / 2, stageHeight / 2);


	// Compute new orientation
	horizontalAngle += mouseSpeed * float(stageWidth / 2 - xpos);
	
	verticalAngle += mouseSpeed * float(stageHeight / 2 - ypos);

	// 마우스
	// 참고로 radius 1.5가 90도 정도 
	if (verticalAngle < -0.2) {
		verticalAngle = -0.2;
	} else if (verticalAngle > 0.4) {
		verticalAngle = 0.4;
	}

	float FoV = initialFoV;// - 5 * glfwGetMouseWheel(); // Now GLFW 3 requires setting up a callback for this. It's a bit too complicated for this beginner's tutorial, so it's disabled instead.

	// Projection matrix : 45?Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	ProjectionMatrix = glm::perspective(glm::radians(FoV), 4.0f / 3.0f, 0.1f, 100.0f);

	// Camera matrix
	/*
	ViewMatrix       = glm::lookAt(
								position,           // Camera is here
								position+direction, // and looks here : at the same position, plus "direction"
								up                  // Head is up (set to 0,-1,0 to look upside-down)
						   );

	*/

	float radius = 1.1f;
	float camX = sin(horizontalAngle) * radius;
	float camZ = cos(horizontalAngle) * radius;
	float camY = sin(verticalAngle) * radius;

	//camY 조정 하는 코드 필요
	/*if (camY >= 2.0) {
		camY = 2.0;
	}
	else if (camY <= 0) {
		camY = 0;
	}*/

	ViewMatrix = glm::lookAt(glm::vec3(camX, 0.8, camZ), glm::vec3(0.0, camY, 0.0), glm::vec3(0.0, 1.0, 0.0));

	
//	ViewMatrix = glm::lookAt(glm::vec3(0, 5, camZ), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0));

	// For the next frame, the "last time" will be "now"
	lastTime = currentTime;
}