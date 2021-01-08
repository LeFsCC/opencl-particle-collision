#include <GL\glut.h>
#include <gl\freeglut.h>
#include <gl\freeglut_ext.h>
#include <gl\freeglut_std.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <cl/cl.h>
#include "SphereSystem.h"
#include "utils.h"

#define GRID_SIZE         64
#define NUM_PARTICLES     1000

using namespace std;
const GLfloat Pi = 3.1415926536f;


//摄像机位置
float gl_view_x = 2;
float gl_view_y = 2.5;
float gl_view_z = 1.75;
//视点位置
float gl_target_x = 0;
float gl_target_y = 0;
float gl_target_z = 0;
//缩放比例
float gl_scale = 1.0f;

// 旋转速率
GLfloat roate = 0.0;
// 旋转角度
GLfloat rote = 0.0;
// 三轴旋转
GLfloat anglex = 0.0;
GLfloat angley = 0.0;
GLfloat anglez = 0.0;
// 窗口大小
GLint WinW = 800;
GLint WinH = 800;
// 记录鼠标坐标
GLfloat oldx;
GLfloat oldy;

void reshapeWindow(GLsizei w, GLsizei h);
void initParticleSystem(int numParticles, uint3 gridSize);
void display(void);
void timer(int id);

// 粒子信息
Spheres* psystem = 0;

float timestep = 0.5f;              // time slice for re-computation iteration
float gravity = 0.0005f;            // Strength of gravity
float damping = 1.0f;
int iterations = 1;
float fParticleRadius = 0.023f;     // Radius of individual particles
int ballr = 8;                      // Radius (in particle diameter equivalents) of dropped/shooting sphere of particles for keys '3' and '4'
float fShootVelocity = -0.07f;      // Velocity of shooting sphere of particles for key '4' (- is away from viewer)
float fColliderRadius = 0.17f;      // Radius of collider for interacting with particles in 'm' mode
float collideSpring = 0.4f;         // Elastic spring constant for impact between particles
float collideDamping = 0.025f;      // Inelastic loss component for impact between particles
float collideShear = 0.12f;         // Friction constant for particles in contact
float collideAttraction = 0.0012f;
uint numParticles = 0;
uint3 gridSize;

//主函数
int main(int argc, char* argv[])
{
	numParticles = NUM_PARTICLES;
	uint gridDim = GRID_SIZE;
	

	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(800, 800);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("sphere collision");

	glEnable(GL_DEPTH_TEST);

	startupOpenCL();
	gridSize.x = gridSize.y = gridSize.z = gridDim;
	initParticleSystem(numParticles, gridSize);
	glutTimerFunc(10, timer, 1);

	glutReshapeFunc(reshapeWindow);

	glutDisplayFunc(display);
	glutMainLoop();
	return 0;
}


void timer(int id) {
	//psystem->update(1);
	glutPostRedisplay();
	glutTimerFunc(10, timer, 1);
}

void reshapeWindow(GLsizei w, GLsizei h)
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45, (GLfloat)w / (GLfloat)h, 0.1, 300);
}

void drawSolidCube(GLfloat x, GLfloat y, GLfloat z, GLfloat xl, GLfloat yl, GLfloat zl, GLubyte red, GLubyte green, GLubyte blue) {
	glPushMatrix();
	glColor3ub(red, green, blue);
	glTranslatef(x, y, z);
	glScalef(xl, yl, zl);
	glutSolidCube(1);
	glPopMatrix();
}

void drawSphere(GLfloat x, GLfloat y, GLfloat z, GLfloat xl, GLfloat yl, GLfloat zl, GLubyte red, GLubyte green, GLubyte blue) {
	glPushMatrix();
	glColor3ub(red, green, blue);
	glTranslatef(x, y, z);
	glScalef(1.0f, 1.0f, 1.0f);
	glutSolidSphere(0.03, 30, 30);
	glPopMatrix();
}


void display(void)
{
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClearDepth(2);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(gl_view_x, gl_view_y, gl_view_z, gl_target_x, gl_target_y, gl_target_z, 0, 1, 0);
	glScalef(gl_scale, gl_scale, gl_scale);

	//地板
	drawSolidCube(0, -1, 0, 2, 0, 2, 241, 241, 241);
	//墙壁1
	drawSolidCube(0, 0, -1, 2, 2, 0, 171, 171, 171);
	//墙壁2
	drawSolidCube(-1, 0, 0, 0, 2, 2, 101, 101, 101);

	float* pos = psystem->getPos();
	for (int i = 0; i < NUM_PARTICLES; i++) {
		drawSphere(pos[i * 4 + 0], pos[i * 4 + 1], pos[i * 4 + 2], pos[i * 4 + 3], pos[i * 4 + 3], pos[i * 4 + 3], (i % 5) * 51, (i / 5) % 5 * 51, (i / 25) * 51);
	}
	glutSwapBuffers();
}

void initParticleSystem(int numParticles, uint3 gridSize) {
	psystem = new Spheres(numParticles, gridSize, fParticleRadius, fColliderRadius);
	psystem->reset(Spheres::CONFIG_RANDOM);
	psystem->setIterations(iterations);
	psystem->setDamping(damping);
	psystem->setGravity(-gravity);
	psystem->setCollideSpring(collideSpring);
	psystem->setCollideDamping(collideDamping);
	psystem->setCollideShear(collideShear);
	psystem->setCollideAttraction(collideAttraction);
}
