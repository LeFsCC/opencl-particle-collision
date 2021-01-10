#include <GL\glut.h>
#include <gl\freeglut.h>
#include <gl\freeglut_ext.h>
#include <gl\freeglut_std.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <time.h>
#include <cl/cl.h>
#include <thread>
#include "SphereSystem.h"
#include "utils.h"

#define GRID_SIZE         64
#define NUM_PARTICLES     1024

using namespace std;

//�����λ��
float view_x = 6;
float view_y = 0.0;
float view_z = 1.75;

//�ӵ�λ��
float target_x = 0;
float target_y = 0;
float target_z = 0;
//���ű���
float scale = 1.0f;

void reshapeWindow(GLsizei w, GLsizei h);
void display(void);
void timer(int id);
void initLight();

// ������Ϣ
Spheres* psystem = 0;
float particle_radius = 0.023f;
float collider_radius = 0.17f;
uint num_particles = 0;
uint3 grid_size;

//������
int main(int argc, char* argv[]) {
	srand(time(0));
	num_particles = NUM_PARTICLES;
	uint gridDim = GRID_SIZE;
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(1000, 1000);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("sphere collision");
	glEnable(GL_DEPTH_TEST);
	prepare_ocl_platform();
	grid_size.x = grid_size.y = grid_size.z = gridDim;
	psystem = new Spheres(num_particles, grid_size);
	psystem->init_particle_params();
	glutTimerFunc(25, timer, 1);
	glutReshapeFunc(reshapeWindow);
	glutDisplayFunc(display);
	initLight();
	glutMainLoop();
	return 0;
}

// ʱ��ص�����
void timer(int id) {
	psystem->update(2.1);
	glutPostRedisplay();
	glutTimerFunc(10, timer, 1);
}

// ���ڴ�С�仯�ص�����
void reshapeWindow(GLsizei w, GLsizei h) {
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45, (GLfloat)w / (GLfloat)h, 0.1, 300);
}

// ������Χ������ǽ
void drawWall(GLfloat x, GLfloat y, GLfloat z, GLfloat xl, GLfloat yl, GLfloat zl, GLfloat red, GLfloat green, GLfloat blue) {
	glPushMatrix();
	GLfloat color[] = { red, green, blue };
	glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color);
	glTranslatef(x, y, z);
	glScalef(xl, yl, zl);
	glutWireCube(1);
	glPopMatrix();
}

void drawSphere(GLfloat x, GLfloat y, GLfloat z, GLfloat r, GLfloat red, GLfloat green, GLfloat blue) {
	glPushMatrix();
	GLfloat color[] = { red, green, blue };
	glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color);
	glTranslatef(x, y, z);
	glScalef(1.0f, 1.0f, 1.0f);
	glutSolidSphere(r, 30, 30);
	glPopMatrix();
}

// ��������
void drawCoordinate() {
	GLfloat T = 0.8;
	drawWall(0, -1, 0, 2, 0, 2, T, T, T);
	drawWall(0, 1, 0, 2, 0, 2, T, T, T);

	drawWall(0, 0, -1, 2, 2, 0, T, T, T);
	drawWall(0, 0, 1, 2, 2, 0, T, T, T);

	drawWall(-1, 0, 0, 0, 2, 2, T, T, T);
	drawWall(1, 0, 0, 0, 2, 2, T, T, T);
}


void initLight() {
	//������Ȳ���
	glEnable(GL_DEPTH_TEST);
	//����ɢ��;�����Ϊ�׹�
	GLfloat color[] = { 0.4, 0.8, 0.1 };
	glLightfv(GL_LIGHT0, GL_DIFFUSE, color);
	glLightfv(GL_LIGHT0, GL_SPECULAR, color);
	//����ǰ����ĸ߹⾵����Ϊ�׹�
	glMaterialfv(GL_FRONT, GL_SPECULAR, color);
	//����ǰ����ɢ��ⷴ��ϵ��
	glMaterialf(GL_FRONT, GL_SHININESS, 30);
	//����ƹ�
	glEnable(GL_LIGHTING);
	//��0��
	glEnable(GL_LIGHT0);
	//��Դλ�ò���
	GLfloat lightPosition[] = { 4, 8, 7, 1 };
	//���ù�Դλ��
	glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
}

void display(void) {
	glClearColor(0.2, 0.2, 0.2, 0.4);
	glClearDepth(2);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(view_x, view_y, view_z, target_x, target_y, target_z, 0, 1, 0);
	glScalef(scale, scale, scale);

	drawCoordinate();

	float* pos = psystem->get_pos();
	
	// ���л�������
	for (int i = 0; i < NUM_PARTICLES; i++) {
		GLfloat red = (i % NUM_PARTICLES) / (float)NUM_PARTICLES;
		GLfloat green = (i % NUM_PARTICLES) / (float)NUM_PARTICLES;
		GLfloat blue = (i % NUM_PARTICLES) / (float)NUM_PARTICLES;
		drawSphere(pos[i * 4 + 0], pos[i * 4 + 1], pos[i * 4 + 2], pos[i * 4 + 3], red, green, blue);
	}
	glutSwapBuffers();

}

