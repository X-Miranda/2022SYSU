#include "myglwidget.h"

MyGLWidget::MyGLWidget(QWidget *parent)
	:QOpenGLWidget(parent),
	scene_id(4)
{
}

MyGLWidget::~MyGLWidget()
{

}

void MyGLWidget::initializeGL()
{
	glViewport(0, 0, width(), height());
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glDisable(GL_DEPTH_TEST);

    
}

void MyGLWidget::paintGL()
{
	if (scene_id==0) {
		scene_0();
	}
    else if (scene_id == 2) {
        scene_2();
    }
    else if (scene_id == 3) {
        scene_3();
    }
    else if (scene_id == 4) {
        scene_4();
    }
	else {
		scene_1();
	}
}

void MyGLWidget::resizeGL(int width, int height)
{
	glViewport(0, 0, width, height);
	update();
}

void MyGLWidget::keyPressEvent(QKeyEvent *e) {
	//Press 0 or 1 to switch the scene
	if (e->key() == Qt::Key_0) {
		scene_id = 0;
		update();
	}
    else if (e->key() == Qt::Key_2) {
        scene_id = 2;
        update();
    }
	else if (e->key() == Qt::Key_1) {
		scene_id = 1;
		update();
	}
    else if (e->key() == Qt::Key_3) {
        scene_id = 3;
        update();
    }
    else if (e->key() == Qt::Key_4) {
        scene_id = 4;
        update();
    }
    else if (e->key() == Qt::Key_W) { // �Ϸ��������X����������ת
        rotationX += 5.0f;
        update();
    }
    else if (e->key() == Qt::Key_S) { // �·��������X�Ḻ������ת
        rotationX -= 5.0f;
        update();
    }
    else if (e->key() == Qt::Key_A) { // ���������Y����������ת
        rotationY += 5.0f;
        update();
    }
    else if (e->key() == Qt::Key_D) { // �ҷ��������Y�Ḻ������ת
        rotationY -= 5.0f;
        update();
    }
    else if (e->key() == Qt::Key_Q) { // Q������Z����������ת
        rotationZ += 5.0f;
        update();
    }
    else if (e->key() == Qt::Key_E) { // E������Z�Ḻ������ת
        rotationZ -= 5.0f;
        update();
    }
}

void MyGLWidget::scene_0()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	// glOrtho(0.0f, 100.0f, 0.0f, 100.0f, -1000.0f, 1000.0f);

    // ����ͶӰ
    glOrtho(0.0, width(), 0.0, height(), -1.0, 1.0);

    // ͸��ͶӰ
    // gluPerspective(45.0, (GLfloat)width() / (GLfloat)height(), 0.1, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(50.0f, 50.0f, 0.0f);
	
	//draw a diagonal "I"
	glPushMatrix();
	glColor3f(0.839f, 0.153f, 0.157f);
	glRotatef(45.0f, 0.0f, 0.0f, 1.0f);
	glTranslatef(-2.5f, -22.5f, 0.0f);
	glBegin(GL_TRIANGLES);
	glVertex2f(0.0f, 0.0f);
	glVertex2f(5.0f, 0.0f);
	glVertex2f(0.0f, 45.0f);

	glVertex2f(5.0f, 0.0f);
	glVertex2f(0.0f, 45.0f);
	glVertex2f(5.0f, 45.0f);

	glEnd();
	glPopMatrix();	
}

void MyGLWidget::scene_1()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0f, width(), 0.0f, height(), -1000.0f, 1000.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.5 * width(), 0.5 * height(), 0.0f);

    glPushMatrix();

    // ������ĸ X
	glColor3f(0.588f, 0.765f, 0.49f); // ��ɫ
    glTranslatef(-200.0f, -50.0f, 0.0f);
    glBegin(GL_TRIANGLES);
    // ���ϵ����¶Խ���
	glVertex2f(70.0f, 0.0f); glVertex2f(-40.0f, 140.0f); glVertex2f(-70.0f, 140.0f);
	glVertex2f(70.0f, 0.0f); glVertex2f(40.0f, 0.0f); glVertex2f(-70.0f, 140.0f);

    // ���ϵ����¶Խ���
	glVertex2f(-70.0f, 0.0f); glVertex2f(40.0f, 140.0f); glVertex2f(70.0f, 140.0f);
	glVertex2f(-70.0f, 0.0f); glVertex2f(-40.0f, 0.0f); glVertex2f(70.0f, 140.0f);
    glEnd();
    glPopMatrix();

    // ������ĸ Y
    glPushMatrix();
    glTranslatef(0.0f, -50.0f, 0.0f);
    glBegin(GL_TRIANGLES);
    // �ϲ�
	// ���ϵ����¶Խ���
	glVertex2f(-70.0f, 140.0f); glVertex2f(-40.0f, 140.0f); glVertex2f(10.0f, 70.0f);
	glVertex2f(-10.0f, 70.0f); glVertex2f(10.0f, 70.0f); glVertex2f(-70.0f, 140.0f);
	// ���ϵ����¶Խ���
	glVertex2f(70.0f, 140.0f); glVertex2f(40.0f, 140.0f); glVertex2f(-10.0f, 70.0f);
	glVertex2f(10.0f, 70.0f); glVertex2f(-10.0f, 70.0f); glVertex2f(70.0f, 140.0f);

    // ��
	glVertex2f(-10.0f, 70.0f); glVertex2f(10.0f, 70.0f); glVertex2f(10.0f, 0.0f);
	glVertex2f(-10.0f, 70.0f); glVertex2f(-10.0f, 0.0f); glVertex2f(10.0f, 0.0f);
    glEnd();
    glPopMatrix();

    // ������ĸ T
    glPushMatrix();
    glTranslatef(200.0f, -50.0f, 0.0f);
    glBegin(GL_TRIANGLES);
    // ��������
    glVertex2f(-70.0f, 140.0f); glVertex2f(70.0f, 140.0f); glVertex2f(-70.0f, 120.0f);
    glVertex2f(70.0f, 140.0f); glVertex2f(70.0f, 120.0f); glVertex2f(-70.0f, 120.0f);

    // ����
    glVertex2f(-10.0f, 0.0f); glVertex2f(-10.0f, 140.0f); glVertex2f(10.0f, 0.0f);
    glVertex2f(10.0f, 140.0f); glVertex2f(-10.0f, 140.0f); glVertex2f(10.0f, 0.0f);
    glEnd();
    glPopMatrix();

    glPopMatrix();
}

void MyGLWidget::scene_2()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0f, width(), 0.0f, height(), -1000.0f, 1000.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.5 * width(), 0.5 * height(), 0.0f);

    glPushMatrix();

    // ������ĸ X
    glColor3f(0.5f, 0.0f, 0.5f);// ��ɫ
    glPushMatrix();
    glTranslatef(-200.0f, -50.0f, 0.0f);
    glBegin(GL_TRIANGLE_STRIP);
    // ���ϵ����¶Խ���
    glVertex2f(-40.0f, 140.0f); // ��1�������ζ���2
    glVertex2f(-70.0f, 140.0f); // ��1�������ζ���3
    glVertex2f(70.0f, 0.0f);   // ��1�������ζ���1
    glVertex2f(40.0f, 0.0f);    // ��2�������ζ���4��������2��3��
    glEnd();
    glBegin(GL_TRIANGLE_STRIP);
    // ���ϵ����¶Խ���
    glVertex2f(40.0f, 140.0f); // ��1�������ζ���2
    glVertex2f(70.0f, 140.0f); // ��1�������ζ���3
    glVertex2f(-70.0f, 0.0f);   // ��1�������ζ���1
    glVertex2f(-40.0f, 0.0f);    // ��2�������ζ���4��������2��3��
    glEnd();
    glPopMatrix();


    // ������ĸ Y
    glColor3f(0.5f, 0.0f, 0.5f);
    glPushMatrix();
    glTranslatef(0.0f, -50.0f, 0.0f);
    glBegin(GL_TRIANGLE_STRIP);
    // �ϲ����Խ���
    glVertex2f(-70.0f, 140.0f);
    glVertex2f(-40.0f, 140.0f);
    glVertex2f(-10.0f, 70.0f);// �����Խ��߹���������
    glVertex2f(10.0f, 70.0f);// �����Խ��߹���������

    glVertex2f(40.0f, 140.0f);
    glVertex2f(70.0f, 140.0f);
    glEnd();

    glBegin(GL_TRIANGLE_STRIP);
    // ����
    glVertex2f(10.0f, 70.0f);
    glVertex2f(10.0f, 0.0f);
    glVertex2f(-10.0f, 70.0f);
    glVertex2f(-10.0f, 0.0f);
    glEnd();
    glPopMatrix();


    // ������ĸ T
    glColor3f(0.5f, 0.0f, 0.5f);
    glPushMatrix();
    glTranslatef(200.0f, -50.0f, 0.0f);
    glBegin(GL_TRIANGLE_STRIP);
    // ��������
    glVertex2f(-70.0f, 140.0f);
    glVertex2f(-70.0f, 120.0f);
    glVertex2f(70.0f, 140.0f);
    glVertex2f(70.0f, 120.0f);
    glEnd();
    glBegin(GL_TRIANGLE_STRIP);
    // ����
    glVertex2f(-10.0f, 0.0f);
    glVertex2f(-10.0f, 140.0f);
    glVertex2f(10.0f, 0.0f);
    glVertex2f(10.0f, 140.0f);
    
    glEnd();
    glPopMatrix();

    glPopMatrix();
}

void MyGLWidget::scene_3()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    
    //glOrtho(0.0f, width(), 0.0f, height(), -1000.0f, 1000.0f);
    // 
    // ����ͶӰ
    //glOrtho(-700.0f, 700.0f, -500.0f, 500.0f, -1000.0f, 1000.0f);
    
    // ͸��ͶӰ
    gluPerspective(45.0, 1.5f, 0.1f, 1000.0f);

    gluLookAt(0.0f, 0.0f, 1000.0f,  // �۲��λ��
        0.0f, 0.0f, 0.0f,     // Ŀ���λ��
        0.0f, 1.0f, 0.0f);    // �Ϸ���/**/

    /*gluLookAt(0.0f, 500.0f, 1000.0f,  // �۲��λ��
          0.0f, 0.0f          , 0.0f,     // Ŀ���λ��
          0.0f, 1.0f          , 0.0f);    // �Ϸ���*/


    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //glTranslatef(0.5 * width(), 0.5 * height(), 0.0f);

    glPushMatrix();

    // ������ĸ X
    glColor3f(0.0f, 0.5f, 1.0f);
    glPushMatrix();
    glTranslatef(-200.0f, -50.0f, 0.0f);

    // ��������Ʋ
    glBegin(GL_QUAD_STRIP);
    // ���ϵ����¶Խ���
    glVertex2f(-40.0f, 140.0f); 
    glVertex2f(-70.0f, 140.0f); 
    glVertex2f(70.0f, 0.0f);   
    glVertex2f(40.0f, 0.0f);    
    glEnd();
    glBegin(GL_QUAD_STRIP);
    // ���ϵ����¶Խ���
    glVertex2f(40.0f, 140.0f); 
    glVertex2f(70.0f, 140.0f);
    glVertex2f(-70.0f, 0.0f); 
    glVertex2f(-40.0f, 0.0f);  
    glEnd();

    glPopMatrix();


    // ������ĸ Y
    glColor3f(0.0f, 0.5f, 1.0f);
    glPushMatrix();
    glTranslatef(0.0f, -50.0f, 0.0f);
    glBegin(GL_QUAD_STRIP);
    // �ϲ����Խ���
    glVertex2f(-70.0f, 140.0f);
    glVertex2f(-40.0f, 140.0f);
    glVertex2f(-10.0f, 70.0f);
    glVertex2f(10.0f, 70.0f);
    // �ϲ��Ҳ�Խ���
    glVertex2f(40.0f, 140.0f);
    glVertex2f(70.0f, 140.0f);
    
    glEnd();

    glBegin(GL_QUAD_STRIP);
    // ����
    glVertex2f(10.0f, 70.0f);
    glVertex2f(10.0f, 0.0f);
    glVertex2f(-10.0f, 70.0f);
    glVertex2f(-10.0f, 0.0f);
    glEnd();
    glPopMatrix();


    // ������ĸ T
    glColor3f(0.0f, 0.5f, 1.0f);
    glPushMatrix();
    glTranslatef(200.0f, -50.0f, 0.0f);
    glBegin(GL_QUAD_STRIP);
    // ��������
    glVertex2f(-70.0f, 140.0f);
    glVertex2f(-70.0f, 120.0f);
    glVertex2f(70.0f, 140.0f);
    glVertex2f(70.0f, 120.0f);
    glEnd();
    glBegin(GL_QUAD_STRIP);
    // ����
    glVertex2f(-10.0f, 0.0f);
    glVertex2f(-10.0f, 140.0f);
    glVertex2f(10.0f, 0.0f);
    glVertex2f(10.0f, 140.0f);

    glEnd();

    glPopMatrix();
    glPopMatrix();
}


void MyGLWidget::scene_4() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    // ����ͶӰ����
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, (GLfloat)width() / (GLfloat)height(), 1.0f, 2000.0f);

    // ����ģ����ͼ����
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.0f, 0.0f, 500.0f, // �۾�λ��
        0.0f, 0.0f, 0.0f,  // �����
        0.0f, 1.0f, 0.0f); // ������

    // Ӧ����ת
    glRotatef(rotationX, 1.0f, 0.0f, 0.0f); // ��X����ת
    glRotatef(rotationY, 0.0f, 1.0f, 0.0f); // ��Y����ת
    glRotatef(rotationZ, 0.0f, 0.0f, 1.0f); // ��Z����ת

    // ������ĸ "X" �ĺ��
    float thickness = 40.0f;

    // ������ĸ "X"
    

    // ���ϵ����¶Խ���
    glBegin(GL_QUAD_STRIP);
    glColor3f(0.588f, 0.765f, 0.49f); // ��ɫ
    glVertex3f(-40.0f, 140.0f, thickness / 2);
    glVertex3f(-70.0f, 140.0f, thickness / 2);
    glVertex3f(70.0f, 0.0f, thickness / 2);
    glVertex3f(40.0f, 0.0f, thickness / 2);
    glEnd();

    glBegin(GL_QUAD_STRIP);
    glColor3f(0.588f, 0.765f, 0.49f); // ��ɫ
    glVertex3f(-40.0f, 140.0f, -thickness / 2);
    glVertex3f(-70.0f, 140.0f, -thickness / 2);
    glVertex3f(70.0f, 0.0f, -thickness / 2);
    glVertex3f(40.0f, 0.0f, -thickness / 2);
    glEnd();

    // ���ϵ����¶Խ���
    glBegin(GL_QUAD_STRIP);
    glColor3f(0.0f, 0.5f, 1.0f); // ��ɫ
    glVertex3f(40.0f, 140.0f, thickness / 2);
    glVertex3f(70.0f, 140.0f, thickness / 2);
    glVertex3f(-70.0f, 0.0f, thickness / 2);
    glVertex3f(-40.0f, 0.0f, thickness / 2);
    glEnd();

    glBegin(GL_QUAD_STRIP);
    glColor3f(0.0f, 0.5f, 1.0f); // ��ɫ
    glVertex3f(40.0f, 140.0f, -thickness / 2);
    glVertex3f(70.0f, 140.0f, -thickness / 2);
    glVertex3f(-70.0f, 0.0f, -thickness / 2);
    glVertex3f(-40.0f, 0.0f, -thickness / 2);
    glEnd();

    //connect
    // ���ϵ����¶Խ���
    glBegin(GL_QUAD_STRIP);
    glColor3f(0.5f, 0.0f, 0.5f);
    glVertex3f(-70.0f, 140.0f, thickness / 2);
    glVertex3f(-70.0f, 140.0f, -thickness / 2);
    glVertex3f(-40.0f, 140.0f, thickness / 2);
    glVertex3f(-40.0f, 140.0f, -thickness / 2);
    glVertex3f(70.0f, 0.0f, thickness / 2);
    glVertex3f(70.0f, 0.0f, -thickness / 2);
    glVertex3f(40.0f, 0.0f, thickness / 2);
    glVertex3f(40.0f, 0.0f, -thickness / 2);
    glVertex3f(-70.0f, 140.0f, thickness / 2);
    glVertex3f(-70.0f, 140.0f, -thickness / 2);
    
    glEnd();

    // ���ϵ����¶Խ���
    glBegin(GL_QUAD_STRIP);
    glColor3f(1.0f, 0.75f, 0.8f);
    glVertex3f(70.0f, 140.0f, thickness / 2);
    glVertex3f(70.0f, 140.0f, -thickness / 2);
    glVertex3f(40.0f, 140.0f, thickness / 2);
    glVertex3f(40.0f, 140.0f, -thickness / 2);
    glVertex3f(-70.0f, 0.0f, thickness / 2);
    glVertex3f(-70.0f, 0.0f, -thickness / 2);
    glVertex3f(-40.0f, 0.0f, thickness / 2);
    glVertex3f(-40.0f, 0.0f, -thickness / 2);
    glVertex3f(70.0f, 140.0f, thickness / 2);
    glVertex3f(70.0f, 140.0f, -thickness / 2);
    glEnd();
   
    
    glPopMatrix();
    glDisable(GL_DEPTH_TEST);
}



