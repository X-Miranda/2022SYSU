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
    else if (e->key() == Qt::Key_W) { // 上方向键，绕X轴正方向旋转
        rotationX += 5.0f;
        update();
    }
    else if (e->key() == Qt::Key_S) { // 下方向键，绕X轴负方向旋转
        rotationX -= 5.0f;
        update();
    }
    else if (e->key() == Qt::Key_A) { // 左方向键，绕Y轴正方向旋转
        rotationY += 5.0f;
        update();
    }
    else if (e->key() == Qt::Key_D) { // 右方向键，绕Y轴负方向旋转
        rotationY -= 5.0f;
        update();
    }
    else if (e->key() == Qt::Key_Q) { // Q键，绕Z轴正方向旋转
        rotationZ += 5.0f;
        update();
    }
    else if (e->key() == Qt::Key_E) { // E键，绕Z轴负方向旋转
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

    // 正交投影
    glOrtho(0.0, width(), 0.0, height(), -1.0, 1.0);

    // 透视投影
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

    // 绘制字母 X
	glColor3f(0.588f, 0.765f, 0.49f); // 绿色
    glTranslatef(-200.0f, -50.0f, 0.0f);
    glBegin(GL_TRIANGLES);
    // 左上到右下对角线
	glVertex2f(70.0f, 0.0f); glVertex2f(-40.0f, 140.0f); glVertex2f(-70.0f, 140.0f);
	glVertex2f(70.0f, 0.0f); glVertex2f(40.0f, 0.0f); glVertex2f(-70.0f, 140.0f);

    // 右上到左下对角线
	glVertex2f(-70.0f, 0.0f); glVertex2f(40.0f, 140.0f); glVertex2f(70.0f, 140.0f);
	glVertex2f(-70.0f, 0.0f); glVertex2f(-40.0f, 0.0f); glVertex2f(70.0f, 140.0f);
    glEnd();
    glPopMatrix();

    // 绘制字母 Y
    glPushMatrix();
    glTranslatef(0.0f, -50.0f, 0.0f);
    glBegin(GL_TRIANGLES);
    // 上部
	// 左上到右下对角线
	glVertex2f(-70.0f, 140.0f); glVertex2f(-40.0f, 140.0f); glVertex2f(10.0f, 70.0f);
	glVertex2f(-10.0f, 70.0f); glVertex2f(10.0f, 70.0f); glVertex2f(-70.0f, 140.0f);
	// 右上到左下对角线
	glVertex2f(70.0f, 140.0f); glVertex2f(40.0f, 140.0f); glVertex2f(-10.0f, 70.0f);
	glVertex2f(10.0f, 70.0f); glVertex2f(-10.0f, 70.0f); glVertex2f(70.0f, 140.0f);

    // 竖
	glVertex2f(-10.0f, 70.0f); glVertex2f(10.0f, 70.0f); glVertex2f(10.0f, 0.0f);
	glVertex2f(-10.0f, 70.0f); glVertex2f(-10.0f, 0.0f); glVertex2f(10.0f, 0.0f);
    glEnd();
    glPopMatrix();

    // 绘制字母 T
    glPushMatrix();
    glTranslatef(200.0f, -50.0f, 0.0f);
    glBegin(GL_TRIANGLES);
    // 顶部横线
    glVertex2f(-70.0f, 140.0f); glVertex2f(70.0f, 140.0f); glVertex2f(-70.0f, 120.0f);
    glVertex2f(70.0f, 140.0f); glVertex2f(70.0f, 120.0f); glVertex2f(-70.0f, 120.0f);

    // 竖线
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

    // 绘制字母 X
    glColor3f(0.5f, 0.0f, 0.5f);// 紫色
    glPushMatrix();
    glTranslatef(-200.0f, -50.0f, 0.0f);
    glBegin(GL_TRIANGLE_STRIP);
    // 左上到右下对角线
    glVertex2f(-40.0f, 140.0f); // 第1个三角形顶点2
    glVertex2f(-70.0f, 140.0f); // 第1个三角形顶点3
    glVertex2f(70.0f, 0.0f);   // 第1个三角形顶点1
    glVertex2f(40.0f, 0.0f);    // 第2个三角形顶点4（共享顶点2和3）
    glEnd();
    glBegin(GL_TRIANGLE_STRIP);
    // 右上到左下对角线
    glVertex2f(40.0f, 140.0f); // 第1个三角形顶点2
    glVertex2f(70.0f, 140.0f); // 第1个三角形顶点3
    glVertex2f(-70.0f, 0.0f);   // 第1个三角形顶点1
    glVertex2f(-40.0f, 0.0f);    // 第2个三角形顶点4（共享顶点2和3）
    glEnd();
    glPopMatrix();


    // 绘制字母 Y
    glColor3f(0.5f, 0.0f, 0.5f);
    glPushMatrix();
    glTranslatef(0.0f, -50.0f, 0.0f);
    glBegin(GL_TRIANGLE_STRIP);
    // 上部左侧对角线
    glVertex2f(-70.0f, 140.0f);
    glVertex2f(-40.0f, 140.0f);
    glVertex2f(-10.0f, 70.0f);// 两个对角线共用这两条
    glVertex2f(10.0f, 70.0f);// 两个对角线共用这两条

    glVertex2f(40.0f, 140.0f);
    glVertex2f(70.0f, 140.0f);
    glEnd();

    glBegin(GL_TRIANGLE_STRIP);
    // 竖线
    glVertex2f(10.0f, 70.0f);
    glVertex2f(10.0f, 0.0f);
    glVertex2f(-10.0f, 70.0f);
    glVertex2f(-10.0f, 0.0f);
    glEnd();
    glPopMatrix();


    // 绘制字母 T
    glColor3f(0.5f, 0.0f, 0.5f);
    glPushMatrix();
    glTranslatef(200.0f, -50.0f, 0.0f);
    glBegin(GL_TRIANGLE_STRIP);
    // 顶部横线
    glVertex2f(-70.0f, 140.0f);
    glVertex2f(-70.0f, 120.0f);
    glVertex2f(70.0f, 140.0f);
    glVertex2f(70.0f, 120.0f);
    glEnd();
    glBegin(GL_TRIANGLE_STRIP);
    // 竖线
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
    // 正交投影
    //glOrtho(-700.0f, 700.0f, -500.0f, 500.0f, -1000.0f, 1000.0f);
    
    // 透视投影
    gluPerspective(45.0, 1.5f, 0.1f, 1000.0f);

    gluLookAt(0.0f, 0.0f, 1000.0f,  // 观察点位置
        0.0f, 0.0f, 0.0f,     // 目标点位置
        0.0f, 1.0f, 0.0f);    // 上方向/**/

    /*gluLookAt(0.0f, 500.0f, 1000.0f,  // 观察点位置
          0.0f, 0.0f          , 0.0f,     // 目标点位置
          0.0f, 1.0f          , 0.0f);    // 上方向*/


    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //glTranslatef(0.5 * width(), 0.5 * height(), 0.0f);

    glPushMatrix();

    // 绘制字母 X
    glColor3f(0.0f, 0.5f, 1.0f);
    glPushMatrix();
    glTranslatef(-200.0f, -50.0f, 0.0f);

    // 左竖和左撇
    glBegin(GL_QUAD_STRIP);
    // 左上到右下对角线
    glVertex2f(-40.0f, 140.0f); 
    glVertex2f(-70.0f, 140.0f); 
    glVertex2f(70.0f, 0.0f);   
    glVertex2f(40.0f, 0.0f);    
    glEnd();
    glBegin(GL_QUAD_STRIP);
    // 右上到左下对角线
    glVertex2f(40.0f, 140.0f); 
    glVertex2f(70.0f, 140.0f);
    glVertex2f(-70.0f, 0.0f); 
    glVertex2f(-40.0f, 0.0f);  
    glEnd();

    glPopMatrix();


    // 绘制字母 Y
    glColor3f(0.0f, 0.5f, 1.0f);
    glPushMatrix();
    glTranslatef(0.0f, -50.0f, 0.0f);
    glBegin(GL_QUAD_STRIP);
    // 上部左侧对角线
    glVertex2f(-70.0f, 140.0f);
    glVertex2f(-40.0f, 140.0f);
    glVertex2f(-10.0f, 70.0f);
    glVertex2f(10.0f, 70.0f);
    // 上部右侧对角线
    glVertex2f(40.0f, 140.0f);
    glVertex2f(70.0f, 140.0f);
    
    glEnd();

    glBegin(GL_QUAD_STRIP);
    // 竖线
    glVertex2f(10.0f, 70.0f);
    glVertex2f(10.0f, 0.0f);
    glVertex2f(-10.0f, 70.0f);
    glVertex2f(-10.0f, 0.0f);
    glEnd();
    glPopMatrix();


    // 绘制字母 T
    glColor3f(0.0f, 0.5f, 1.0f);
    glPushMatrix();
    glTranslatef(200.0f, -50.0f, 0.0f);
    glBegin(GL_QUAD_STRIP);
    // 顶部横线
    glVertex2f(-70.0f, 140.0f);
    glVertex2f(-70.0f, 120.0f);
    glVertex2f(70.0f, 140.0f);
    glVertex2f(70.0f, 120.0f);
    glEnd();
    glBegin(GL_QUAD_STRIP);
    // 竖线
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

    // 设置投影矩阵
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, (GLfloat)width() / (GLfloat)height(), 1.0f, 2000.0f);

    // 设置模型视图矩阵
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.0f, 0.0f, 500.0f, // 眼睛位置
        0.0f, 0.0f, 0.0f,  // 看向点
        0.0f, 1.0f, 0.0f); // 上向量

    // 应用旋转
    glRotatef(rotationX, 1.0f, 0.0f, 0.0f); // 绕X轴旋转
    glRotatef(rotationY, 0.0f, 1.0f, 0.0f); // 绕Y轴旋转
    glRotatef(rotationZ, 0.0f, 0.0f, 1.0f); // 绕Z轴旋转

    // 设置字母 "X" 的厚度
    float thickness = 40.0f;

    // 绘制字母 "X"
    

    // 左上到右下对角线
    glBegin(GL_QUAD_STRIP);
    glColor3f(0.588f, 0.765f, 0.49f); // 绿色
    glVertex3f(-40.0f, 140.0f, thickness / 2);
    glVertex3f(-70.0f, 140.0f, thickness / 2);
    glVertex3f(70.0f, 0.0f, thickness / 2);
    glVertex3f(40.0f, 0.0f, thickness / 2);
    glEnd();

    glBegin(GL_QUAD_STRIP);
    glColor3f(0.588f, 0.765f, 0.49f); // 绿色
    glVertex3f(-40.0f, 140.0f, -thickness / 2);
    glVertex3f(-70.0f, 140.0f, -thickness / 2);
    glVertex3f(70.0f, 0.0f, -thickness / 2);
    glVertex3f(40.0f, 0.0f, -thickness / 2);
    glEnd();

    // 右上到左下对角线
    glBegin(GL_QUAD_STRIP);
    glColor3f(0.0f, 0.5f, 1.0f); // 蓝色
    glVertex3f(40.0f, 140.0f, thickness / 2);
    glVertex3f(70.0f, 140.0f, thickness / 2);
    glVertex3f(-70.0f, 0.0f, thickness / 2);
    glVertex3f(-40.0f, 0.0f, thickness / 2);
    glEnd();

    glBegin(GL_QUAD_STRIP);
    glColor3f(0.0f, 0.5f, 1.0f); // 蓝色
    glVertex3f(40.0f, 140.0f, -thickness / 2);
    glVertex3f(70.0f, 140.0f, -thickness / 2);
    glVertex3f(-70.0f, 0.0f, -thickness / 2);
    glVertex3f(-40.0f, 0.0f, -thickness / 2);
    glEnd();

    //connect
    // 左上到右下对角线
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

    // 右上到左下对角线
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



