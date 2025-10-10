#ifndef MYGLWIDGET_H
#define MYGLWIDGET_H

#ifdef MAC_OS
#include <QtOpenGL/QtOpenGL>
#else
#include <GL/glew.h>
#endif
#include <QtGui>
#include <QtOpenGLWidgets/QOpenGLWidget>
#include <QOpenGLFunctions>

class MyGLWidget : public QOpenGLWidget{
    Q_OBJECT

public:
    MyGLWidget(QWidget *parent = nullptr);
    ~MyGLWidget();

protected:
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);
	void keyPressEvent(QKeyEvent *e);

private:
	int scene_id;
    float rotationX = 0.0f; // ��X�����ת�Ƕ�
    float rotationY = 0.0f; // ��Y�����ת�Ƕ�
    float rotationZ = 0.0f; // ��Z�����ת�Ƕ�
	void scene_0();
	void scene_1();
    void scene_2();
    void scene_3();
    void scene_4();
};
#endif // MYGLWIDGET_H
