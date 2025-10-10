#ifndef MYGLWIDGET_H
#define MYGLWIDGET_H

#include <QtGui>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QPainter>
#include <QPen>
#include <QFont>
#include <QTime>
#include "camera.h"
#include "maze.h"
#include "textures.h"
#include "player.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <QOpenGLFunctions_3_3_Core>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


class MyGLWidget : public QOpenGLWidget {
    Q_OBJECT

public:
    MyGLWidget(QWidget* parent = nullptr);
    ~MyGLWidget();
    void paintMaze();
    void paintSkyBox();
    void paintMark();
    bool keyStates[256]; // ������256�����ܵļ�ֵ
    bool isJumping;          // �Ƿ�������Ծ
    bool isFalling;          // �Ƿ� ������Ծ �� ��Ȼ����
    bool isFlying;          // �Ƿ����ڷ���
    bool isforwarding;
    float forwardStartTime;
    float jumpStartTime;     // ��Ծ��ʼ��ʱ�䣨��Ծ������Ծ��׹��
    float jumpInitialHeight; // ��Ծ��ʼʱ�ĸ߶�
    float fallStartTime;  // ��¼��Ȼ��׹�Ŀ�ʼʱ�䣨��������Ծ��׹��
    float fallInitialHeight; // ��Ȼ��׹��ʼʱ�ĸ߶�
    float forwardinitial;
    float jumpPeakHeight;    // ��Ծ����ߵ�
    float jumpVelocity; // ��Ծ�ĳ�ʼ�ٶ�
    float moveVelocity;
    float gravity = 1000;
    int updateCounter = 0;  // ��¼���´���
    int currentFrameForTrap = 0;  // ���嶯������ǰ֡��
    int frameSizeForTrap;  // ���嶯��������֡��
    int currentFrameForWall = 0;  // ǽ�ڶ�������ǰ֡��
    int frameSizeForWall;  // ǽ�ڶ���������֡��
    GLfloat lightPosition[4] = { 1000.0f, 1300.0f, 1000.0f, 1.0f }; // ��ʼ��Դλ��
    const float frameTime = 0.01f; // 10 ����ת��Ϊ��
    float skyboxAngle = 0.0f; // ��պ���ת�Ƕ�
    void setDiamondMaterial();
    void setDiamondMaterial2();
    void setDefaultMaterial();
    void setLight();

protected:
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);
    void keyPressEvent(QKeyEvent* e);
    void keyReleaseEvent(QKeyEvent* event);
    void mouseMoveEvent(QMouseEvent* e);
    void updateJumpState();
    void updateFallState();  // �Ӱ�ǽ����ͨ���ߣ�Ҫ��Ȼ��׹
    void updatespacestate(void);
    void drawMiniMap();
    void updateforward();
    void updateFlyState();
    void updateLightPosition();


private:
    QTimer* timer;  //��ʱ��
    Player* player; //���
    Textures skyBox;    //��պ�����
    std::vector<vec3> snow; //����յ������
    float lastX, lastY;     //���λ��

    bool firstMouse;
};
#endif // MYGLWIDGET_H