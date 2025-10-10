#ifndef CAMERA_H
#define CAMERA_H
#include <GL/glut.h>
#include <cmath>
#include <algorithm>
#include "vec3.h"

enum Action;
/*###################################################
##  ö�ٱ���: Action
##  �����������ƶ�����
#####################################################*/
enum Action {
    forward, backward, upward, downward, toleft, toright
};
/*###################################################
##  ��: Camera
##  �������� �����
#####################################################*/
class Camera
{
private:
    float mouse_sensitivity;    //���������
    float fbward_speed;         //ǰ���ƶ��ٶ�
    float vertical_speed;       //��ֱ�ƶ��ٶ�
    float horizontal_speed;     //ˮƽ�ƶ��ٶ�
    float pitch_speed;      //�����Ǳ任�ٶ�
    float yaw_speed;        //ƫ�����ƶ��ٶ�
    float pitch, yaw;       //�����ǣ�ƫ����ƫ�ƽǶ�
    vec3 worldUp;           //���������Up����
public:
    float zoom;     //�ӽǴ�С
    vec3 position;  //λ��
    vec3 in;
    vec3 right;
    vec3 front;
    vec3 up;

    Camera(const vec3& pos, const vec3& target, const vec3& r, float Pit, float Yaw);

    void FBwardMove(Action action);
    void HorizontalMove(Action action);
    void VerticalMove(Action action);
    void ProcessMouseMovement(float xOffset, float yOffset);
    void setFixed();
};
float radian(const float angle);
#endif // CAMERA_H#pragma once
