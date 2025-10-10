#include "camera.h"
/*###################################################
##  ����: Camera
##  ���������� �������������
##  ���������� pos �����λ��
##          target ������۲�Ŀ��
##          r �����right����
##          pitch �����ǽǶȣ� yaw ƫ���ǽǶ�
#####################################################*/
Camera::Camera(const vec3& pos, const vec3& target, const vec3& r, float Pit, float Yaw)
{
    //�������������ϵ
    position = pos;
    right = normalize(r);
    in = normalize(pos - target);
    front = -in;
    up = normalize(cross(in, right));
    //�����ٶ�
    fbward_speed = vertical_speed = horizontal_speed = 5.0f;
    pitch_speed = yaw_speed = 0.3f;
    //�����ӽ�
    zoom = 100.0f;
    mouse_sensitivity = 0.3f;
    pitch = Pit;
    yaw = Yaw;
    worldUp = vec3(0.0f, 1.0f, 0.0);
}
/*###################################################
##  ����: FBwardMove
##  ���������� ���������ǰ�����˶�
##  ���������� action �ƶ�����
#####################################################*/
void Camera::FBwardMove(Action action)
{
    if (action == forward)
    {
        position += fbward_speed * vec3(front.x, 0.0f, front.z);
    }
    else
    {
        position -= fbward_speed * vec3(front.x, 0.0f, front.z);
    }
}
/*###################################################
##  ����: HorizontalMove
##  ���������� ����������������˶�
##  ���������� action �ƶ�����
#####################################################*/
void Camera::HorizontalMove(Action action)
{
    if (action == toleft)
    {
        position -= horizontal_speed * vec3(right.x, 0.0f, right.z);
    }
    else
    {
        position += horizontal_speed * vec3(right.x, 0.0f, right.z);
    }
}
/*###################################################
##  ����: VerticalMove
##  ���������� �����������ֱ���˶�
##  ���������� action �ƶ�����
#####################################################*/
void Camera::VerticalMove(Action action)
{
    if (action == upward)
    {
        position.y += 10 * vertical_speed;
    }
    else
    {
        position.y -= 10 * vertical_speed;
    }
}
/*###################################################
##  ����: ProcessMouseMovement
##  ���������� ����������ӽ��ƶ�
##  ���������� xOffset ˮƽ�����ƶ�����
##           yOffset ��ֱ�����ƶ�����
#####################################################*/
void Camera::ProcessMouseMovement(float xOffset, float yOffset)
{
    //�����ӽ�ƫ����
    xOffset *= yaw_speed;
    yOffset *= pitch_speed;
    //���¸����ǣ�ƫ���ǽǶ�
    yaw += xOffset;
    pitch += yOffset;

    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;
    //�������������ϵ
    vec3 tmp;
    tmp.x = std::cos(radian(yaw)) * std::cos(radian(pitch));
    tmp.y = std::sin(radian(pitch));
    tmp.z = std::sin(radian(yaw)) * std::cos(radian(pitch));
    front = normalize(tmp);
    right = normalize(cross(front, worldUp));
    up = normalize(cross(right, front));

}
/*###################################################
##  ����: radian
##  ���������� �Ƕ�ת����
##  ���������� angle �Ƕ�
#####################################################*/
float radian(const float angle)
{
    return angle * 3.1415926535897626f / 180.0f;
}

/*###################################################
##  ����: setFixed
##  ���������� ���ڹ̶������
##  ����������
#####################################################*/
void Camera::setFixed() {
    fbward_speed = vertical_speed = horizontal_speed = 0.0f;
    pitch_speed = yaw_speed = 0.0f;
}