#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>
/*###################################################
##  ��: vec3
##  �������� ��ά����
#####################################################*/
class vec3
{
public:
    float x, y, z;
    vec3() {}
    vec3(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}
    inline vec3 operator-() const { return vec3(-x, -y, -z); }
    inline vec3& operator+=(const vec3& v);
    inline vec3& operator-=(const vec3& v);
    inline vec3& operator*=(const vec3& v);
    inline vec3& operator/=(const vec3& v);
    inline vec3& operator*=(const float t);
    inline vec3& operator/=(const float t);

    friend inline vec3 operator+(const vec3& v1, const vec3& v2);
    friend inline vec3 operator-(const vec3& v1, const vec3& v2);
    friend inline vec3 operator*(const vec3& v1, const vec3& v2);
    friend inline vec3 operator/(const vec3& v1, const vec3& v2);
    friend inline vec3 operator*(const float t, const vec3& v);
    friend inline vec3 operator/(const vec3& v, const float t);
    friend inline std::ostream& operator<<(std::ostream& os, const vec3& v);
    //������������
    inline float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }
    //�������ȵ�ƽ��
    inline float squared_length() const {
        return x * x + y * y + z * z;
    }

};
inline float dot(const vec3& v1, const vec3& v2);
inline vec3 cross(const vec3& v1, const vec3& v2);
inline vec3 normalize(const vec3& v);
/*###################################################
##  ����: operator<<
##  ���������� ����������������������
##  ���������� os ��������� v Ŀ������
#####################################################*/
inline std::ostream& operator<<(std::ostream& os, const vec3& v)
{
    os << v.x << " " << v.y << " " << v.z;
    return os;
}
/*###################################################
##  ����: operator+=
##  ���������� ���������+=
##  ���������� v ��������
#####################################################*/
inline vec3& vec3::operator+=(const vec3& v)
{
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}
/*###################################################
##  ����: operator-=
##  ���������� ���������-=
##  ���������� v ��������
#####################################################*/
inline vec3& vec3::operator-=(const vec3& v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
}
/*###################################################
##  ����: operator*=
##  ���������� ���������*=
##  ���������� v ��������
#####################################################*/
inline vec3& vec3::operator*=(const vec3& v)
{
    x *= v.x;
    y *= v.y;
    z *= v.z;
    return *this;
}
/*###################################################
##  ����: operator/=
##  ���������� ���������/=
##  ���������� v ��������
#####################################################*/
inline vec3& vec3::operator/=(const vec3& v)
{
    x /= v.x;
    y /= v.y;
    z /= v.z;
    return *this;
}
/*###################################################
##  ����: operator*=
##  ���������� ���������
##  ���������� t ������
#####################################################*/
inline vec3& vec3::operator*=(const float t)
{
    x *= t;
    y *= t;
    z *= t;
    return *this;
}
/*###################################################
##  ����: operator/=
##  ���������� ���������/=
##  ���������� t ������
#####################################################*/
inline vec3& vec3::operator/=(const float t)
{
    x /= t;
    y /= t;
    z /= t;
    return *this;
}
/*###################################################
##  ����: operator+
##  ���������� ��Ԫ���������������+
##  ���������� v1��v2 ��������
#####################################################*/
inline vec3 operator+(const vec3& v1, const vec3& v2)
{
    return vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}
/*###################################################
##  ����: operator-
##  ���������� ��Ԫ���������������-
##  ���������� v1��v2 ��������
#####################################################*/
inline vec3 operator-(const vec3& v1, const vec3& v2)
{
    return vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}
/*###################################################
##  ����: operator*
##  ���������� ��Ԫ���������������*
##  ���������� v1��v2 ��������
#####################################################*/
inline vec3 operator*(const vec3& v1, const vec3& v2)
{
    return vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}
/*###################################################
##  ����: operator/
##  ���������� ��Ԫ���������������/
##  ���������� v1��v2 ��������
#####################################################*/
inline vec3 operator/(const vec3& v1, const vec3& v2)
{
    return vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}
/*###################################################
##  ����: operator*
##  ���������� ��Ԫ���������������*
##  ���������� t ������ v ��������
#####################################################*/
inline vec3 operator*(const float t, const vec3& v)
{
    return vec3(t * v.x, t * v.y, t * v.z);
}
/*###################################################
##  ����: operator/
##  ���������� ��Ԫ���������������/
##  ���������� t ������ v ��������
#####################################################*/
inline vec3 operator/(const vec3& v, const float t)
{
    float invt = 1.0f / t;
    return vec3(v.x * invt, v.y * invt, v.z * invt);
}
/*###################################################
##  ����: dot
##  ���������� �����ڻ�
##  ���������� v1 v2 ��������
#####################################################*/
inline float dot(const vec3& v1, const vec3& v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
/*###################################################
##  ����: cross
##  ���������� �������
##  ���������� v1 v2 ��������
#####################################################*/
inline vec3 cross(const vec3& v1, const vec3& v2)
{
    return vec3(
        v1.y * v2.z - v2.y * v1.z,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    );
}
/*###################################################
##  ����: normalize
##  ���������� ������һ��
##  ���������� v ��������
#####################################################*/
inline vec3 normalize(const vec3& v)
{
    float invlen = 1.0f / v.length();
    return vec3(v.x * invlen, v.y * invlen, v.z * invlen);
}

#endif // VEC3_H#pragma once
