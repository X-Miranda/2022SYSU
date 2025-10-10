#ifndef TEXTURES_H
#define TEXTURES_H
#include <QOpenGLFunctions>
#include <vector>
#include <iostream>
#include <algorithm>
#include <GL/glut.h>
#include <windows.h>

/*###################################################
##  ��: Textures
##  �������� ��¼������Ϣ
#####################################################*/
class Textures
{
public:
    GLuint textures2D[20];  //2D������Ϣ
    GLuint texturesCube[20];    //������������Ϣ
    int count2D;    //2D�������
    int countCube;  //����������ĸ���
    void LoadTexture2D(const char* filename);
    void LoadTextureCube(const char (*filenames)[100]);
    Textures();
};

#endif // TEXTURES_H
