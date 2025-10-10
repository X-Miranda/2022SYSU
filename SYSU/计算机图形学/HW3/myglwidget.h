#ifndef MYGLWIDGET_H
#define MYGLWIDGET_H

#ifdef MAC_OS
#include <QtOpenGL/QtOpenGL>
#else
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <QtGui>
#include <QOpenGLWidget>
#include <qopenglextrafunctions.h>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include "utils.h"

#endif

#define MAX_Z_BUFFER 99999999.0f
#define MIN_FLOAT 1e-10f

using namespace glm;

class MyGLWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
    Q_OBJECT

public:
    MyGLWidget(QWidget* parent = nullptr);
    ~MyGLWidget();

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int width, int height) override;
    void keyPressEvent(QKeyEvent* e);

private:

    void scene_0();
    void scene_1();
    void scene_2();
    void scene_3();
    GLuint shaderProgram;
    GLuint vao;
    GLuint vbo;
    GLuint ebo;
    Model objModel;
    vec3 camPosition;
    vec3 camLookAt;
    vec3 camUp;
    vec2 offset;
    mat4 projMatrix;
    mat4 viewMatrix;
    mat4 modelMatrix;
    vec3 lightPosition;
    int scene_id;
    int draw_id;
    int degree = 0; 
    int WindowSizeH = 0;
    int WindowSizeW = 0;
    float camSpeed = 0.5f;
    int depth = 5;

    std::vector<float> vertexData;
    std::vector<GLuint> indices; // Ë÷ÒýÊý¾Ý
    std::vector<vec3> triangles;
    std::vector<vec3> triangles_v;
    std::vector<vec3> triangles_n;

    void initShader();
    void initVBO();
    void initVBOVAO();

    void Ball(int depth);
    void divide_triangle(vec3 v1, vec3 v2, vec3 v3, int depth);
    void getTriangleData(Triangle triangle); 
    void update_VBO();

};

#endif // MYGLWIDGET_H