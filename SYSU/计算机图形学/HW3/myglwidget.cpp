#include "GL/glew.h"
#include "myglwidget.h"
#include <algorithm>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <qopenglextrafunctions.h>
#include <QOpenGLFunctions>
#include <QtCore/qdir.h>
#include <sstream>
#include <time.h>

using namespace std;


// 顶点着色器源码
const char* vertexShaderSource = R"glsl(
#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

out vec3 fragNormal;
out vec3 fragPosition;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projMatrix;

uniform vec3 lightPosition;
uniform vec3 viewPosition;

void main() {
	// 计算变换后的顶点位置
	fragPosition = vec3(modelMatrix * vec4(position, 1.0));

	// 计算变换后的法向量
	fragNormal = mat3(modelMatrix) * normal;

	// 计算裁剪坐标
	gl_Position = projMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);

	// 计算从顶点位置到光源位置的向量
	vec4 vertex_in_modelview_space = viewMatrix * modelMatrix * vec4(position, 1.0);
	vec3 vertex_to_light_vector = lightPosition - vertex_in_modelview_space.xyz; // 
}
)glsl";

// 片段着色器代码
const char* fragmentShaderSource = R"glsl(
#version 330 core

in vec3 fragNormal;
in vec3 fragPosition;

out vec4 fragColor;

uniform vec3 lightPosition;
uniform vec3 lightColor = vec3(0.5, 0.0, 0.8); // 光源颜色
uniform vec3 viewPosition; // 观察者位置
uniform float shininess = 250.0; // 镜面高光参数

void main() {
    // 基础颜色
    vec3 baseColor = vec3(1.0, 1.0, 1.0);
    vec3 ambientColor = vec3(0.1, 0.1, 0.1);  // 环境光颜色
    vec3 specularColor = vec3(0.2, 0.2, 0.2); // 镜面反射颜色
    
    // 环境光分量
    vec3 color = baseColor * ambientColor * 0.2 * lightColor;

    // 漫反射光
    vec3 diffuseColor = vec3(0.9, 0.9, 0.9); // 漫反射颜色
    vec3 lightDir = normalize(lightPosition - fragPosition); // 光源方向
    vec3 norm = normalize(fragNormal); // 法线
    float diff = max(dot(norm, lightDir), 0.0f); // 漫反射强度
    color += diff * diffuseColor;

    // 镜面反射分量
    vec3 viewDir = normalize(viewPosition - fragPosition); // 观察方向
    vec3 reflectDir = reflect(-lightDir, norm); // 反射方向
    float spec = pow(max(dot(viewDir, reflectDir), 0.0f), shininess);
    color += spec * specularColor;

    fragColor = vec4(color, 1.0); // 输出最终颜色
}
)glsl";


MyGLWidget::MyGLWidget(QWidget* parent)
    :QOpenGLWidget(parent)
{
}

MyGLWidget::~MyGLWidget()
{
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(shaderProgram);
}


void MyGLWidget::initializeGL() {

    initializeOpenGLFunctions();
    WindowSizeW = width();
    WindowSizeH = height();
    glViewport(0, 0, WindowSizeW, WindowSizeH);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    offset = vec2(WindowSizeH / 2, WindowSizeW / 2);
    // 开启深度，实现遮挡关系
    glEnable(GL_DEPTH_TEST); 
    // 清理缓存
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
}


// 初始化着色器
void MyGLWidget::initShader() {
    // 编译顶点着色器
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, 0);
    glCompileShader(vertexShader);

    // 编译片段着色器
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, 0);
    glCompileShader(fragmentShader);

    // 创建着色器程序并链接
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);

    // 这里添加顶点属性的绑定
    glBindAttribLocation(shaderProgram, 0, "position");
    glBindAttribLocation(shaderProgram, 1, "normal");

    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);

    // 启用顶点属性数组
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    // 指定顶点属性数据的格式
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (GLvoid*)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (GLvoid*)(3 * sizeof(float)));

    // 释放着色器资源
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}


// 初始化VBO
void MyGLWidget::initVBO() {
    // 创建和绑定 VBO
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
}


void MyGLWidget::initVBOVAO() {
    // 生成并绑定VAO
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // 生成并绑定VBO
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // 生成并绑定EBO
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
}

// 细分三角面片
void MyGLWidget::divide_triangle(vec3 v1, vec3 v2, vec3 v3, int d) {
    if (d > 0) {
        vec3 v_new = normalize(v1 + v2 + v3) ;
        divide_triangle(v1, v2, v_new, d - 1);
        divide_triangle(v1, v3, v_new, d - 1);
        divide_triangle(v2, v3, v_new, d - 1);
    }
    else {
        triangles.push_back(v1);
        triangles.push_back(v2);
        triangles.push_back(v3);
    }
}



void MyGLWidget::Ball(int d) {
    // 定义正四面体的顶点
    glm::vec3 v[4] = {
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(0.0f, 0.942809f, -0.333333f),
        glm::vec3(-0.816497f, -0.471405f, -0.333333f),
        glm::vec3(0.816497f, -0.471405f, -0.333333f)
    };

    // 清空三角形列表
    triangles.clear();

    // 递归细分四面体的每个面
    divide_triangle(v[0], v[1], v[2], d);
    divide_triangle(v[0], v[1], v[3], d);
    divide_triangle(v[0], v[2], v[3], d);
    divide_triangle(v[1], v[2], v[3], d);

    // 生成索引
    for (int i = 0; i < triangles.size(); i += 3) {
        indices.push_back(i);
        indices.push_back(i + 1);
        indices.push_back(i + 2);
    }

    // 处理每个细分后的三角形
    for (int i = 0;i < triangles.size();i++) {
        // 添加顶点位置
        vec3 vertex = normalize(triangles[i]);
        vertexData.push_back(vertex.x);
        vertexData.push_back(vertex.y);
        vertexData.push_back(vertex.z);
        // 计算并添加法线
        vertexData.push_back(vertex.x);
        vertexData.push_back(vertex.y);
        vertexData.push_back(vertex.z);
    }
}


// 获取模型信息
void MyGLWidget::getTriangleData(Triangle triangle)
{
    for (int i = 0; i < 3; ++i) {
        vertexData.push_back(triangle.triangleVertices[i].x);
        vertexData.push_back(triangle.triangleVertices[i].y);
        vertexData.push_back(triangle.triangleVertices[i].z);
        vertexData.push_back(triangle.triangleNormals[i].x);
        vertexData.push_back(triangle.triangleNormals[i].y);
        vertexData.push_back(triangle.triangleNormals[i].z);
    }
}


void MyGLWidget::keyPressEvent(QKeyEvent* e) {

    switch (e->key()) {
    case Qt::Key_0: scene_id = 0; draw_id = 0; update(); break; // 茶壶 + Phong + VBO
    case Qt::Key_1: scene_id = 1; draw_id = 0; update(); break; // 小球 + Phong + VBO
    case Qt::Key_2: scene_id = 2; update(); break;              // 茶壶 + glvertex + GL_SMOOTH
    case Qt::Key_3: scene_id = 3; update(); break;              // 小球 + glvertex + GL_SMOOTH
    case Qt::Key_4: scene_id = 0; draw_id = 4; update(); break; // 茶壶 + Phong + VBO + Index Array
    case Qt::Key_5: scene_id = 1; draw_id = 5; update(); break; // 小球 + Phong + VBO + Index Array
    case Qt::Key_9: degree += 35; update(); break;              // 旋转
    }
}


void MyGLWidget::paintGL()
{
    switch (scene_id) {
    case 0:scene_0(); break;
    case 1:scene_1(); break;
    case 2:scene_2(); break;
    case 3:scene_3(); break;
    }
}


void MyGLWidget::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);
    offset = vec2(width / 2, height / 2);
    update();
}

void MyGLWidget::update_VBO() {
    // 将顶点数据上传到 VBO
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertexData.size(), vertexData.data(), GL_STATIC_DRAW);

    if (4 <= draw_id <= 5) {
        // 填充索引数组
        for (int i = 0; i < objModel.triangleCount; ++i) {
            for (int j = 0; j < 3; ++j) {
                indices.push_back(static_cast<GLuint>(i * 3 + j));
            }
        }
    }
    
    // 上传索引数据到EBO
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);
    
    // 传入着色参数
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "viewMatrix"), 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projMatrix"), 1, GL_FALSE, glm::value_ptr(projMatrix));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "modelMatrix"), 1, GL_FALSE, glm::value_ptr(modelMatrix));
    glUniform3f(glGetUniformLocation(shaderProgram, "lightPosition"), lightPosition.x, lightPosition.y, lightPosition.z);
    glUniform3f(glGetUniformLocation(shaderProgram, "viewPosition"), camPosition.x, camPosition.y, camPosition.z);
    glUniform4f(glGetUniformLocation(shaderProgram, "lightColor"), 0.5, 0.0, 0.8, 1.0);
   
    if (scene_id == 0 || scene_id == 2) {
        glUniform2f(glGetUniformLocation(shaderProgram, "offset"), offset.x, offset.y);
    }
    
    // 定义输入变量格式
    // 绑定VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    
    // 指定顶点坐标的格式
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // 指定法线的格式，假设法线在每个顶点数据的后面，每个法线有3个分量
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    auto draw_start_time = std::chrono::high_resolution_clock::now();
    
    // 执行绘制
    if (draw_id == 0)
        glDrawArrays(GL_TRIANGLES, 0, vertexData.size());
    else
        // 此时使用glDrawElements而不是glDrawArrays
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

    auto draw_end_time = std::chrono::high_resolution_clock::now(); // 结束时间
    // 计算并输出算法的执行时间
    std::chrono::duration<double, std::milli> drawTime = draw_end_time - draw_start_time;
    std::cout << "draw Time: " << drawTime.count() << " ms" << std::endl;
    
}


// 模型：phongshading + VBO
void MyGLWidget::scene_0() {

    switch (draw_id) {
    case 4:
        printf("----------------------Phongshading + VBO + Index Array----------------------------\n");
        initVBOVAO();
        break;
    default:
        printf("----------------------Phongshading + VBO----------------------------\n");
        initVBO();
        break;
    }

    // 初始化
    initShader();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    objModel.loadModel("./objs/teapot_8000.obj");

    // 获取顶点数据
    vertexData.clear();
    for (int i = 0; i < objModel.triangleCount; i++) {
        Triangle nowTriangle = objModel.getTriangleByID(i);
        getTriangleData(nowTriangle);
    }

    // 自主设置变换矩阵
    camPosition = vec3(300 * sin(degree * 3.14 / 180.0) + objModel.centralPoint.y, 400 * cos(degree * 3.14 / 180.0) + objModel.centralPoint.x, 10 + objModel.centralPoint.z);
    camLookAt = objModel.centralPoint;     // 例如，看向物体中心
    camUp = vec3(0, 1, 0);         // 上方向向量
    projMatrix = glm::perspective(radians(20.0f), 1.0f, 0.1f, 2000.0f);
    viewMatrix = glm::lookAt(camPosition, camLookAt, camUp);

    // 单一点光源，可以改为数组实现多光源
    lightPosition = objModel.centralPoint + vec3(0, 200, 200);  
    modelMatrix = glm::mat4(1.0f); // 使用单位矩阵，不进行任何变换

    if (draw_id == 4) {
        // 填充vertices, normals和indices数组
        for (int i = 0; i < objModel.triangleCount; ++i) {
            Triangle triangle = objModel.getTriangleByID(i);
            for (int j = 0; j < 3; ++j) {
                // 假设triangles中的索引是基于1的，我们需要减去1以转换为基于0的索引
                int vertexIndex = objModel.triangles[i][j] - 1;
                triangles_v.push_back(objModel.vertices_data[vertexIndex]);
                triangles_n.push_back(objModel.normals_data[objModel.triangle_normals[i][j] - 1]);
                indices.push_back(static_cast<GLuint>(triangles_v.size() - 1));
            }
        }
    }
    
    auto VBO_start_time = std::chrono::high_resolution_clock::now();     
    update_VBO();
    auto VBO_end_time = std::chrono::high_resolution_clock::now(); // 结束时间
    // 计算并输出算法的执行时间
    std::chrono::duration<double, std::milli> VBOTime = VBO_end_time - VBO_start_time;
    std::cout << "VBO Time: " << VBOTime.count() << " ms" << std::endl;   
    
    // 解绑VBO
    switch (draw_id) {
    case 4:
        // 解绑VBO和EBO
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        break;
    default:
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        break;
    }
}

// 小球：phongshading + VBO
void MyGLWidget::scene_1() {
    // 初始化
    switch (draw_id) {
    case 5:
        printf("----------------------Phongshading + VBO + Index Array----------------------------\n");
        initVBOVAO();
        break;
    default:
        printf("----------------------Phongshading + VBO----------------------------\n");
        initVBO();
        break;
    }

    initShader();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // 获取顶点数据
    vertexData.clear();
    Ball(10);

    //计算摄像机、光照点信息
    float dist = 5.0;
    vec3 centralPoint = vec3(0.0, 0.0, 0.0);
    camPosition = vec3(dist * sin(degree * 3.14 / 180.0) + centralPoint.y, dist * cos(degree * 3.14 / 180.0) + centralPoint.x, 10 + centralPoint.z);
    camLookAt = centralPoint;     // 例如，看向物体中心
    camUp = vec3(0, 1, 0);         // 上方向向量

    lightPosition = centralPoint + vec3(0, 15.0, 15.0);
    projMatrix = glm::perspective(radians(20.0f), 1.0f, 0.1f, 2000.0f);
    viewMatrix = glm::lookAt(camPosition, camLookAt, camUp);
    modelMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.5));

    auto VBO_start_time = std::chrono::high_resolution_clock::now();
    update_VBO();
    auto VBO_end_time = std::chrono::high_resolution_clock::now(); // 结束时间
    std::chrono::duration<double, std::milli> VBOTime = VBO_end_time - VBO_start_time;
    std::cout << "VBO Time: " << VBOTime.count() << " ms" << std::endl;
    
    // 解绑VBO
    switch (draw_id) {
    case 5:
        // 解绑VBO和EBO
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        break;
    default:
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        break;
    }
}

// 模型： glvertex + GL_SMOOTH
void MyGLWidget::scene_2() {

    printf("-----------------------Glvertex + GL_SMOOTH---------------------------\n");
        
    glUseProgram(0);
    // 初始化
    glEnable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glShadeModel(GL_SMOOTH);

    objModel.loadModel("./objs/teapot_8000.obj");

    // 获取顶点数据
    vertexData.clear();
    for (int j = 0; j < objModel.triangleCount; j++) {
        Triangle nowTriangle = objModel.getTriangleByID(j);
        getTriangleData(nowTriangle);
    }
    
    // 计算摄像机、光照点信息
    float dist = 200.0;

    // 自主设置变换矩阵
    camPosition = vec3(100 * sin(degree * 3.14 / 180.0) + objModel.centralPoint.y, 100 * cos(degree * 3.14 / 180.0) + objModel.centralPoint.x, 10 + objModel.centralPoint.z);
    camLookAt = objModel.centralPoint;     // 例如，看向物体中心
    camUp = vec3(0, 1, 0);         // 上方向向量
    // 单一点光源，可以改为数组实现多光源
    lightPosition = objModel.centralPoint + vec3(-100, 0, 20);
    
    // 设置光照颜色
    GLfloat lightColor[] = {1.0f, 1.0f, 1.0f, 1.0f};
    GLfloat lightPosition[] = { objModel.centralPoint.x,   dist + objModel.centralPoint.y, dist + objModel.centralPoint.z };
    // 设置模型本身颜色
    GLfloat modelAmbient[] = { 0.1f, 0.1f, 0.1f, 1.0f }; // 环境光
    GLfloat modelDiffuse[] = { 0.9, 0.9, 0.9, 1.0 };   // 漫反射光
    GLfloat modelSpecular[] = { 0.75f, 0.75f, 0.75f, 1.0f };  // 镜面反射光
    GLfloat modelShininess = 250.0f; // 反光度
    
    auto smooth_start_time = std::chrono::high_resolution_clock::now();
    
    glMaterialfv(GL_FRONT, GL_AMBIENT, modelAmbient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, modelDiffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR, modelSpecular);
    glMaterialf(GL_FRONT, GL_SHININESS, modelShininess);
    
    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightColor);
    glEnable(GL_LIGHTING);//启用光照  
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);//启用颜色追踪  
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
    
    // 设置投影矩阵
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    int width = 100;
    int height = 100;
    int farness = 500;
    glOrtho(-width, width, -height, height, 0.1, farness);

    // 设置视图矩阵
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(camPosition.x, camPosition.y, camPosition.z, camLookAt.x, camLookAt.y, camLookAt.z, camUp.x, camUp.y, camUp.z);
    auto smooth_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> smoothTime = smooth_end_time - smooth_start_time;
    std::cout << "Smooth Time: " << smoothTime.count() << " ms" << std::endl;
    

    auto draw_start_time = std::chrono::high_resolution_clock::now();
    // 执行绘制
    glPushMatrix();
    glBegin(GL_TRIANGLES);
    for (size_t i = 0; i < vertexData.size(); i += 6) {
        glColor3f(0.75f, 0.75f, 0.75f);
        glNormal3f(vertexData[i + 3], vertexData[i + 4], vertexData[i + 5]);
        glVertex3f(vertexData[i], vertexData[i + 1], vertexData[i + 2]);
    }
    glEnd();
    glPopMatrix();
    auto draw_end_time = std::chrono::high_resolution_clock::now(); // 结束时间
    // 计算并输出算法的执行时间
    std::chrono::duration<double, std::milli> drawTime = draw_end_time - draw_start_time;
    std::cout << "draw Time: " << drawTime.count() << " ms" << std::endl;

}


// 小球：glvertex + GL_SMOOTH
void MyGLWidget::scene_3() {

    printf("------------------------Glvertex + GL_SMOOTH--------------------------\n");
    glUseProgram(0);

    // 初始化
    glEnable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glShadeModel(GL_SMOOTH);
    
    vertexData.clear();
    Ball(10);

    //计算摄像机、光照点信息
    float dist = 5.0;
    vec3 centralPoint = vec3(0.0, 0.0, 0.0);
    camPosition = vec3(dist * sin(degree * 3.14 / 180.0) + centralPoint.y, dist * cos(degree * 3.14 / 180.0) + centralPoint.x, dist + centralPoint.z);
    camLookAt = centralPoint;     // 例如，看向物体中心
    camUp = vec3(0, 1, 0);         // 上方向向量

    lightPosition = centralPoint + vec3(0, 15.0, 15.0);
    projMatrix = glm::perspective(radians(20.0f), 1.0f, 0.1f, 2000.0f);
    viewMatrix = glm::lookAt(camPosition, camLookAt, camUp);
    modelMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.5));

    // 设置光照颜色
    GLfloat lightColor[] = {1.0f, 1.0f, 1.0f, 1.0f};
    GLfloat lightPosition[] = { objModel.centralPoint.x,   dist + objModel.centralPoint.y, dist + objModel.centralPoint.z };
    // 设置模型本身颜色
    GLfloat modelAmbient[] = { 0.1f, 0.1f, 0.1f, 1.0f }; // 环境光
    GLfloat modelDiffuse[] = { 0.9, 0.9, 0.9, 1.0 };   // 漫反射光
    GLfloat modelSpecular[] = { 0.75f, 0.75f, 0.75f, 1.0f };  // 镜面反射光
    GLfloat modelShininess = 250.0f; // 反光度

    auto smooth_start_time = std::chrono::high_resolution_clock::now();
    glMaterialfv(GL_FRONT, GL_AMBIENT, modelAmbient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, modelDiffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR, modelSpecular);
    glMaterialf(GL_FRONT, GL_SHININESS, modelShininess);

    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightColor);
    glEnable(GL_LIGHTING);//启用光照  
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);//启用颜色追踪  
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);

    // 设置投影矩阵
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-2, 2, -2, 2, 0.1, 10);
   
    // 设置视图矩阵
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(camPosition.x, camPosition.y, camPosition.z, camLookAt.x, camLookAt.y, camLookAt.z, camUp.x, camUp.y, camUp.z);
    
    auto smooth_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> smoothTime = smooth_end_time - smooth_start_time;
    std::cout << "Smooth Time: " << smoothTime.count() << " ms" << std::endl;

    auto draw_start_time = std::chrono::high_resolution_clock::now(); // 结束时间

    // 执行绘制
    glPushMatrix();
    glBegin(GL_TRIANGLES);
    for (size_t i = 0; i < vertexData.size(); i += 6) {
        glColor3f(0.75f, 0.75f, 0.75f);
        glNormal3f(vertexData[i + 3], vertexData[i + 4], vertexData[i + 5]);
        glVertex3f(vertexData[i], vertexData[i + 1], vertexData[i + 2]);
    }
    glEnd();
    glPopMatrix();

    auto draw_end_time = std::chrono::high_resolution_clock::now(); // 结束时间
    // 计算并输出算法的执行时间
    std::chrono::duration<double, std::milli> drawTime = draw_end_time - draw_start_time;
    std::cout << "draw Time: " << drawTime.count() << " ms" << std::endl;
    
}

