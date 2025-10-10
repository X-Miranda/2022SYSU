#include <GL/glew.h>
#include "myglwidget.h"
#include <iostream>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <QRandomGenerator>

double GetCurrentTimeInSeconds() {
    using namespace std::chrono;
    static auto startTime = steady_clock::now();

    return duration_cast<duration<double>>(steady_clock::now() - startTime).count();
}
/*###################################################
##  函数: MyGLWidget
##  函数描述： MyGLWidget类的构造函数，实例化定时器timer
##  参数描述：
##  parent: MyGLWidget的父对象
#####################################################*/

MyGLWidget::MyGLWidget(QWidget* parent)
    :QOpenGLWidget(parent)
{
    timer = new QTimer(this); // 实例化一个定时器
    timer->start(10); // 时间间隔设置为10ms，可以根据需要调整
    connect(timer, SIGNAL(timeout()), this, SLOT(update()));
    setWindowTitle("TuWei Maze");
    this->resize(QSize(1200, 800));
    setMouseTracking(true);
    player = new Player;
    firstMouse = true;
}


/*###################################################
##  函数: ~MyGLWidget
##  函数描述： ~MyGLWidget类的析构函数，删除timer
##  参数描述： 无
#####################################################*/
MyGLWidget::~MyGLWidget()
{
    delete timer;
    delete player;

}


/*###################################################
##  函数: initializeGL
##  函数描述： 初始化绘图参数，如视窗大小、背景色等
##  参数描述： 无
#####################################################*/
void MyGLWidget::initializeGL()
{
    glViewport(0, 0, width(), height());
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glDisable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);
    glEnable(GL_TEXTURE_2D);

    float baseX, baseZ;
    std::fill(std::begin(keyStates), std::end(keyStates), false);
    jumpVelocity = 550.0f; // 例如，设置跳跃的初始速度为20.0f
    moveVelocity = 50.0f;
    jumpInitialHeight = 1.6f * player->maze->wallsize;
    fallInitialHeight = 1.6f * player->maze->wallsize;
    jumpPeakHeight = jumpInitialHeight + 10 * player->maze->wallsize;
    isJumping = 0;
    isFalling = 0;
    isforwarding = 0;
    baseX = (player->maze->yEnd * 2.0f - 0.5f) * player->maze->wallsize;
    baseZ = (player->maze->xEnd * 2.0f - 0.5f) * player->maze->wallsize;
    for (int i = 0; i < 20; ++i)
    {
        snow.push_back(vec3(baseX + QRandomGenerator::global()->bounded(player->maze->wallsize), 0.0f, baseZ + QRandomGenerator::global()->bounded(player->maze->wallsize)));
    }

    
    player->maze->textures.LoadTexture2D("texture/path.bmp");  // 路径纹理
    player->maze->textures.LoadTexture2D("texture/wall.bmp");  // 墙壁纹理 
    glDisable(GL_TEXTURE_2D);

    //天空盒
    glEnable(GL_TEXTURE_CUBE_MAP);
    char skyboxFile[6][100] = {
      "texture/tidepool_R.bmp",
      "texture/tidepool_L.bmp",
      "texture/tidepool_U.bmp",
      "texture/tidepool_D.bmp",
      "texture/tidepool_F.bmp",
      "texture/tidepool_B.bmp"
    };

    /*char skyboxFile[6][100] = {
      "texture/shanmai_R.bmp",
      "texture/shanmai_L.bmp",
      "texture/shanmai_U.bmp",
      "texture/shanmai_D.bmp",
      "texture/shanmai_F.bmp",
      "texture/shanmai_B.bmp"
    };*/

    skyBox.LoadTextureCube(skyboxFile);
    glDisable(GL_TEXTURE_CUBE_MAP);


    /*###########################################################
    ##  功能: Phong 光照模型
    ##  功能描述： 设置物体的材质属性，设置光源，光照模型
    ##  组成部分：
        环境光（Ambient）：全局环境光设置，影响场景中所有物体的基本颜色。
        漫反射（Diffuse）：光线照射到物体上并在多个方向上散射的效果。
        镜面反射（Specular）：模拟光线照射到物体上并在特定方向上反射的效果。
        衰减（Attenuation）：模拟光线随着距离增加而减弱的效果。
        光源位置（Position）：光源在场景中的位置。
    #############################################################*/

    
    glEnable(GL_LIGHTING);
    setDefaultMaterial();
    setLight();

    glEnable(GL_DEPTH_TEST);


}

void MyGLWidget::setDiamondMaterial() {
    GLfloat materialDiamondAmbient[] = { 0.1f, 0.1f, 0.3f, 0.5f };
    GLfloat materialDiamondDiffuse[] = { 0.3f, 0.3f, 0.7f, 0.5f };
    GLfloat materialDiamondSpecular[] = { 0.5f, 0.5f, 0.5f, 0.5f };
    GLfloat shininessDiamond[] = { 28.0f };

    glMaterialfv(GL_FRONT, GL_AMBIENT, materialDiamondAmbient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, materialDiamondDiffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR, materialDiamondSpecular);
    glMaterialfv(GL_FRONT, GL_SHININESS, shininessDiamond);
}

void MyGLWidget::setDiamondMaterial2() {
    GLfloat materialDiamondAmbient[] = { 0.3f, 0.1f, 0.1f, 1.0f };
    GLfloat materialDiamondDiffuse[] = { 0.7f, 0.3f, 0.3f, 1.0f };
    GLfloat materialDiamondSpecular[] = { 0.5f, 0.5f, 0.5f, 1.0f };
    GLfloat shininessDiamond[] = { 28.0f };

    glMaterialfv(GL_FRONT, GL_AMBIENT, materialDiamondAmbient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, materialDiamondDiffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR, materialDiamondSpecular);
    glMaterialfv(GL_FRONT, GL_SHININESS, shininessDiamond);
}

void MyGLWidget::setDefaultMaterial() {
    // 设置物体材质属性
    GLfloat materialAmbient[] = { 1.0f, 1.0f, 1.0f, 1.0f }; // 环境光属性（红，绿，蓝，Alpha）
    GLfloat materialDiffuse[] = { 0.5f, 0.5f, 0.5f, 1.0f }; // 漫反射属性（红，绿，蓝，Alpha）
    GLfloat materialSpecular[] = { 0.8f, 0.8f, 0.8f, 1.0f }; // 镜面反射属性（红，绿，蓝，Alpha）
    GLfloat shininess[] = { 64.0f }; // 光泽度（数值越高，反射越强烈）
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, materialAmbient);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, materialDiffuse);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, materialSpecular);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shininess);

}
void MyGLWidget::setLight() {
    // 设置光源的属性
    GLfloat lightSpecular[] = { 0.7f, 0.7f, 0.8f, 1.0f }; // 镜面反射属性（红，绿，蓝，Alpha）
    GLfloat lightDiffuse[] = { 0.6f, 0.6f, 0.9f, 1.0f }; // 漫反射属性（红，绿，蓝，Alpha）
    GLfloat lightAmbient[] = { 0.2f, 0.2f, 0.4f, 1.0f }; // 环境光属性（红，绿，蓝，Alpha）
    GLfloat lightConstAttenuation[] = { 1.0f }; // 光源恒定衰减系数
    GLfloat lightLinearAttenuation[] = { 0.0f }; // 光源线性衰减系数
    GLfloat lightQuadAttenuation[] = { 0.00f }; // 光源二次方衰减系数
    //GLfloat lightPosition[] = { 1.0f, 1.0f, 1.0f, 0.0f }; // 光源位置（X，Y，Z，W）W=1代表点光源
    GLfloat LightModelAmbient[] = { 0.6f, 0.6f, 0.6f, 1.0 }; // 全局环境光属性（红，绿，蓝，Alpha）
    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
    glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpecular);
    glLightfv(GL_LIGHT0, GL_CONSTANT_ATTENUATION, lightConstAttenuation);
    glLightfv(GL_LIGHT0, GL_LINEAR_ATTENUATION, lightLinearAttenuation);
    glLightfv(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, lightQuadAttenuation);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, LightModelAmbient);

    glEnable(GL_LIGHT0);
}


/*###################################################
##  函数: paintGL
##  函数描述： 绘图函数，实现图形绘制，会被update()函数调用
##  参数描述： 无
#####################################################*/
void MyGLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(player->camera->zoom, width() / height(), 0.1, 60000.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    vec3 target = player->camera->position + player->camera->front;
    gluLookAt(player->camera->position.x, player->camera->position.y, player->camera->position.z,
        target.x, target.y, target.z,
        player->camera->up.x, player->camera->up.y, player->camera->up.z);

    updatespacestate();
    updateJumpState();
    updateFlyState();
    updateFallState();

    // 保存当前矩阵，应用旋转，绘制天空盒
    glPushMatrix();
    glRotatef(skyboxAngle, 0.0f, 1.0f, 0.0f); // 旋转天空盒
    paintSkyBox();

    glPopMatrix();
    paintSkyBox();

    paintMaze();
    paintMark();
    updateLightPosition();
    drawMiniMap(); // 绘制小地图

    if (player->checkWin())
    {
        // 禁用深度测试，以便在屏幕上绘制文本
        glDisable(GL_DEPTH_TEST);

        // 输出“You win”文本
        QPainter painter(this);
        painter.begin(this);
        QPen pen;
        pen.setColor(Qt::white);
        QFont font;
        font.setPointSize(48);
        painter.setFont(font);
        painter.setPen(pen);

        // 将变量转换为字符串并拼接
        // QString text = QString("Win!\n   Total Diamonds: %1").arg(player->money);
        // painter.drawText(width() / 2 - 100, height() / 2, text);
        painter.drawText(width() / 2 - 100, height() / 2, "Win!");
        painter.end();

        glEnable(GL_DEPTH_TEST); // 重新启用深度测试
    }

    if (player->checkMoney()) {

        glDisable(GL_DEPTH_TEST);
        player->money++;
        QPainter painter;
        painter.begin(this);
        QPen pen;
        pen.setColor(Qt::blue);  // 你可以根据需要选择颜色
        QFont font;
        font.setPointSize(30);
        painter.setFont(font);
        painter.setPen(pen);
        if(player->maze->score )
        painter.drawText(width() / 2 - 200, height() / 2, "Find a Diamond");
        painter.end();
        glEnable(GL_DEPTH_TEST);

    }
}




/*###################################################
##  函数: updateLightPosition()
##  函数描述： 实现光源的移动
##  参数描述： 无
#####################################################*/
void MyGLWidget::updateLightPosition() {
    // 示例：让光源绕着某个轴旋转
    float radius = 1000.0f; // 光源旋转半径
    float speed = 2.0f; // 旋转速度
    static float angle = 0.0f; // 初始角度

    // 计算新的光源位置
    lightPosition[0] = radius * cos(angle);
    lightPosition[2] = radius * sin(angle);
    angle += speed * frameTime; // 根据帧时间更新角度

    // 更新光源位置
    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
}


/*###################################################
##  函数: updateJumpState
##  函数描述： 实现跳跃效果
##  参数描述： 无
#####################################################*/
void MyGLWidget::updateJumpState() {
    if (isJumping && !player->isFixed()) {
        float currentTime = GetCurrentTimeInSeconds();
        float timeSinceJump = currentTime - jumpStartTime;

        // 使用简单的物理公式来更新位置
        // 这里可以根据需要调整计算方式
        float jumpHeight = jumpInitialHeight + jumpVelocity * timeSinceJump - 0.5f * gravity * timeSinceJump * timeSinceJump;

        // 计算当前位置对应的块，从而实现 z 轴方向的碰撞检测
        int block_x = int(player->camera->position.z * player->invWallSize + 0.5f);
        int block_y = int(player->camera->position.x * player->invWallSize + 0.5f);
        // 检查是否结束跳跃（踩到当前块的上表面 则 结束跳跃）(碰到迷宫地图上边界也结束跳跃）
        if (jumpHeight <= player->maze->blockHeight[block_x][block_y] + 1.6f * player->maze->wallsize || jumpHeight >= jumpPeakHeight) {
            isJumping = false;
            jumpHeight = player->maze->blockHeight[block_x][block_y] + 1.6f * player->maze->wallsize; // 重置高度，位于当前块的上表面
            jumpInitialHeight = jumpHeight;  // 下次起跳高度为当前块的上表面高度
            fallInitialHeight = jumpHeight;  // 下次坠落高度为当前块的上表面高度
        }
        // 更新摄像机或玩家的垂直位置
        // 假设摄像机或玩家的位置是 camera->position.y
        player->camera->position.y = jumpHeight;
    }
}

/*###################################################
##  函数: updateFlyState
##  函数描述： 实现飞行效果
##  参数描述： 无
#####################################################*/
void MyGLWidget::updateFlyState() {
    if (isFlying)
    {
        float flyheight = 4000;
        player->camera->position.y = flyheight;
    }
}



/*###################################################
##  函数: updateFallState
##  函数描述： 实现自然坠落效果，当没有跳跃，且从矮墙平移到通道时，受重力作用自然下坠
##  参数描述： 无
#####################################################*/
void MyGLWidget::updateFallState() {
    // 计算当前位置对应的块，从而实现 z 轴方向的自然下坠
    int block_x = int(player->camera->position.z * player->invWallSize + 0.5f);
    int block_y = int(player->camera->position.x * player->invWallSize + 0.5f);
    // 当 没有在跳跃 且 没有在下坠 且 当前高度高于当前上表面高度 时，开始下坠
    if (!isJumping && !isFalling && (player->camera->position.y > player->maze->blockHeight[block_x][block_y] + 1.6f * player->maze->wallsize) && !isFlying) {
        isFalling = true; // 开始下坠
        fallStartTime = GetCurrentTimeInSeconds();  // 记录下坠开始时间
    }
    if (isFalling) {
        float currentTime = GetCurrentTimeInSeconds();
        float timeSinceFall = currentTime - fallStartTime;

        // 使用简单的物理公式来更新位置
        // 这里可以根据需要调整计算方式
        float fallHeight = fallInitialHeight - 0.5f * gravity * timeSinceFall * timeSinceFall;

        // 检查是否结束下坠（踩到当前块的上表面）
        if (fallHeight <= player->maze->blockHeight[block_x][block_y] + 1.6f * player->maze->wallsize) {
            isFalling = false;
            fallHeight = player->maze->blockHeight[block_x][block_y] + 1.6f * player->maze->wallsize; // 重置高度
            jumpInitialHeight = fallHeight;  // 下次起跳高度为当前块的上表面高度
            fallInitialHeight = fallHeight;  // 下次坠落高度为当前块的上表面高度
        }
        // 更新摄像机或玩家的垂直位置
        // 假设摄像机或玩家的位置是 camera->position.y
        player->camera->position.y = fallHeight;
    }
}


/*###################################################
##  函数: resizeGL
##  函数描述： 当窗口大小改变时调整视窗尺寸
##  参数描述： 无
#####################################################*/
void MyGLWidget::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);
    update();
}


/*###################################################
##  函数: keyPressEvent/keyRealeaseEvent
##  函数描述： 键盘控制玩家移动
##  参数描述： 无
#####################################################*/

void MyGLWidget::keyPressEvent(QKeyEvent* e)
{
    keyStates[e->key()] = true;
    updatespacestate();
}

void MyGLWidget::keyReleaseEvent(QKeyEvent* event)
{
    keyStates[event->key()] = false;
    updatespacestate();
}

void MyGLWidget::updatespacestate(void)
{
    //Key Control
    if (keyStates[Qt::Key_W])
    {
        player->FBwardMove(forward);
        isforwarding = true;
        forwardStartTime = GetCurrentTimeInSeconds();
    }
    if (keyStates[Qt::Key_S])
        player->FBwardMove(backward);
    if (keyStates[Qt::Key_A])
        player->HorizontalMove(toleft);
    if (keyStates[Qt::Key_D])
        player->HorizontalMove(toright);
    // 按下空格 且 没有在跳跃过程中 且 不在自然坠落过程中 才起跳
    if (keyStates[Qt::Key_Space] && isJumping == false && isFalling == false)
    {
        isJumping = true;
        jumpStartTime = GetCurrentTimeInSeconds();
        forwardinitial = player->posZ;
    }
    // 按下F 且 没有在跳跃过程 中 且 不在自然坠落过程中 才起飞
    if (keyStates[Qt::Key_F] && isJumping == false && isFalling == false)
    {
        isFlying = true;
    }
    if (keyStates[Qt::Key_T] && isJumping == false && isFalling == false)
    {
        isFlying = false;
    }
    if (keyStates[Qt::Key_I])
        player->camera->ProcessMouseMovement(0.0f, 5.0f);
    if (keyStates[Qt::Key_K])
        player->camera->ProcessMouseMovement(0.0f, -5.0f);
    if (keyStates[Qt::Key_J])
        player->camera->ProcessMouseMovement(-5.0f, 0.0f);
    if (keyStates[Qt::Key_L])
        player->camera->ProcessMouseMovement(5.0f, 0.0f);
}



/*###################################################
##  函数: mouseMoveEvent
##  函数描述： 鼠标控制玩家视角移动
##  参数描述： 无
#####################################################*/
void MyGLWidget::mouseMoveEvent(QMouseEvent* e)
{
    float xpos = e->x();
    float ypos = e->y();
    if (firstMouse)
    {
        //捕获鼠标位置
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }
    //获取鼠标位移
    float xOffset = xpos - lastX;
    float yOffset = lastY - ypos;
    //更新鼠标位置
    lastX = xpos;
    lastY = ypos;
    //调用视角控制函数
    player->camera->ProcessMouseMovement(xOffset, yOffset);
    update();
}

/*###################################################
##  函数: paintMaze
##  函数描述： 绘制3D迷宫
##  参数描述： 无
#####################################################*/
void MyGLWidget::paintMaze()
{
    //开启纹理
    glEnable(GL_TEXTURE_2D);
    float baseZ, baseX;
    float wallsize = player->maze->wallsize;
    float dwallsize = wallsize * 6.0f;
    updateCounter = (updateCounter + 1) % 1500;
    /*if (updateCounter % 150 == 0)
        currentFrameForTrap = (currentFrameForTrap + 1) % frameSizeForTrap;
    if (updateCounter % 5 == 0) {
        currentFrameForWall = (currentFrameForWall + 1) % frameSizeForWall;
    }*/
    //绘制
    for (int i = 0; i < player->maze->Height; i++)
    {
        for (int j = 0; j < player->maze->Width; j++)
        {
            //绘制墙
            if (player->maze->maze[i][j] == 1)
            {
                //选取纹理
                glBindTexture(GL_TEXTURE_2D, player->maze->textures.textures2D[1]);

                baseZ = i * 2 * wallsize;
                baseX = j * 2 * wallsize;
                glBegin(GL_QUADS);

                //Front
                glTexCoord2f(0.0f, 0.0f);
                glNormal3f(0.0f, 0.0f, 1.0f);
                glVertex3f(baseX - wallsize, 0.0f, baseZ + wallsize);
                glTexCoord2f(1.0f, 0.0f);
                glNormal3f(0.0f, 0.0f, 1.0f);
                glVertex3f(baseX + wallsize, 0.0f, baseZ + wallsize);
                glTexCoord2f(1.0f, 1.0f);
                glNormal3f(0.0f, 0.0f, 1.0f);
                glVertex3f(baseX + wallsize, dwallsize, baseZ + wallsize);
                glTexCoord2f(0.0f, 1.0f);
                glNormal3f(0.0f, 0.0f, 1.0f);
                glVertex3f(baseX - wallsize, dwallsize, baseZ + wallsize);

                //top
                glTexCoord2f(0.0f, 0.0f);
                glNormal3f(0.0f, 1.0f, 0.0f);
                glVertex3f(baseX - wallsize, dwallsize, baseZ + wallsize);
                glTexCoord2f(1.0f, 0.0f);
                glNormal3f(0.0f, 1.0f, 0.0f);
                glVertex3f(baseX + wallsize, dwallsize, baseZ + wallsize);
                glTexCoord2f(1.0f, 1.0f);
                glNormal3f(0.0f, 1.0f, 0.0f);
                glVertex3f(baseX + wallsize, dwallsize, baseZ - wallsize);
                glTexCoord2f(0.0f, 1.0f);
                glNormal3f(0.0f, 1.0f, 0.0f);
                glVertex3f(baseX - wallsize, dwallsize, baseZ - wallsize);

                //back
                glTexCoord2f(0.0f, 0.0f);
                glNormal3f(0.0f, 0.0f, -1.0f);
                glVertex3f(baseX - wallsize, 0.0f, baseZ - wallsize);
                glTexCoord2f(1.0f, 0.0f);
                glNormal3f(0.0f, 0.0f, -1.0f);
                glVertex3f(baseX + wallsize, 0.0f, baseZ - wallsize);
                glTexCoord2f(1.0f, 1.0f);
                glNormal3f(0.0f, 0.0f, -1.0f);
                glVertex3f(baseX + wallsize, dwallsize, baseZ - wallsize);
                glTexCoord2f(0.0f, 1.0f);
                glNormal3f(0.0f, 0.0f, -1.0f);
                glVertex3f(baseX - wallsize, dwallsize, baseZ - wallsize);
                //left

                glTexCoord2f(0.0f, 0.0f);
                glNormal3f(-1.0f, 0.0f, 0.0f);
                glVertex3f(baseX - wallsize, 0.0f, baseZ - wallsize);
                glTexCoord2f(1.0f, 0.0f);
                glNormal3f(-1.0f, 0.0f, 0.0f);
                glVertex3f(baseX - wallsize, 0.0f, baseZ + wallsize);
                glTexCoord2f(1.0f, 1.0f);
                glNormal3f(-1.0f, 0.0f, 0.0f);
                glVertex3f(baseX - wallsize, dwallsize, baseZ + wallsize);
                glTexCoord2f(0.0f, 1.0f);
                glNormal3f(-1.0f, 0.0f, 0.0f);
                glVertex3f(baseX - wallsize, dwallsize, baseZ - wallsize);
                //right

                glTexCoord2f(0.0f, 0.0f);
                glNormal3f(1.0f, 0.0f, 0.0f);
                glVertex3f(baseX + wallsize, 0.0f, baseZ + wallsize);
                glTexCoord2f(1.0f, 0.0f);
                glNormal3f(1.0f, 0.0f, 0.0f);
                glVertex3f(baseX + wallsize, 0.0f, baseZ - wallsize);
                glTexCoord2f(1.0f, 1.0f);
                glNormal3f(1.0f, 0.0f, 0.0f);
                glVertex3f(baseX + wallsize, dwallsize, baseZ - wallsize);
                glTexCoord2f(0.0f, 1.0f);
                glNormal3f(1.0f, 0.0f, 0.0f);
                glVertex3f(baseX + wallsize, dwallsize, baseZ + wallsize);

                glEnd();
            }

            // 可跃过的矮墙
            else if (player->maze->maze[i][j] == 3) {
                float LowDwallsize = player->maze->blockHeight[i][j];

                //选取纹理
                // glBindTexture(GL_TEXTURE_2D, player->maze->textures.textures2D[3 + currentFrame]);
                glBindTexture(GL_TEXTURE_2D, player->maze->textures.textures2D[0]);
                baseZ = i * 2 * wallsize;
                baseX = j * 2 * wallsize;
                glBegin(GL_QUADS);

                //Front
                glTexCoord2f(0.0f, 0.0f);
                glNormal3f(0.0f, 0.0f, 1.0f);
                glVertex3f(baseX - wallsize, 0.0f, baseZ + wallsize);
                glTexCoord2f(1.0f, 0.0f);
                glNormal3f(0.0f, 0.0f, 1.0f);
                glVertex3f(baseX + wallsize, 0.0f, baseZ + wallsize);
                glTexCoord2f(1.0f, 1.0f);
                glNormal3f(0.0f, 0.0f, 1.0f);
                glVertex3f(baseX + wallsize, LowDwallsize, baseZ + wallsize);
                glTexCoord2f(0.0f, 1.0f);
                glNormal3f(0.0f, 0.0f, 1.0f);
                glVertex3f(baseX - wallsize, LowDwallsize, baseZ + wallsize);

                //top
                glTexCoord2f(0.0f, 0.0f);
                glNormal3f(0.0f, 1.0f, 0.0f);
                glVertex3f(baseX - wallsize, LowDwallsize, baseZ + wallsize);
                glTexCoord2f(1.0f, 0.0f);
                glNormal3f(0.0f, 1.0f, 0.0f);
                glVertex3f(baseX + wallsize, LowDwallsize, baseZ + wallsize);
                glTexCoord2f(1.0f, 1.0f);
                glNormal3f(0.0f, 1.0f, 0.0f);
                glVertex3f(baseX + wallsize, LowDwallsize, baseZ - wallsize);
                glTexCoord2f(0.0f, 1.0f);
                glNormal3f(0.0f, 1.0f, 0.0f);
                glVertex3f(baseX - wallsize, LowDwallsize, baseZ - wallsize);

                //back
                glTexCoord2f(0.0f, 0.0f);
                glNormal3f(0.0f, 0.0f, -1.0f);
                glVertex3f(baseX - wallsize, 0.0f, baseZ - wallsize);
                glTexCoord2f(1.0f, 0.0f);
                glNormal3f(0.0f, 0.0f, -1.0f);
                glVertex3f(baseX + wallsize, 0.0f, baseZ - wallsize);
                glTexCoord2f(1.0f, 1.0f);
                glNormal3f(0.0f, 0.0f, -1.0f);
                glVertex3f(baseX + wallsize, LowDwallsize, baseZ - wallsize);
                glTexCoord2f(0.0f, 1.0f);
                glNormal3f(0.0f, 0.0f, -1.0f);
                glVertex3f(baseX - wallsize, LowDwallsize, baseZ - wallsize);
                //left

                glTexCoord2f(0.0f, 0.0f);
                glNormal3f(-1.0f, 0.0f, 0.0f);
                glVertex3f(baseX - wallsize, 0.0f, baseZ - wallsize);
                glTexCoord2f(1.0f, 0.0f);
                glNormal3f(-1.0f, 0.0f, 0.0f);
                glVertex3f(baseX - wallsize, 0.0f, baseZ + wallsize);
                glTexCoord2f(1.0f, 1.0f);
                glNormal3f(-1.0f, 0.0f, 0.0f);
                glVertex3f(baseX - wallsize, LowDwallsize, baseZ + wallsize);
                glTexCoord2f(0.0f, 1.0f);
                glNormal3f(-1.0f, 0.0f, 0.0f);
                glVertex3f(baseX - wallsize, LowDwallsize, baseZ - wallsize);
                //right

                glTexCoord2f(0.0f, 0.0f);
                glNormal3f(1.0f, 0.0f, 0.0f);
                glVertex3f(baseX + wallsize, 0.0f, baseZ + wallsize);
                glTexCoord2f(1.0f, 0.0f);
                glNormal3f(1.0f, 0.0f, 0.0f);
                glVertex3f(baseX + wallsize, 0.0f, baseZ - wallsize);
                glTexCoord2f(1.0f, 1.0f);
                glNormal3f(1.0f, 0.0f, 0.0f);
                glVertex3f(baseX + wallsize, LowDwallsize, baseZ - wallsize);
                glTexCoord2f(0.0f, 1.0f);
                glNormal3f(1.0f, 0.0f, 0.0f);
                glVertex3f(baseX + wallsize, LowDwallsize, baseZ + wallsize);

                glEnd();
            }

          
            else if (player->maze->maze[i][j] == 0 || player->maze->maze[i][j] == 4)
            {   //绘制通道
                glBindTexture(GL_TEXTURE_2D, player->maze->textures.textures2D[0]);
                baseZ = i * 2 * wallsize;
                baseX = j * 2 * wallsize;
                glBegin(GL_QUADS);
                glTexCoord2f(0.0f, 0.0f);
                glNormal3f(0.0f, 1.0f, 0.0f);
                glVertex3f(baseX - wallsize, 0.0f, baseZ + wallsize);
                glTexCoord2f(1.0f, 0.0f);
                glNormal3f(0.0f, 1.0f, 0.0f);
                glVertex3f(baseX + wallsize, 0.0f, baseZ + wallsize);
                glTexCoord2f(1.0f, 1.0f);
                glNormal3f(0.0f, 1.0f, 0.0f);
                glVertex3f(baseX + wallsize, 0.0f, baseZ - wallsize);
                glTexCoord2f(0.0f, 1.0f);
                glNormal3f(0.0f, 1.0f, 0.0f);
                glVertex3f(baseX - wallsize, 0.0f, baseZ - wallsize);
                glEnd();
            }

            // 绘制金币
            if (player->maze->maze[i][j] == 4) {
                glPushMatrix();
                float baseZ = i * 2 * wallsize;
                float baseX = j * 2 * wallsize;
                glTranslatef(baseX, 0.0f, baseZ);
                // 启用光照
                glEnable(GL_LIGHTING);
                glEnable(GL_LIGHT0);

                // 设置材质属性
                if (player->maze->score[i][j] == 1)
                    setDiamondMaterial();
                else
                    setDiamondMaterial2();
                //setLight();
                const int numVertices = 24; // 8个三角形，每个三角形3个顶点
                float size = 0.1f * wallsize;
                GLfloat vertices[numVertices][3] = {

                    // 顶点连接顺序
                    // 三角形1
                    {size, 0.0 + 2 * wallsize, 0.0},
                    {0.0, size + 2 * wallsize, 0.0},
                    {size, size + 2 * wallsize, -size},
                    // 三角形2
                    {size, 0.0 + 2 * wallsize, 0.0},
                    {0.0, size + 2 * wallsize, 0.0},
                    {size, size + 2 * wallsize, size},
                    // 三角形3
                    {0.0, size + 2 * wallsize, 0.0},
                    {size, 2 * size + 2 * wallsize, 0.0},
                    {size, size + 2 * wallsize, size},
                    // 三角形4
                    {0.0, size + 2 * wallsize, 0.0},
                    {size, 2 * size + 2 * wallsize, 0.0},
                    {size, size + 2 * wallsize, -size},
                    // 三角形5
                    {size, 0.0 + 2 * wallsize, 0.0},
                    {2 * size, size + 2 * wallsize, 0.0},
                    {size, size + 2 * wallsize, -size},
                    // 三角形6
                    {size, 0.0 + 2 * wallsize, 0.0},
                    {2 * size, size + 2 * wallsize, 0.0},
                    {size, size + 2 * wallsize, size},
                    // 三角形7
                    {size, 2 * size + 2 * wallsize, 0.0},
                    {2 * size, size + 2 * wallsize, 0.0},
                    {size, size + 2 * wallsize, size},
                    // 三角形8
                    {size, 2 * size + 2 * wallsize, 0.0},
                    {2 * size, size + 2 * wallsize, 0.0},
                    {size, size + 2 * wallsize, -size}
                };

                // 钻石的颜色（蓝色）
                GLfloat blue[] = { 0.0, 0.0, 1.0, 1.0 };

                glBegin(GL_TRIANGLES);
                for (int i = 0; i < numVertices; i += 3) {
                    glColor4fv(blue); // 设置颜色
                    glVertex3fv(vertices[i]); // 顶点1
                    glVertex3fv(vertices[i + 1]); // 顶点2
                    glVertex3fv(vertices[i + 2]); // 顶点3
                }
                glEnd();

                setDefaultMaterial();
                glPopMatrix();
            }

        }
    }
    glDisable(GL_TEXTURE_2D);
}

/*###################################################
##  函数: paintSkyBox
##  函数描述： 绘制天空盒
##  参数描述： 无
#####################################################*/
void MyGLWidget::paintSkyBox()
{
    //开启立方体贴图
    glEnable(GL_TEXTURE_CUBE_MAP);
    float baseZ, baseX;
    float wallsize = player->maze->wallsize;
    float fwallsize = 20.0f * wallsize * float(player->maze->Width) * 1.5f;
    baseX = 1.0f * float(player->maze->Width) * wallsize;
    baseZ = 1.0f * float(player->maze->Height) * wallsize;
    // 在绘制前保存当前的模型视图矩阵
    glPushMatrix();
    // 将天空盒的中心设置为原点
    glTranslatef(0, 0, 0);
    // 让天空盒绕Y轴旋转（可以根据需要调整旋转角度和速度）
    static float skyboxAngle = 0.0f;
    skyboxAngle += 0.01f; // 每帧增加的角度
    if (skyboxAngle > 360.0f) {
        skyboxAngle -= 360.0f;
    }
    glRotatef(skyboxAngle, 0.0f, 1.0f, 0.0f);
    //绑定纹理
    glBindTexture(GL_TEXTURE_CUBE_MAP, skyBox.texturesCube[0]);
    glBegin(GL_QUADS);
    //right
    glTexCoord3f(1.0f, 1.0f, -1.0f);
    glVertex3f(baseX + fwallsize, -fwallsize, baseZ + fwallsize);
    glTexCoord3f(1.0f, 1.0f, 1.0f);
    glVertex3f(baseX + fwallsize, -fwallsize, baseZ - fwallsize);
    glTexCoord3f(1.0f, -1.0f, 1.0f);
    glVertex3f(baseX + fwallsize, fwallsize, baseZ - fwallsize);
    glTexCoord3f(1.0f, -1.0f, -1.0f);
    glVertex3f(baseX + fwallsize, fwallsize, baseZ + fwallsize);
    //left
    glTexCoord3f(-1.0f, 1.0f, 1.0f);
    glVertex3f(baseX - fwallsize, -fwallsize, baseZ - fwallsize);
    glTexCoord3f(-1.0f, 1.0f, -1.0f);
    glVertex3f(baseX - fwallsize, -fwallsize, baseZ + fwallsize);
    glTexCoord3f(-1.0f, -1.0f, -1.0f);
    glVertex3f(baseX - fwallsize, fwallsize, baseZ + fwallsize);
    glTexCoord3f(-1.0f, -1.0f, 1.0f);
    glVertex3f(baseX - fwallsize, fwallsize, baseZ - fwallsize);
    //top
    glTexCoord3f(-1.0f, 1.0f, 1.0f);
    glVertex3f(baseX - fwallsize, fwallsize, baseZ + fwallsize);
    glTexCoord3f(1.0f, 1.0f, 1.0f);
    glVertex3f(baseX + fwallsize, fwallsize, baseZ + fwallsize);
    glTexCoord3f(1.0f, 1.0f, -1.0f);
    glVertex3f(baseX + fwallsize, fwallsize, baseZ - fwallsize);
    glTexCoord3f(-1.0f, 1.0f, -1.0f);
    glVertex3f(baseX - fwallsize, fwallsize, baseZ - fwallsize);
    //down
    glTexCoord3f(-1.0f, -1.0f, -1.0f);
    glVertex3f(baseX - fwallsize, -fwallsize, baseZ - fwallsize);
    glTexCoord3f(1.0f, -1.0f, -1.0f);
    glVertex3f(baseX + fwallsize, -fwallsize, baseZ - fwallsize);
    glTexCoord3f(1.0f, -1.0f, 1.0f);
    glVertex3f(baseX + fwallsize, -fwallsize, baseZ + fwallsize);
    glTexCoord3f(-1.0f, -1.0f, 1.0f);
    glVertex3f(baseX - fwallsize, -fwallsize, baseZ + fwallsize);


    //front
    glTexCoord3f(1.0f, 1.0f, 1.0f);
    glVertex3f(baseX - fwallsize, -fwallsize, baseZ + fwallsize);
    glTexCoord3f(-1.0f, 1.0f, 1.0f);
    glVertex3f(baseX + fwallsize, -fwallsize, baseZ + fwallsize);
    glTexCoord3f(-1.0f, -1.0f, 1.0f);
    glVertex3f(baseX + fwallsize, fwallsize, baseZ + fwallsize);
    glTexCoord3f(1.0f, -1.0f, 1.0f);
    glVertex3f(baseX - fwallsize, fwallsize, baseZ + fwallsize);
    //back
    glTexCoord3f(-1.0f, 1.0f, -1.0f);
    glVertex3f(baseX + fwallsize, -fwallsize, baseZ - fwallsize);
    glTexCoord3f(1.0f, 1.0f, -1.0f);
    glVertex3f(baseX - fwallsize, -fwallsize, baseZ - fwallsize);
    glTexCoord3f(1.0f, -1.0f, -1.0f);
    glVertex3f(baseX - fwallsize, fwallsize, baseZ - fwallsize);
    glTexCoord3f(-1.0f, -1.0f, -1.0f);
    glVertex3f(baseX + fwallsize, fwallsize, baseZ - fwallsize);
    glEnd();
    glPopMatrix();
    glDisable(GL_TEXTURE_CUBE_MAP);
}

/*###################################################
##  函数: paintMark
##  函数描述： 绘制重点标志
##  参数描述： 无
#####################################################*/
void MyGLWidget::paintMark()
{
    //更改点的大小
    glPointSize(20.0f);
    glBegin(GL_POINTS);
    for (unsigned int i = 0; i < snow.size(); ++i)
    {
        //标记随即升高
        snow[i].y += float(QRandomGenerator::global()->bounded(100));
        //达到高度返回
        if (snow[i].y > 4000.0f)
        {
            snow[i].y = 0.0f;
        }
        else
        {
            //绘制
            glColor3f(1.0f, 1.0f, 1.0f);
            glVertex3f(snow[i].x, snow[i].y, snow[i].z);
        }
    }
    glEnd();
}


/*###################################################
##  函数: drawMiniMap()
##  函数描述： 绘制重点标志
##  参数描述： 无
#####################################################*/

void MyGLWidget::drawMiniMap()
{
    int miniMapSize = 150; // 小地图的大小
    int mapX = width() - miniMapSize - 20; // 小地图的X坐标
    int mapY = 20; // 小地图的Y坐标
    glPushAttrib(GL_ALL_ATTRIB_BITS); // 保存当前OpenGL状态
    glViewport(mapX, mapY, miniMapSize, miniMapSize); // 设置视口为小地图的区域
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, player->maze->Width, 0, player->maze->Height); // 设置正交投影
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // 绘制迷宫
    for (int i = 0; i < player->maze->Height; i++)
    {
        for (int j = 0; j < player->maze->Width; j++)
        {
            float aj = player->maze->Width - j - 1;
            if (player->maze->isWall(i, j))
            {
                glColor3f(1.0f, 0.0f, 0.0f); // 白色表示墙壁
                glRectf(aj, i, aj + 1, i + 1); // 绘制一个单元格表示墙壁
            }
        }
    }

    // 标出玩家位置
    glColor3f(1.0f, 0.0f, 0.0f); // 红色表示玩家位置
    float playerX = player->camera->position.x / 130.0f - 0.6f; // 计算玩家在小地图上的X坐标
    float playerZ = player->camera->position.z / 130.0f + 0.5f; // 计算玩家在小地图上的Y坐标
    //std::cout << playerX << "," << playerZ <<"\n" << endl;
    // 绘制玩家位置标记
    float aX = player->maze->Width - playerX - 1;
    // 增大玩家标记的大小
    glRectf(aX - 0.4f, playerZ - 0.4f, aX + 0.4f, playerZ + 0.4f);


    glPopAttrib(); // 恢复之前保存的OpenGL状态
    glViewport(0, 0, width(), height());
    // 恢复到主视图
}
