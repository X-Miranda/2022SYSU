#include "player.h"
/*###################################################
##  ����: Player
##  ���������� ��ʼ����Ϸ��Ϣ
##  ���������� ��
#####################################################*/
Player::Player()
{
    //�����Թ�
    maze = new Maze(16, 16);
    //�������������µ������յ���Ϣ
    posX = maze->wallsize * 2.0f * maze->yOrigin;
    posZ = maze->wallsize * 2.0f * maze->xOrigin;
    targetXL = maze->wallsize * (2.0f * maze->yEnd - 1.0f);
    targetXR = maze->wallsize * (2.0f * maze->yEnd + 1.0f);
    targetZF = maze->wallsize * (2.0f * maze->xEnd + 1.0f);
    targetZB = maze->wallsize * (2.0f * maze->xEnd - 1.0f);
    invWallSize = 0.5f / maze->wallsize;
    //��ʼ���������Ϣ
    camera = new Camera(
        //vec3(posX, 1.6f * maze->wallsize, posZ),

        vec3(posX, 0, posZ),
        vec3(0.0f, 0.0f, 1000.0f),
        vec3(-1.0f, 0.0f, 0.0f), 0.0f, 90.0f);
}
/*###################################################
##  ����: ~Player
##  ���������� ��������
##  ���������� ��
#####################################################*/
Player::~Player()
{
    delete maze;
    delete camera;
}
/*###################################################
##  ����: FBwardMove
##  ���������� ���ǰ���ƶ�
##  ���������� action �ƶ�����
#####################################################*/
void Player::FBwardMove(Action action)
{
    //��¼��ǰλ��
    posX = camera->position.x;
    posZ = camera->position.z;
    if (action == forward)
    {
        //��ǰ�ƶ�
        camera->FBwardMove(forward);
    }
    else if (action == backward)
    {
        //����ƶ�
        camera->FBwardMove(backward);

    }
    //���λ�÷Ƿ�����ָ�Ϊ�ƶ�ǰλ��
    if (!maze->isValid(int(camera->position.z * invWallSize + 0.5f), int(camera->position.x * invWallSize + 0.5f), camera->position.y))
    {
        camera->position.x = posX;
        camera->position.z = posZ;
    }
}
/*###################################################
##  ����: HorizontalMove
##  ���������� ��������ƶ�
##  ���������� action �ƶ�����
#####################################################*/
void Player::HorizontalMove(Action action)
{
    //��¼��ǰλ��
    posX = camera->position.x;
    posZ = camera->position.z;
    if (action == toleft)
    {
        //�����ƶ�
        camera->HorizontalMove(toleft);
    }
    else if (action == toright)
    {
        //�����ƶ�
        camera->HorizontalMove(toright);
    }
    //���λ�÷Ƿ�����ָ�Ϊ�ƶ�ǰλ��
    if (!maze->isValid(int(camera->position.z * invWallSize + 0.5f), int(camera->position.x * invWallSize + 0.5f), camera->position.y))
    {
        camera->position.x = posX;
        camera->position.z = posZ;
    }
}

/*###################################################
##  ����: checkWin
##  ���������� �������Ƿ񵽴��յ�
##  ���������� ��
#####################################################*/
bool Player::checkWin()
{
    return (posX >= targetXL) && (posX <= targetXR) && (posZ >= targetZB) && (posZ <= targetZF);
}


bool Player::checkMoney()
{
    // ���㵱ǰλ�ö�Ӧ�Ŀ飬�Ӷ��鿴�Ƿ�Ϊ����
    int block_x = int(camera->position.z * invWallSize + 0.5f);
    int block_y = int(camera->position.x * invWallSize + 0.5f);
    // �����ǰλ�ö�Ӧ�Ŀ�����ʯ, ���жϻ����ʯ
    if (maze->isMoney(block_x, block_y)) maze->score[block_x][block_y] = 0;
    return maze->isMoney(block_x, block_y);
}

/*###################################################
##  ����: setFixed
##  ���������� ������Ҳ��ܶ�
##  ���������� ��
#####################################################*/
void Player::setFixed() {
    isEnded = true;
    this->camera->setFixed();
}

/*###################################################
##  ����: isFixed
##  ���������� �������Ƿ��ܶ�
##  ���������� ��
#####################################################*/
bool Player::isFixed() {
    return isEnded;
}
