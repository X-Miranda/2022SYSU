#ifndef PLAYER_H
#define PLAYER_H

#include "camera.h"
#include "maze.h"



/*###################################################
##  ��: Player
##  �������� ������Ϸ��Ϣ
#####################################################*/
class Player
{

public:
    Maze* maze; //�Թ���ͼ��Ϣ
    Camera* camera; //�����

    float posX, posZ;
    float targetXL, targetXR, targetZF, targetZB;       //�յ㷶Χ
    float invWallSize;  //1/ǽ���С
    bool isJumping = 0;
    float jumpStartTime;
    float currentTime;
    bool isEnded = false;
    int money = 0;
    Player();
    ~Player();
    void FBwardMove(Action action);
    void HorizontalMove(Action action);
    void VerticalMove(Action action);
    bool checkWin();
    //bool checkTrap();
    bool checkMoney();
    void setFixed();
    bool isFixed();
};

#endif // PLAYER_H