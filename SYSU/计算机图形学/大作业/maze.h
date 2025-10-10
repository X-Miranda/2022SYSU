#ifndef MAZE_H
#define MAZE_H
#include <iostream>
#include <stack>
#include <vector>
#include <algorithm>
#include <time.h>
#include "textures.h"
#include <random>
#include <numeric>
#include <cstdlib>
#include <ctime>
/*###################################################
##  ��: Maze
##  ���������Թ���Ϣ
#####################################################*/
const int dir[4][2] = {
    {-1,0}, {0, 1},
    {1, 0}, {0, -1}
};
class Maze
{
public:
    int Height, Width;  //�Թ���С
    int xOrigin, yOrigin;       //���
    int xEnd, yEnd;         //�յ�
    int maze[100][100];     //�Թ�
    float blockHeight[100][100];  //�Թ���ÿ��Ԫ�ؿ�ĸ߶�
    float wallsize;     //ǽ���С
    Textures textures;  //����
    Maze(int height, int width);
    ~Maze();
    bool isArea(int x, int y);
    bool isWall(int x, int y);
    bool isLowWall(int x, int y);
    bool isMoney(int x, int y);
    bool isValid(int x, int y, int z);
    void randGo(int x, int y);

    int score[100][100];

};


#endif // MAZE_H#pragma once
