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
##  类: Maze
##  类描述：迷宫信息
#####################################################*/
const int dir[4][2] = {
    {-1,0}, {0, 1},
    {1, 0}, {0, -1}
};
class Maze
{
public:
    int Height, Width;  //迷宫大小
    int xOrigin, yOrigin;       //起点
    int xEnd, yEnd;         //终点
    int maze[100][100];     //迷宫
    float blockHeight[100][100];  //迷宫中每个元素块的高度
    float wallsize;     //墙体大小
    Textures textures;  //纹理
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
