#include "maze.h"
/*###################################################
##  ����: Maze
##  ���������� ��ʼ���Թ���������������Թ�
##  ���������� height, width �Թ����
#####################################################*/
Maze::Maze(int height, int width)
{
    Height = height;
    Width = width;
    xOrigin = 0;
    yOrigin = 1;
    xEnd = Height - 2;
    yEnd = Width - 1;
    wallsize = 70.0f;
    std::fill(maze[0], maze[0] + Height * Width, 1);
    std::fill(blockHeight[0], blockHeight[0] + Height * Width, 10.0f * wallsize);
    blockHeight[xOrigin][yOrigin] = 0.0f;
    //��������Թ�
    randGo(1, 1);
}
Maze::~Maze()
{

}

/*###################################################
##  ����: isArea
##  ���������� �ж�λ���Ƿ����Թ���
##  ���������� x, y Ŀ��λ��
#####################################################*/
bool Maze::isArea(int x, int y)
{
    return x >= 0 && x < Height && y >= 0 && y < Width;
}

/*###################################################
##  ����: isWall
##  ���������� �ж�λ���Ƿ���ǽ��
##  ���������� x, y Ŀ��λ��
#####################################################*/
bool Maze::isWall(int x, int y)
{
    return isArea(x, y) && maze[x][y] == 1;
}



/*###################################################
##  ����: isLowWall
##  ���������� �ж�λ���Ƿ��ǰ�ǽ
##  ���������� x, y Ŀ��λ��
#####################################################*/
bool Maze::isLowWall(int x, int y)
{
    return isArea(x, y) && maze[x][y] == 3;
}

/*###################################################
##  ����: isMoney
##  ���������� �ж�λ���Ƿ�������
##  ���������� x, y Ŀ��λ��
#####################################################*/
bool Maze::isMoney(int x, int y)
{
    return isArea(x, y) && maze[x][y] == 4;
}

/*###################################################
##  ����: isValid
##  ���������� �ж�λ���Ƿ��ǿ���·��������ͨ������ǽ�Ϸ��������Ϸ���
##  ���������� x, y Ŀ��λ��
#####################################################*/
bool Maze::isValid(int x, int y, int z)
{
    return isArea(x, y) && (maze[x][y] != 1 && z >= 1.6f * wallsize + blockHeight[x][y]);
}

/*###################################################
##  ����: randGo
##  ���������� ����Թ������㷨
##  ���������� x, y �Թ����
#####################################################*/
void Maze::randGo(int x, int y)
{
    //����һ���м�Ķ�ά���鱣���Թ���Ϣ
    int tmpMaze[100][100];
    //���Թ��ı�Ե��ʼ��Ϊ·����ֹ���ֶ������
    //����ط�ȫ����ʼ��Ϊǽ
    for (int i = 0; i < Height + 2; i++)
    {
        for (int j = 0; j < Width + 2; j++)
        {
            tmpMaze[i][j] = 1;
            if (i == 0 || i == Height + 1) tmpMaze[i][j] = 0;
            if (j == 0 || j == Width + 1) tmpMaze[i][j] = 0;
        }
    }

    std::vector<std::pair<int, int>> wall;          //����ǽ����
    wall.push_back(std::pair<int, int>(x + 1, y + 1));  //�������ӽ�ǽ����
    srand(time(NULL));
    while (!wall.empty())    //���ǽ���в�Ϊ��
    {
        //��ǽ���������ѡ��һ��ǽ�����
        int index = rand() % wall.size();
        std::pair<int, int> pos = wall[index];
        //ͳ�Ƹ�ǽ������������·�ĸ���
        int count = 0;
        for (int i = 0; i < 4; i++)
        {
            int newX = pos.first + dir[i][0];
            int newY = pos.second + dir[i][1];
            if (tmpMaze[newX][newY] == 0) count++;
        }
        //�����λ��������������·�ĸ���С��1������Խ���λ����Ϊ·
        //ͬʱ����λ��������������ǽ��λ����ӽ�ǽ����
        if (count <= 1)
        {
            tmpMaze[pos.first][pos.second] = 0;
            for (int i = 0; i < 4; i++)
            {
                int newX = pos.first + dir[i][0];
                int newY = pos.second + dir[i][1];
                if (tmpMaze[newX][newY] != 0) wall.push_back(std::pair<int, int>(newX, newY));

            }
        }
        //����ǽ���ǽ�������Ƴ�
        wall.erase(wall.begin() + index);
    }
    //����tmpMaze��ʵ�ʵ��Թ�����һȦ���õı�Ե
    //���ǽ�tmpMaze��ԵһȦȥ�����õ�ʵ�����ɵ��Թ�
    for (int i = 0; i < Height; i++)
    {
        for (int j = 0; j < Width; j++)
        {
            maze[i][j] = tmpMaze[i + 1][j + 1];
            score[i][j] = 0;
        }
    }


    // �ռ�����ͨ����λ��
    std::vector<std::pair<int, int>> paths;
    for (int i = 0; i < Height; i++) {
        for (int j = 0; j < Width; j++) {
            if (maze[i][j] == 0) {
                paths.push_back(std::make_pair(i, j));
                blockHeight[i][j] = 0.0f;  // ���øÿ�ĸ߶�Ϊ0.0
            }
        }
    }


    for (int i = 0; i < Height * Width / 4; i++) {
        // �������ͨ���������ѡ��һ������Ϊ��ǽ
        if (!paths.empty()) {
            // srand(time(NULL)); // ��ʼ�������������
            int randomIndex = rand() % paths.size(); // ���ѡ��һ������
            std::pair<int, int> trapPos = paths[randomIndex];
            maze[trapPos.first][trapPos.second] = 3; // ����ѡ�е�ͨ��Ϊ��ǽ
            score[trapPos.first][trapPos.second] = 0;
            blockHeight[trapPos.first][trapPos.second] = float(rand() % 4 + 1) / 4 * wallsize * 1.0f;  // ������ɰ�ǽ�ĸ߶�\
                                        }
        }

        for (int i = 0; i < Height * Width / 12; i++) {
            // �������ͨ���������ѡ��һ������Ϊ���
            if (!paths.empty()) {
                int randomIndex = rand() % paths.size(); // ���ѡ��һ������
                std::pair<int, int> trapPos = paths[randomIndex];
                maze[trapPos.first][trapPos.second] = 4; // ����ѡ�е�ͨ��Ϊ���
                score[trapPos.first][trapPos.second] = 1;
                blockHeight[trapPos.first][trapPos.second] = 0.0f;  // ��Ҹ߶�Ϊ0
                break;
            }
        }

        //���������յ�λ��
        maze[xOrigin][yOrigin] = 0;
        for (int i = Height - 1; i >= 0; i--)
        {
            if (maze[i][Width - 2] == 0)
            {
                maze[i][Width - 1] = 0;
                xEnd = i;
                yEnd = Width - 1;
                break;
            }
        }

        maze[xEnd][yEnd] = 0;
        blockHeight[xEnd][yEnd] = 0.0f;
    }
}
