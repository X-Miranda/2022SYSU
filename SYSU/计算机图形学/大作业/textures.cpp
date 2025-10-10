#include "textures.h"
#include <fstream>
/*###################################################
##  函数: Textures
##  函数描述： 初始化纹理类
##  参数描述： 无
#####################################################*/
Textures::Textures()
{
    //二维平面纹理和立方体贴图纹理个数
    count2D = countCube = 0;
}
/*###################################################
##  函数: LoadTexture2D
##  函数描述： 加载2D纹理
##  参数描述： filename 文件名
#####################################################*/
void Textures::LoadTexture2D(const char* filename)
{
    GLint width, height, i;               //位图大小
    GLubyte* image;                    //色素储存地址,中间交换量
    FILE* pf;                             //位图打开地址
    BITMAPFILEHEADER fileHeader;          //位图文件头（包含文件类型，文件信息头大小）
    BITMAPINFOHEADER infoHeader;          //位图信息头（包含位图宽高，色素总量大小）
    fopen_s(&pf, filename, "rb");             //只读方式打开图片文件
    if (pf == nullptr) {
        std::cout << "Failed to read files!" << std::endl;
        return;
    }
    fread(&fileHeader, sizeof(BITMAPFILEHEADER), 1, pf);
    if (fileHeader.bfType != 0x4D42) {
        std::cout << "It is not the bmp file!" << std::endl;
        fclose(pf);
        return;
    }
    fread(&infoHeader, sizeof(BITMAPINFOHEADER), 1, pf);
    width = infoHeader.biWidth;
    height = infoHeader.biHeight;
    if (infoHeader.biSizeImage == 0)                 //计算图片像素总量
        infoHeader.biSizeImage = width * height * 4;
    image = (GLubyte*)malloc(sizeof(GLubyte) * infoHeader.biSizeImage);  //申请空间
    if (image == nullptr) {
        std::cout << "The space is not enough!" << std::endl;
        fclose(pf);
        free(image);
        return;
    }
    fseek(pf, fileHeader.bfOffBits, SEEK_SET);              //将文件读写头移到文件头处
    fread(image, infoHeader.biSizeImage, 1, pf);
    for (i = 0; i < infoHeader.biSizeImage; i += 4) {          //openGL识别的是BGR,所以要置换过来
        std::swap(image[i], image[i + 2]);
    }
    fclose(pf);
    glGenTextures(1, &textures2D[count2D]);
    glBindTexture(GL_TEXTURE_2D, textures2D[count2D]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); //纹理设置，线性滤波
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);     //纹理超出\不足则重复绘制
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    gluBuild2DMipmaps(GL_TEXTURE_2D, 4, width, height, GL_RGBA, GL_UNSIGNED_BYTE, image);
    count2D++;//分配纹理编号
}
/*###################################################
##  函数: LoadTextureCube
##  函数描述： 加载立方体贴图纹理
##  参数描述： filenames 文件名集合
#####################################################*/
void Textures::LoadTextureCube(const char (*filenames)[100])
{
    GLint width, height;
    std::vector<GLubyte> image;
    std::ifstream file;
    BITMAPFILEHEADER fileHeader;
    BITMAPINFOHEADER infoHeader;
    glGenTextures(1, &texturesCube[countCube]);
    glBindTexture(GL_TEXTURE_CUBE_MAP, texturesCube[countCube]);
    countCube++;
    for (int j = 0; j < 6; j++) {
        file.open(filenames[j], std::ios::binary);
        if (!file.is_open()) {
            std::cout << "Failed to read files!" << std::endl;
            return;
        }
        file.read(reinterpret_cast<char*>(&fileHeader), sizeof(BITMAPFILEHEADER));
        if (fileHeader.bfType != 0x4D42) {
            std::cout << "It is not the bmp file!" << std::endl;
            file.close();
            return;
        }
        file.read(reinterpret_cast<char*>(&infoHeader), sizeof(BITMAPINFOHEADER));
        width = infoHeader.biWidth;
        height = infoHeader.biHeight;
        int bytesPerPixel = infoHeader.biBitCount / 8;
        if (infoHeader.biSizeImage == 0) {
            infoHeader.biSizeImage = width * height * bytesPerPixel;
        }
        image.resize(infoHeader.biSizeImage);
        file.seekg(fileHeader.bfOffBits, std::ios::beg);
        file.read(reinterpret_cast<char*>(image.data()), infoHeader.biSizeImage);
        size_t bytesRead = file.gcount();  // 检查实际读取的字节数
        if (bytesRead != infoHeader.biSizeImage) {
            std::cout << "Error: Incomplete file read." << std::endl;
            file.close();
            return;
        }
        // 根据位深度处理像素数据
        if (bytesPerPixel == 3) {
            for (int i = 0; i < infoHeader.biSizeImage; i += 3) {
                std::swap(image[i], image[i + 2]);
            }
        }
        else if (bytesPerPixel == 4) {
            for (int i = 0; i < infoHeader.biSizeImage; i += 4) {
                std::swap(image[i], image[i + 2]);
            }
        }
        file.close();
        // 确保使用正确的纹理创建参数
        if (bytesPerPixel == 3) {
            gluBuild2DMipmaps(GL_TEXTURE_CUBE_MAP_POSITIVE_X + j, GL_RGB, width, height, GL_RGB, GL_UNSIGNED_BYTE, image.data());
        }
        else if (bytesPerPixel == 4) {
            gluBuild2DMipmaps(GL_TEXTURE_CUBE_MAP_POSITIVE_X + j, GL_RGBA, width, height, GL_RGBA, GL_UNSIGNED_BYTE, image.data());
        }
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
}
