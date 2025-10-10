#include "textures.h"
#include <fstream>
/*###################################################
##  ����: Textures
##  ���������� ��ʼ��������
##  ���������� ��
#####################################################*/
Textures::Textures()
{
    //��άƽ���������������ͼ�������
    count2D = countCube = 0;
}
/*###################################################
##  ����: LoadTexture2D
##  ���������� ����2D����
##  ���������� filename �ļ���
#####################################################*/
void Textures::LoadTexture2D(const char* filename)
{
    GLint width, height, i;               //λͼ��С
    GLubyte* image;                    //ɫ�ش����ַ,�м佻����
    FILE* pf;                             //λͼ�򿪵�ַ
    BITMAPFILEHEADER fileHeader;          //λͼ�ļ�ͷ�������ļ����ͣ��ļ���Ϣͷ��С��
    BITMAPINFOHEADER infoHeader;          //λͼ��Ϣͷ������λͼ��ߣ�ɫ��������С��
    fopen_s(&pf, filename, "rb");             //ֻ����ʽ��ͼƬ�ļ�
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
    if (infoHeader.biSizeImage == 0)                 //����ͼƬ��������
        infoHeader.biSizeImage = width * height * 4;
    image = (GLubyte*)malloc(sizeof(GLubyte) * infoHeader.biSizeImage);  //����ռ�
    if (image == nullptr) {
        std::cout << "The space is not enough!" << std::endl;
        fclose(pf);
        free(image);
        return;
    }
    fseek(pf, fileHeader.bfOffBits, SEEK_SET);              //���ļ���дͷ�Ƶ��ļ�ͷ��
    fread(image, infoHeader.biSizeImage, 1, pf);
    for (i = 0; i < infoHeader.biSizeImage; i += 4) {          //openGLʶ�����BGR,����Ҫ�û�����
        std::swap(image[i], image[i + 2]);
    }
    fclose(pf);
    glGenTextures(1, &textures2D[count2D]);
    glBindTexture(GL_TEXTURE_2D, textures2D[count2D]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); //�������ã������˲�
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);     //������\�������ظ�����
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    gluBuild2DMipmaps(GL_TEXTURE_2D, 4, width, height, GL_RGBA, GL_UNSIGNED_BYTE, image);
    count2D++;//����������
}
/*###################################################
##  ����: LoadTextureCube
##  ���������� ������������ͼ����
##  ���������� filenames �ļ�������
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
        size_t bytesRead = file.gcount();  // ���ʵ�ʶ�ȡ���ֽ���
        if (bytesRead != infoHeader.biSizeImage) {
            std::cout << "Error: Incomplete file read." << std::endl;
            file.close();
            return;
        }
        // ����λ��ȴ�����������
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
        // ȷ��ʹ����ȷ������������
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
