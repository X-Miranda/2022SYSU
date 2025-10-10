QT += core gui opengl openglwidgets

CONFIG += console qt c++11

DEFINES += QT_DEPRECATED_WARNINGS
INCLUDEPATH += "./glm"

INCLUDEPATH += "D:\homework\cg\glew-2.2.0-win32\glew-2.2.0\include"



LIBS += \
	Glu32.lib \
	OpenGL32.lib


SOURCES += \
    camera.cpp \
    main.cpp \
    maze.cpp \
    myglwidget.cpp \
    player.cpp \
    textures.cpp \
    vec3.cpp

HEADERS += \
    camera.h \
    maze.h \
    myglwidget.h \
    player.h \
    textures.h \
    vec3.h

DISTFILES += \
    blue_bk.bmp \
    blue_dn.bmp \
    blue_ft.bmp \
    blue_lf.bmp \
    blue_rt.bmp \
    blue_up.bmp \
    wall1.bmp \
    wall2.bmp \
    wall3.bmp