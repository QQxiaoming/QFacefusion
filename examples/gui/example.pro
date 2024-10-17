
TARGET_ARCH=$${QT_ARCH}
CONFIG += c++11
DEFINES += QT_DEPRECATED_WARNINGS
QT += core gui widgets

include(../../qfacefusion.pri)

SOURCES += \
    $$PWD/main.cpp
    