
TARGET_ARCH=$${QT_ARCH}
CONFIG += c++11 cmdline
DEFINES += QT_DEPRECATED_WARNINGS
QT += core

include(../../qfacefusion.pri)

SOURCES += \
    $$PWD/main.cpp
    
