SOURCES += \
    $$PWD/face68landmarks.cpp \
    $$PWD/faceenhancer.cpp \
    $$PWD/facerecognizer.cpp \
    $$PWD/faceclassifier.cpp \
    $$PWD/faceswap.cpp \
    $$PWD/yolov8face.cpp \
    $$PWD/utils.cpp \
    $$PWD/qfacefusion_api.cpp

HEADERS += \
    $$PWD/face68landmarks.h \
    $$PWD/faceenhancer.h \
    $$PWD/facerecognizer.h \
    $$PWD/faceclassifier.h \
    $$PWD/faceswap.h \
    $$PWD/yolov8face.h \
    $$PWD/model_matrix.bin.h \
    $$PWD/utils.h \
    $$PWD/qfacefusion_api.h

INCLUDEPATH += $$PWD

DEFINES += USE_QT_WRAPPER
DEFINES += MODEL_PATH=\\\"$$PWD/model\\\"

win32:!wasm {
    DEFINES += WINDOWS_FACEFUSION_BUILD
    DEFINES += CUDA_FACEFUSION_BUILD
        
    OPENCV_DIR=$$PWD/../../depend/opencv/build/install
    ONNXRUNTIME_DIR=$$PWD/../../depend/onnxruntime-win-x64-gpu-1.18.1
    
    INCLUDEPATH += -I $${OPENCV_DIR}/include/opencv4
    DEPENDPATH += $${OPENCV_DIR}/include/opencv4
    INCLUDEPATH += -I $${OPENCV_DIR}/include
    DEPENDPATH += $${OPENCV_DIR}/include
    LIBS += -L$${OPENCV_DIR}/x64/vc17/lib \
        -lopencv_calib3d4100 \
        -lopencv_core4100 \
        -lopencv_dnn4100 \
        -lopencv_features2d4100 \
        -lopencv_flann4100 \
        -lopencv_gapi4100 \
        -lopencv_highgui4100 \
        -lopencv_imgcodecs4100 \
        -lopencv_imgproc4100 \
        -lopencv_ml4100 \
        -lopencv_objdetect4100 \
        -lopencv_photo4100 \
        -lopencv_stitching4100 \
        -lopencv_video4100 \
        -lopencv_videoio4100
        
    INCLUDEPATH += -I $${ONNXRUNTIME_DIR}/include
    DEPENDPATH +=$${ONNXRUNTIME_DIR}/include
    LIBS += -L$${ONNXRUNTIME_DIR}/lib/ -lonnxruntime
}

unix:!macx:!android:!ios:!wasm {
    DEFINES += LINUX_FACEFUSION_BUILD
    #DEFINES += CUDA_FACEFUSION_BUILD

    OPENCV_DIR=/usr
    ONNXRUNTIME_DIR=$$PWD/../../depend/onnxruntime-linux-x64-1.19.2

    INCLUDEPATH += -I $${OPENCV_DIR}/include/opencv4
    DEPENDPATH += $${OPENCV_DIR}/include/opencv4
    INCLUDEPATH += -I $${OPENCV_DIR}/include
    DEPENDPATH += $${OPENCV_DIR}/include
    LIBS += -L$${OPENCV_DIR}/lib/ \
        -lopencv_stitching \
        -lopencv_highgui \
        -lopencv_video \
        -lopencv_dnn \
        -lopencv_objdetect \
        -lopencv_calib3d \
        -lopencv_imgcodecs \
        -lopencv_features2d \
        -lopencv_flann \
        -lopencv_photo \
        -lopencv_imgproc \
        -lopencv_core \
        -lopencv_videoio

    INCLUDEPATH += -I $${ONNXRUNTIME_DIR}/include
    DEPENDPATH +=$${ONNXRUNTIME_DIR}/include
    LIBS += -L$${ONNXRUNTIME_DIR}/lib -lonnxruntime

}

macx:!ios:!wasm {
    DEFINES += MACOS_FACEFUSION_BUILD
    DEFINES += COREML_FACEFUSION_BUILD

    OPENCV_DIR=/usr/local
    ONNXRUNTIME_DIR=$$PWD/../../depend/onnxruntime-osx-arm64-1.19.2

    INCLUDEPATH += -I $${OPENCV_DIR}/include/opencv4
    DEPENDPATH += $${OPENCV_DIR}/include/opencv4
    INCLUDEPATH += -I $${OPENCV_DIR}/include
    DEPENDPATH += $${OPENCV_DIR}/include
    LIBS += -L$${OPENCV_DIR}/lib/ \
        -lopencv_stitching \
        -lopencv_highgui \
        -lopencv_video \
        -lopencv_dnn \
        -lopencv_objdetect \
        -lopencv_calib3d \
        -lopencv_imgcodecs \
        -lopencv_features2d \
        -lopencv_flann \
        -lopencv_photo \
        -lopencv_imgproc \
        -lopencv_core \
        -lopencv_videoio

    INCLUDEPATH += -I $${ONNXRUNTIME_DIR}/include
    DEPENDPATH +=$${ONNXRUNTIME_DIR}/include
    LIBS += -L$${ONNXRUNTIME_DIR}/lib/ -lonnxruntime
}

android { 

}

ios {

}
