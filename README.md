# QFacefusion

QFacefusion是基于opencv和onnxruntime的人脸融合用于Qt项目，其核心代码主要参考[facefusion-onnxrun](https://github.com/hpc203/facefusion-onnxrun)，并在此基础上进行了一些小型修改以及封装。便于直接在项目中引用。

本项目主要依赖opencv和onnxruntime，因此在使用前需要安装这两个库。建议使用opencv-4.10.0和onnxruntime1.18.1版本。源码中包含宏USE_QT_WRAPPER，如果您不希望依赖Qt，可以取消该定义可以轻松使用facefusion_api.h封装的标准接口。如果您使用USE_QT_WRAPPER则可以获得更多适配Qt的API接口。

## API

### 非Qt版本

包含两种使用方式：

- 您可以选择使用class FaceFusion（只加载一次模型），示例如下：

```cpp
FaceFusion *faceFusion = new FaceFusion("/path/MODEL_PATH");
cv::Mat source_img = cv::imread("/path/source_path");
cv::Mat target_img = cv::imread("/path/target_path");
cv::Mat output_img;
faceFusion->runSwap(source_img, target_img, output_img);
```

- 您也可以分别设置source和target，这样可以同一个source进行多次融合，示例如下：

```cpp
FaceFusion *faceFusion = new FaceFusion("/path/MODEL_PATH");
cv::Mat source_img = cv::imread("/path/source_path");
faceFusion->setSource(source_img);
cv::Mat target_img = cv::imread("/path/target_path");
cv::Mat output_img;
faceFusion->runSwap(target_img, output_img);
```

- 您也可以直接使用静态函数接口（每次都调用会加载模型），示例如下：

```cpp
cv::Mat source_img = cv::imread("/path/source_path");
cv::Mat target_img = cv::imread("/path/target_path");
cv::Mat output_img;
FaceFusion::faceSwap(source_img, target_img, output_img);
```

```cpp
FaceFusion::faceSwap("/path/source_path", "/path/target_path", "/path/output_path");
```

### Qt版本

- Qt版本在class FaceFusion上进一步封装，使用QImage和QString作为输入输出，示例如下：

```cpp
QFaceFusion *faceFusion = new QFaceFusion("/path/MODEL_PATH");
QImage source_img("/path/source_path");
QImage target_img("/path/target_path");
QImage output_img;
faceFusion->runSwap(source_img, target_img, output_img);
```

```cpp
QFaceFusion *faceFusion = new QFaceFusion("/path/MODEL_PATH");
QImage source_img("/path/source_path");
faceFusion->setSource(source_img);
QImage target_img("/path/target_path");
QImage output_img;
faceFusion->runSwap(target_img, output_img);
```

```cpp
QImage source_img("/path/source_path");
QImage target_img("/path/target_path");
QImage output_img;
QFaceFusion::faceSwap(source_img, target_img, output_img);
```

```cpp
QFaceFusion::faceSwap("/path/source_path", "/path/target_path", "/path/output_path");
```

- 最后由于onnxruntime的推理时间较长，因此建议使用多线程进行推理，因此封装了QFaceFusionThread便于使用，示例如下：

```cpp
QFaceFusionThread *faceFusionThread = new QFaceFusionThread("/path/MODEL_PATH", this);
connect(faceFusionThread, &QFaceFusionThread::swapFinished, this, [this](bool ok, const QImage& target, const QImage& output){
    if(ok) {
        QLabel *label = new QLabel(this);
        label->setPixmap(QPixmap::fromImage(output));
        label->show();
    }
});
connect(faceFusionThread, &QFaceFusionThread::swapProgress, this, [this](uint64_t progress){
    progressBar->setValue(progress);
});
//启动进程
faceFusionThread->start();

//添加任务，可连续添加，线程后台队列执行，每一帧完成后emit swapFinished
faceFusionThread->setTarget(frameImage[0]);
faceFusionThread->setTarget(frameImage[1]);
faceFusionThread->setTarget(frameImage[2]);
faceFusionThread->setTarget(frameImage[3]);

```

## 配置

- 本项目使用QMake，因此在.pro文件中添加如下配置：

```
include(./QFacefusion/qfacefusion.pri)
```

- pri文件中包含以下宏定义，请根据您的硬件平台选择合适的选项，注释不需要的选项

```
DEFINES += WINDOWS_FACEFUSION_BUILD
DEFINES += LINUX_FACEFUSION_BUILD
DEFINES += MACOS_FACEFUSION_BUILD
DEFINES += CUDA_FACEFUSION_BUILD
DEFINES += COREML_FACEFUSION_BUILD
```

    备注：不使用cuda或coreml加速则会使用cpu进行推理，耗时较大。

## 模型文件

请参考[facefusion-onnxrun](https://github.com/hpc203/facefusion-onnxrun)中提供的模型文件。

```
[md5sum]
2dfan4.onnx             b6d33e0ab221bc9249d558cf0cbe44b0
arcface_w600k_r50.onnx  80248d427976241cbd1343889ed132b3
fairface.onnx           77e3cbd585d748893860df0064d4fa35
gfpgan_1.4.onnx         2f9d93ad985a8f45eb6dc32268a4576d
inswapper_128.onnx      a3a155b90354160350efd66fed6b3d80
inswapper_128_fp16.onnx 9b5b5acbb3023cc4bf7dd831f9854434 (可选)
yoloface_8n.onnx        bcd3728be297428848c809ae9fb4b701
```

## 相比[facefusion-onnxrun](https://github.com/hpc203/facefusion-onnxrun)做的修改点

- 修改只支持高版本onnxruntime，建议大于1.18.1，修复存在的内存泄漏
- API接口增加输入检查对于异常输入正确报错防止发生不可预计错误
- 增加对目标图像多个人脸的支持（包括指定id和设置排序关系、接口setReference设置参考脸、接口setDetect获取人脸信息等工具）
- 增加faceclassifier从[facefusion](https://github.com/facefusion/facefusion)移植，以更好分类目标图像多个人脸
- 增加示例代码，方便参考[examples](./examples)

## 致谢

最后还是要感谢[facefusion-onnxrun](https://github.com/hpc203/facefusion-onnxrun)和[facefusion](https://github.com/facefusion/facefusion)原项目，本项目只是做了些微不足道说明和封装，目的是方便小白使用。
