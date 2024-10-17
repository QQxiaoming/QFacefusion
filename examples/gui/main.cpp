#include <QApplication>
#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QMessageBox>

#include "qfacefusion_api.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    QMainWindow mainWindow;
    QLabel *srcLabel = new QLabel;
    QLabel *dstLabel = new QLabel;
    QPushButton *srcPushButton = new QPushButton("选择原图");
    QPushButton *dstPushButton = new QPushButton("选择目标图");
    QPushButton *startPushButton = new QPushButton("模型加载中请稍后...");
    startPushButton->setEnabled(false);

    QHBoxLayout *labelLayout = new QHBoxLayout;
    labelLayout->addWidget(srcLabel);
    labelLayout->addWidget(dstLabel);
    QHBoxLayout *pushButtonLayout = new QHBoxLayout;
    pushButtonLayout->addWidget(srcPushButton);
    pushButtonLayout->addWidget(dstPushButton);

    QWidget *labelWidget = new QWidget;
    labelWidget->setLayout(labelLayout);
    QWidget *pushButtonWidget = new QWidget;
    pushButtonWidget->setLayout(pushButtonLayout);

    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(labelWidget);
    mainLayout->addWidget(pushButtonWidget);
    mainLayout->addWidget(startPushButton);
    QWidget *main = new QWidget;
    main->setLayout(mainLayout);

    mainWindow.setCentralWidget(main);

    QFaceFusionThread faceSwapThread(MODEL_PATH);
    faceSwapThread.start();

    QObject::connect(srcPushButton,&QPushButton::clicked, [&](){
        QString srcPath = QFileDialog::getOpenFileName(&mainWindow, "选择源图像", "", "Images (*.png *.jpg *.jpeg *.bmp)");
        if(srcPath.isEmpty()) {
            return;
        }
        srcLabel->setPixmap(QPixmap(srcPath).scaled(256,256,Qt::KeepAspectRatio));
    });
    QObject::connect(dstPushButton,&QPushButton::clicked, [&](){
        QString dstPath = QFileDialog::getOpenFileName(&mainWindow, "选择目标图像", "", "Images (*.png *.jpg *.jpeg *.bmp)");
        if(dstPath.isEmpty()) {
            return;
        }
        dstLabel->setPixmap(QPixmap(dstPath).scaled(256,256,Qt::KeepAspectRatio));
    });
    QObject::connect(startPushButton,&QPushButton::clicked, [&](){
        if(srcLabel->pixmap().isNull() || dstLabel->pixmap().isNull()) {
            QMessageBox::warning(&mainWindow,"警告","未检测到有效的输入图像！");
            return;
        }
        faceSwapThread.setSource(srcLabel->pixmap().toImage());
        faceSwapThread.setTarget(dstLabel->pixmap().toImage());
        startPushButton->setText("处理中...");
        startPushButton->setEnabled(false);
    });
    QObject::connect(&faceSwapThread, &QFaceFusionThread::loadModelState, [&](uint32_t state) {
        if(state) {
            startPushButton->setText("启动");
            startPushButton->setEnabled(true);
        }
    });
    QObject::connect(&faceSwapThread, &QFaceFusionThread::swapFinished,&mainWindow, [&](bool ok, const QImage& target, const QImage& output, const QStringList &args){
        startPushButton->setText("启动");
        startPushButton->setEnabled(true);
        if(ok) {
            QDialog show;
            QLabel *m = new QLabel;
            QPixmap map = QPixmap::fromImage(output);
            m->setPixmap(map);
            QVBoxLayout *l = new QVBoxLayout;
            l->addWidget(m);
            show.setLayout(l);
            show.exec();
        } else {
            QMessageBox::warning(&mainWindow,"警告","无有效目标！");
        }
    }, Qt::BlockingQueuedConnection);

    mainWindow.resize(600,400);
    mainWindow.show();
    return app.exec();
}
