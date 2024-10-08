#ifndef _FACEFUSION_API_H_
#define _FACEFUSION_API_H_

#include <string>
#include "opencv2/opencv.hpp"
#include "yolov8face.h"
#include "face68landmarks.h"
#include "facerecognizer.h"
#include "faceswap.h"
#include "faceenhancer.h"

class FaceFusion {
public:
    FaceFusion(const std::string &model_path);
    ~FaceFusion();
    int setSource(const cv::Mat &source_img);
    int runSwap(const cv::Mat &target_img, cv::Mat &output_img, std::function<void(uint64_t)> progress = nullptr);
    int runSwap(const cv::Mat &source_img, const cv::Mat &target_img, cv::Mat &output_img, std::function<void(uint64_t)> progress = nullptr);
    int setDetect(const cv::Mat &source_img, cv::Mat &output_img);

    static int faceSwap(const std::string &source_path, const std::string &target_path, const std::string &output_path);
    static int faceSwap(const cv::Mat &source_img, const cv::Mat &target_img, cv::Mat &output_img);

private:
    std::string m_model_path;
    Yolov8Face *m_detect_face_net = nullptr;
	Face68Landmarks *m_detect_68landmarks_net = nullptr;
	FaceEmbdding *m_face_embedding_net = nullptr;
	SwapFace *m_swap_face_net = nullptr;
	FaceEnhance *m_enhance_face_net = nullptr;

    bool m_source = false;
    std::vector<float> m_source_face_embedding;
};

#ifdef USE_QT_WRAPPER
#include <QImage>
#include <QThread>
#include <QMutex>
#include <QQueue>
#include <QMutexLocker>
#include <QWaitCondition>

class QFaceFusion : public QObject {
    Q_OBJECT
public:
    explicit QFaceFusion(const QString &model_path, QObject *parent = nullptr)
        : QObject(parent) {
        faswap = new FaceFusion(model_path.toStdString());
    }
    ~QFaceFusion() {
        if (faswap) {
            delete faswap;
            faswap = nullptr;
        }
    }
    int setSource(const QImage &source_img) {
        cv::Mat source_mat = to_cvmat(source_img);
        return faswap->setSource(source_mat);
    }
    int runSwap(const QImage &target_img, QImage &output_img, std::function<void(uint64_t)> progress = nullptr) {
        if(progress) progress(1);
        cv::Mat target_mat = to_cvmat(target_img);
        if(progress) progress(2);
        cv::Mat output_mat;
        int ret = faswap->runSwap(target_mat, output_mat, [progress](uint64_t vale) {
            if(progress) progress(2+vale*96/100);
        });
        if(progress) progress(99);
        output_img = to_qimage(output_mat);
        if(progress) progress(100);
        return ret;
    }
    int runSwap(const QImage &source_img, const QImage &target_img, QImage &output_img, std::function<void(uint64_t)> progress = nullptr) {
        if(progress) progress(1);
        cv::Mat source_mat = to_cvmat(source_img);
        if(progress) progress(2);
        cv::Mat target_mat = to_cvmat(target_img);
        if(progress) progress(3);
        cv::Mat output_mat;
        int ret = faswap->runSwap(source_mat, target_mat, output_mat, [progress](uint64_t vale) {
            if(progress) progress(3+vale*95/100);
        });
        if(progress) progress(99);
        output_img = to_qimage(output_mat);
        if(progress) progress(100);
        return ret;
    }
    int setDetect(const QImage &source_img, QImage &output_img) {
        cv::Mat source_mat = to_cvmat(source_img);
        cv::Mat output_mat;
        int ret = faswap->setDetect(source_mat, output_mat);
        output_img = to_qimage(output_mat);
        return ret;
    }
    static cv::Mat to_cvmat(QImage img) {
        img = img.convertToFormat(QImage::Format_RGB888, Qt::ColorOnly).rgbSwapped();
        return cv::Mat(img.height(), img.width(), CV_8UC3, img.bits(), img.bytesPerLine()).clone();
    };
    static QImage to_qimage(cv::Mat img) {
        QImage qImg((const unsigned char*)(img.data), img.cols, img.rows, img.step, QImage::Format_RGB888, nullptr, nullptr);
        return qImg.rgbSwapped();
    };

public:
    static int faceSwap(const QString &source_path, const QString &target_path, const QString &output_path) {
        return FaceFusion::faceSwap(source_path.toStdString(), target_path.toStdString(), output_path.toStdString());
    }
    static int faceSwap(const QImage &source_img, const QImage &target_img, QImage &output_img) {
        cv::Mat source_mat = to_cvmat(source_img);
        cv::Mat target_mat = to_cvmat(target_img);
        cv::Mat output_mat;
        int ret = FaceFusion::faceSwap(source_mat, target_mat, output_mat);
        output_img = to_qimage(output_mat);
        return ret;
    }

private:
    FaceFusion *faswap = nullptr;
};

class QFaceFusionThread : public QThread {
    Q_OBJECT
public:
    enum ImgType {
        source,
        target,
        detect
    };
    struct msg_t {
        ImgType type;
        QImage img;
        QStringList args;
    };
    explicit QFaceFusionThread(const QString &model_path, QObject *parent = nullptr)
        : QThread(parent), modelPath(model_path) {
        exit = false;
    }
    ~QFaceFusionThread() {
        exit = true;
        condition.wakeOne();
        wait();
    }
    void setSource(const QImage& img) {
        QMutexLocker locker(&mutex);
        msg_t msg = { source, img, QStringList()};
        msgList.enqueue(msg);
        condition.wakeOne();
    }
    void setTarget(const QImage& img, const QStringList &args = QStringList()) {
        QMutexLocker locker(&mutex);
        msg_t msg = { target, img, args };
        msgList.enqueue(msg);
        condition.wakeOne();
    }
    void setDetect(const QImage& img, const QStringList &args = QStringList()) {
        QMutexLocker locker(&mutex);
        msg_t msg = { detect, img, args };
        msgList.enqueue(msg);
        condition.wakeOne();
    }
    bool isBusy() {
        QMutexLocker locker(&mutex);
        return !msgList.isEmpty();
    }

signals:
    void swapProgress(uint64_t progress);
    void swapFinished(bool ok, const QImage& target, const QImage& output, const QStringList &args = QStringList());

protected:
    void run() override {
        faswap = new QFaceFusion(modelPath);
        while (!exit) {
            mutex.lock();
            while(msgList.isEmpty()) {
                condition.wait(&mutex);
                if (exit) {
                    mutex.unlock();
                    goto exit;
                }
            }
            if(exit) {
                mutex.unlock();
                goto exit;
            }
            msg_t msg = msgList.dequeue();
            mutex.unlock();
            if (msg.type == source) {
                int ret = faswap->setSource(msg.img);
            } else if (msg.type == target) {
                QImage output;
                emit swapProgress(2);
                int ret = faswap->runSwap(msg.img, output, [this](uint64_t progress) {
                    emit swapProgress(2+progress*97/100);
                });
                emit swapProgress(100);
                if(ret < 0) {
                    emit swapFinished(false, msg.img, msg.img, msg.args);
                } else {
                    emit swapFinished(true, msg.img, output, msg.args);
                }
            } else if (msg.type == detect) {
                QImage output;
                int ret = faswap->setDetect(msg.img, output);
                if(ret < 0) {
                    emit swapFinished(false, msg.img, msg.img, msg.args);
                } else {
                    emit swapFinished(true, msg.img, output, msg.args);
                }
            }
        }
    exit:
        delete faswap;
    }

private:
    QFaceFusion *faswap = nullptr;
    QString modelPath;
    bool exit;
    QMutex mutex;
    QQueue<msg_t> msgList;
    QWaitCondition condition;
};

#endif

#endif /* _FACEFUSION_API_H_ */
