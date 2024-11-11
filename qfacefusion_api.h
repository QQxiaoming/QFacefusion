#ifndef _FACEFUSION_API_H_
#define _FACEFUSION_API_H_

#include <string>
#include "opencv2/opencv.hpp"
#include "yolov8face.h"
#include "face68landmarks.h"
#include "facerecognizer.h"
#include "faceswap.h"
#include "faceenhancer.h"
#include "faceclassifier.h"
#include "styleganexage.h"

class FaceFusion {
public:
    FaceFusion(const std::string &model_path);
    ~FaceFusion();
    int setSource(const cv::Mat &source_img, uint32_t id = 0);
    void clearSource(void);
    int setReference(const cv::Mat &reference_img, uint32_t id = 0);
    void clearReference(void);
    int runSwap(const cv::Mat &target_img, 
                cv::Mat &output_img, 
                uint32_t id = 0, 
                uint32_t order = 0, 
                int multipleFace = 0, 
                int genderMask = 0,
                float similar_thres = 0.4f,
                std::function<void(uint64_t)> progress = nullptr);
    int runSwap(const cv::Mat &source_img, 
                const cv::Mat &target_img, 
                cv::Mat &output_img, 
                uint32_t id = 0, 
                uint32_t order = 0, 
                int multipleFace = 0, 
                int genderMask = 0,
                std::function<void(uint64_t)> progress = nullptr);
    int setDetect(const cv::Mat &source_img, 
                cv::Mat &output_img,
                uint32_t order = 0,
                int genderMask = 0);
    int setAgeModify(const cv::Mat &source_img, cv::Mat &output_img, float direction);
    void setFaceDetectThreshold(const float conf_thres, const float iou_thresh) {
        m_detect_face_net->setThreshold(conf_thres, iou_thresh);
	}
    uint32_t getFindFace(void){
        return m_findFace;
    }
    std::vector<std::vector<float>> getFindSimilarity(void){
        return m_Similarity;
    }
    static int faceSwap(const std::string &source_path, 
                        const std::string &target_path, 
                        const std::string &output_path, 
                        uint32_t id = 0, 
                        uint32_t order = 0, 
                        int multipleFace = 0,
                        int genderMask = 0);
    static int faceSwap(const cv::Mat &source_img, 
                        const cv::Mat &target_img, 
                        cv::Mat &output_img, 
                        uint32_t id = 0, 
                        uint32_t order = 0, 
                        int multipleFace = 0,
                        int genderMask = 0);
    static std::string getModelInfo(std::string model_path);

private:
    template<typename T> static void sortBoxes(std::vector<T> &boxes, uint32_t order);
    FaceClassifier::FaceGender checkGender(const cv::Mat &source_img, const FaceFusionUtils::Bbox &box);

private:
    template <typename T>
    class Lazy {
    public:
        Lazy(std::string path) : instance(nullptr) {
            this->path = path;
        }

        T* getInstance(void) {
            if (!instance) {
                instance = std::make_unique<T>(this->path);
            }
            return instance.get();
        }

        T* operator->() {
            return this->getInstance();
        }

    private:
        std::unique_ptr<T> instance;
        std::string path;
    };
    std::string m_model_path;
    Lazy<Yolov8Face> m_detect_face_net;
    Lazy<Face68Landmarks> m_detect_68landmarks_net;
    Lazy<FaceEmbdding> m_face_embedding_net;
    Lazy<FaceClassifier> m_face_classifier_net;
    Lazy<SwapFace> m_swap_face_net;
    Lazy<FaceEnhance> m_enhance_face_net;
    Lazy<StyleganexAge> m_styleganexage_net;

    bool m_source = false;
    std::vector<std::vector<float>> m_source_face_embedding_arr;
    std::vector<std::vector<float>> m_reference_face_embedding_arr;

    uint32_t m_findFace = 0;
    std::vector<std::vector<float>> m_Similarity;
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
    int setSource(const QImage &source_img, uint32_t id = 0) {
        cv::Mat source_mat = to_cvmat(source_img);
        return faswap->setSource(source_mat, id);
    }
    void clearSource(void) {
        faswap->clearSource();
    }
    int setReference(const QImage &source_img, uint32_t id = 0) {
        cv::Mat source_mat = to_cvmat(source_img);
        return faswap->setReference(source_mat, id);
    }
    void clearReference(void) {
        faswap->clearReference();
    }
    int runSwap(const QImage &target_img, QImage &output_img, 
                uint32_t id = 0, uint32_t order = 0, int multipleFace = 0, int genderMask = 0, float similar_thres = 0.4f,
                std::function<void(uint64_t)> progress = nullptr) {
        if(progress) progress(1);
        cv::Mat target_mat = to_cvmat(target_img);
        if(progress) progress(2);
        cv::Mat output_mat;
        int ret = faswap->runSwap(target_mat, output_mat, id, order, multipleFace, genderMask, similar_thres, [progress](uint64_t vale) {
            if(progress) progress(2+vale*96/100);
        });
        if(progress) progress(99);
        output_img = to_qimage(output_mat);
        if(progress) progress(100);
        return ret;
    }
    int runSwap(const QImage &source_img, const QImage &target_img, QImage &output_img, 
                uint32_t id = 0, uint32_t order = 0, int multipleFace = 0,  int genderMask = 0,
                std::function<void(uint64_t)> progress = nullptr) {
        if(progress) progress(1);
        cv::Mat source_mat = to_cvmat(source_img);
        if(progress) progress(2);
        cv::Mat target_mat = to_cvmat(target_img);
        if(progress) progress(3);
        cv::Mat output_mat;
        int ret = faswap->runSwap(source_mat, target_mat, output_mat, id, order, multipleFace, genderMask, [progress](uint64_t vale) {
            if(progress) progress(3+vale*95/100);
        });
        if(progress) progress(99);
        output_img = to_qimage(output_mat);
        if(progress) progress(100);
        return ret;
    }
    int setDetect(const QImage &source_img, QImage &output_img, uint32_t order = 0, int genderMask = 0) {
        cv::Mat source_mat = to_cvmat(source_img);
        cv::Mat output_mat;
        int ret = faswap->setDetect(source_mat, output_mat, order, genderMask);
        output_img = to_qimage(output_mat);
        return ret;
    }
    int setAgeModify(const QImage &source_img, QImage &output_img, float direction) {
        cv::Mat source_mat = to_cvmat(source_img);
        cv::Mat output_mat = source_mat.clone();
        int ret = -1;
        int number = 1;
        float last = direction;
        bool flag = (direction>=0.0f);
        if(direction/2.5f >= 1.0f) {
            number = direction/2.5f;
            last = direction - number*2.5f;
        } else if(direction/2.5f <= -1.0f) {
            number = -direction/2.5f;
            last = direction + number*2.5f;
        }
        for(int i = 0; i < number; i++) {
            if(i+1 == number) {
                ret = faswap->setAgeModify(output_mat, output_mat, last);
            } else {
                if(flag) {
                    ret = faswap->setAgeModify(output_mat, output_mat, 2.5f);
                } else {
                    ret = faswap->setAgeModify(output_mat, output_mat, -2.5f);
                }
            }
        }
        output_img = to_qimage(output_mat);
        return ret;
    }
    void setFaceDetectThreshold(const float conf_thres, const float iou_thresh) {
        faswap->setFaceDetectThreshold(conf_thres, iou_thresh);
    }
    uint32_t getFindFace(void){
        return faswap->getFindFace();
    }
    QList<QList<float>> getFindSimilarity(void){
        std::vector<std::vector<float>> similarity = faswap->getFindSimilarity();
        QList<QList<float>> list;
        for(auto &sim : similarity) {
            QList<float> l;
            for(auto &s : sim) {
                l.append(s);
            }
            list.append(l);
        }
        return list;
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
    static int faceSwap(const QString &source_path, const QString &target_path, const QString &output_path, 
            uint32_t id = 0, uint32_t order = 0, int multipleFace = 0, int genderMask = 0) {
        return FaceFusion::faceSwap(source_path.toStdString(), 
                                    target_path.toStdString(), 
                                    output_path.toStdString(), 
                                    id, order, multipleFace, genderMask);
    }
    static int faceSwap(const QImage &source_img, const QImage &target_img, QImage &output_img, 
            uint32_t id = 0, uint32_t order = 0, int multipleFace = 0, int genderMask = 0) {
        cv::Mat source_mat = to_cvmat(source_img);
        cv::Mat target_mat = to_cvmat(target_img);
        cv::Mat output_mat;
        int ret = FaceFusion::faceSwap(source_mat, target_mat, output_mat, 
                                    id, order, multipleFace, genderMask);
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
        detect,
        reference,
        agemodify,
    };
    struct msg_t {
        ImgType type;
        QImage img;
        QStringList args;
        float conf_thres;
        float iou_thresh;
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
    void setSource(const QImage& img, uint32_t id = 0, const float conf_thres = 0.5f, const float iou_thresh = 0.4f) {
        QStringList args;
        args.append(QString::number(id));
        QMutexLocker locker(&mutex);
        msg_t msg = { source, img, args, conf_thres, iou_thresh};
        msgList.enqueue(msg);
        condition.wakeOne();
    }
    void clearSource(void) {
        QStringList args;
        args.append(QString::number(-1));
        QMutexLocker locker(&mutex);
        msg_t msg = { source, QImage(), args, 0.5f, 0.4f};
        msgList.enqueue(msg);
        condition.wakeOne();
    }
    void setReference(const QImage& img, uint32_t id = 0, const float conf_thres = 0.5f, const float iou_thresh = 0.4f) {
        QStringList args;
        args.append(QString::number(id));
        QMutexLocker locker(&mutex);
        msg_t msg = { reference, img, args, conf_thres, iou_thresh};
        msgList.enqueue(msg);
        condition.wakeOne();
    }
    void clearReference(void) {
        QStringList args;
        args.append(QString::number(-1));
        QMutexLocker locker(&mutex);
        msg_t msg = { reference, QImage(), args, 0.5f, 0.4f};
        msgList.enqueue(msg);
        condition.wakeOne();
    }
    void setTarget(const QImage& img, const QStringList &args = QStringList(), const float conf_thres = 0.5f, const float iou_thresh = 0.4f) {
        QMutexLocker locker(&mutex);
        msg_t msg = { target, img, args, conf_thres, iou_thresh };
        msgList.enqueue(msg);
        condition.wakeOne();
    }
    void setDetect(const QImage& img, const QStringList &args = QStringList(), const float conf_thres = 0.5f, const float iou_thresh = 0.4f) {
        QMutexLocker locker(&mutex);
        msg_t msg = { detect, img, args, conf_thres, iou_thresh };
        msgList.enqueue(msg);
        condition.wakeOne();
    }
    int setAgeModify(const QImage& img, float direction, const QStringList &args = QStringList(), const float conf_thres = 0.5f, const float iou_thresh = 0.4f) {
        QMutexLocker locker(&mutex);
        msg_t msg = { agemodify, img, args, conf_thres, iou_thresh };
        msgList.enqueue(msg);
        m_direction = direction;
        condition.wakeOne();
        return 0;
    }
    bool isBusy() {
        QMutexLocker locker(&mutex);
        return !msgList.isEmpty();
    }
    int64_t currentProgress() {
        QMutexLocker locker(&mutex);
        return msgList.count();
    }
    void popOldProgress(int64_t countThres) {
        QMutexLocker locker(&mutex);
        if(msgList.count() <= countThres) {
            return;
        } else {
            while(msgList.count() > countThres) {
                msgList.dequeue();
            }
        }
    }
    void clearProgress(void) {
        QMutexLocker locker(&mutex);
        msgList.clear();
    }
    void setMultipleFaceMode(int multipleFace) {
        m_multipleFace = multipleFace;
    }
    void setTargetFaceId(uint32_t id) {
        m_targetFaceId = id;
    }
    void setTargetFaceOrder(uint32_t order) {
        m_targetFaceOrder = order;
    }
    void setGenderMask(int mask) {
        m_genderMask = mask;
    }
    void setSimilarityThreshold(float thres) {
        m_similar_thres = thres;
    }

signals:
    void loadModelState(uint32_t state);
    void swapProgress(uint64_t progress);
    void swapFinished(bool ok, 
                      const QImage& target, const QImage& output, 
                      const QStringList &args = QStringList(),
                      uint32_t findFace = 0, QList<QList<float>> findSimilarity = QList<QList<float>>());

protected:
    void run() override {
        emit loadModelState(0);
        faswap = new QFaceFusion(modelPath);
        emit loadModelState(1);
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
            faswap->setFaceDetectThreshold(msg.conf_thres,msg.iou_thresh);
            if (msg.type == source) {
                int32_t id = msg.args.isEmpty() ? 0 : msg.args[0].toInt();
                if(id < 0) {
                    faswap->clearSource();
                } else {
                    faswap->setSource(msg.img,id);
                }
            } else if (msg.type == reference) {
                int32_t id = msg.args.isEmpty() ? 0 : msg.args[0].toInt();
                if(id < 0) {
                    faswap->clearReference();
                } else {
                    faswap->setReference(msg.img,id);
                }
            } else if (msg.type == target) {
                QImage output;
                emit swapProgress(2);
                int ret = faswap->runSwap(msg.img, output, 
                    m_targetFaceId, m_targetFaceOrder, m_multipleFace, m_genderMask, m_similar_thres,
                    [this](uint64_t progress) {
                    emit swapProgress(2+progress*97/100);
                });
                emit swapProgress(100);
                if(ret < 0) {
                    emit swapFinished(false, msg.img, msg.img, msg.args);
                } else {
                    uint32_t findFace = faswap->getFindFace();
                    QList<QList<float>> findSimilarity = faswap->getFindSimilarity();
                    emit swapFinished(true, msg.img, output, msg.args, findFace, findSimilarity);
                }
            } else if (msg.type == detect) {
                QImage output;
                emit swapProgress(2);
                int ret = faswap->setDetect(msg.img, output, m_targetFaceOrder, m_genderMask);
                emit swapProgress(100);
                if(ret < 0) {
                    emit swapFinished(false, msg.img, msg.img, msg.args);
                } else {
                    emit swapFinished(true, msg.img, output, msg.args);
                }
            } else if (msg.type == agemodify) {
                QImage output;
                emit swapProgress(2);
                int ret = faswap->setAgeModify(msg.img, output, m_direction);
                emit swapProgress(100);
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
    int m_multipleFace = 0;
    uint32_t m_targetFaceId = 0;
    uint32_t m_targetFaceOrder = 0;
    int m_genderMask = 0;
    float m_similar_thres = 0.4f;
    float m_direction = 0.0f;
};

#endif

#endif /* _FACEFUSION_API_H_ */
