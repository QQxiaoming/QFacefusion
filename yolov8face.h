# ifndef YOLOV8FACE
# define YOLOV8FACE
#include <fstream>
#include <sstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>
#include "onnxbase.h"
#include "utils.h"


class Yolov8Face : public OnnxBase
{
public:
	Yolov8Face(std::string modelpath, const float conf_thres=0.5, const float iou_thresh=0.4);
    void detect(cv::Mat srcimg, std::vector<FaceFusionUtils::Bbox> &boxes);   ////只返回检测框,置信度和5个关键点这两个信息在后续的模块里没有用到
    void detect_with_kp5(cv::Mat srcimg, std::vector<FaceFusionUtils::BboxWithKP5> &boxes);
	void setThreshold(const float conf_thres, const float iou_thresh) {
		this->conf_threshold = conf_thres;
		this->iou_threshold = iou_thresh;
	}
private:
	void preprocess(cv::Mat img);
	std::vector<float> input_image;
	int input_height;
	int input_width;
	float ratio_height;
	float ratio_width;
	float conf_threshold;
	float iou_threshold;
};
#endif
