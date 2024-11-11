#ifndef STYLEGANEXAGE_H
#define STYLEGANEXAGE_H

#include <fstream>
#include <sstream>
#include "opencv2/opencv.hpp"
#include <onnxruntime_cxx_api.h>
#include "onnxbase.h"
#include "utils.h"

class StyleganexAge : public OnnxBase
{
public:
    StyleganexAge(std::string modelpath);
    cv::Mat process(cv::Mat target_img, const std::vector<cv::Point2f> face_landmark_5, float direction);

private:
    void preprocessTarget(cv::Mat srcimg, const std::vector<cv::Point2f> face_landmark_5);
    void preprocessTargetWithBackground(cv::Mat srcimg, const std::vector<cv::Point2f> face_landmark_5);

private:
    std::vector<float> input_image_with_background;
    std::vector<float> input_image;
    std::vector<cv::Point2f> normed_template;
    std::vector<cv::Point2f> normed_background_template;
    const float MODEL_MEAN[3] = {0.5, 0.5, 0.5};
	const float MODEL_STD[3] = {0.5, 0.5, 0.5};
};

#endif // STYLEGANEXAGE_H
