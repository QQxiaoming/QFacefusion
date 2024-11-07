# ifndef FACEENHANCE
# define FACEENHANCE
#include <fstream>
#include <sstream>
#include "opencv2/opencv.hpp"
#include <onnxruntime_cxx_api.h>
#include "onnxbase.h"
#include "utils.h"


class FaceEnhance : public OnnxBase
{
public:
	FaceEnhance(std::string modelpath);
	cv::Mat process(cv::Mat target_img, const std::vector<cv::Point2f> target_landmark_5);
private:
	void preprocess(cv::Mat target_img, const std::vector<cv::Point2f> face_landmark_5, cv::Mat& affine_matrix, cv::Mat& box_mask);
	std::vector<float> input_image;
	int input_height;
	int input_width;
	std::vector<cv::Point2f> normed_template;
    const float FACE_MASK_BLUR = 0.3;
	const int FACE_MASK_PADDING[4] = {0, 0, 0, 0};
};
#endif