# ifndef FACESWAP
# define FACESWAP
#include <fstream>
#include <sstream>
#include "opencv2/opencv.hpp"
#include <onnxruntime_cxx_api.h>
#include "onnxbase.h"
#include "utils.h"


class SwapFace : public OnnxBase
{
public:
	SwapFace(std::string modelpath);
	cv::Mat process(cv::Mat target_img, const std::vector<float> source_face_embedding, const std::vector<cv::Point2f> target_landmark_5);
	~SwapFace();  // 析构函数, 释放内存
private:
	void preprocess(cv::Mat target_img, const std::vector<cv::Point2f> face_landmark_5, const std::vector<float> source_face_embedding, cv::Mat& affine_matrix, cv::Mat& box_mask);
	std::vector<float> input_image;
	std::vector<float> input_embedding;
	int input_height;
	int input_width;
	const int len_feature = 512;
	std::vector<cv::Point2f> normed_template;
	const float FACE_MASK_BLUR = 0.3;
	const int FACE_MASK_PADDING[4] = {0, 0, 0, 0};
	const float INSWAPPER_128_MODEL_MEAN[3] = {0.0, 0.0, 0.0};
	const float INSWAPPER_128_MODEL_STD[3] = {1.0, 1.0, 1.0};
};
#endif
