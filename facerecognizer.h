# ifndef FACERECOGNIZER
# define FACERECOGNIZER
#include <fstream>
#include <sstream>
#include <onnxruntime_cxx_api.h>
#include "onnxbase.h"
#include "utils.h"


class FaceEmbdding : public OnnxBase
{
public:
	FaceEmbdding(std::string modelpath);
	std::vector<float> detect(cv::Mat srcimg, const std::vector<cv::Point2f> face_landmark_5);
private:
	void preprocess(cv::Mat img, const std::vector<cv::Point2f> face_landmark_5);
	std::vector<float> input_image;
	int input_height;
	int input_width;
    std::vector<cv::Point2f> normed_template;
};
#endif