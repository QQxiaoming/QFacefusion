# ifndef DETECT_FACE68LANDMARKS
# define DETECT_FACE68LANDMARKS
#include <fstream>
#include <sstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>
#include "onnxbase.h"
#include "utils.h"

class Face68Landmarks : public OnnxBase
{
public:
	Face68Landmarks(std::string modelpath);
    std::vector<cv::Point2f> detect(cv::Mat srcimg, const FaceFusionUtils::Bbox bounding_box, std::vector<cv::Point2f> &face_landmark_5of68);
private:
    void preprocess(cv::Mat img, const FaceFusionUtils::Bbox bounding_box);
	std::vector<float> input_image;
	int input_height;
	int input_width;
    cv::Mat inv_affine_matrix;
};
#endif
