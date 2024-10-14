# ifndef FACECLASSIFIER
# define FACECLASSIFIER
#include <fstream>
#include <sstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>
#if defined(COREML_FACEFUSION_BUILD)
#include <coreml_provider_factory.h>
#endif 
#include"utils.h"

class FaceClassifier
{
public:
	enum FaceGender {
		MALE = 0,
		FEMALE = 1,
	};
	enum FaceAge {
		AGE_0_2 = 0,
		AGE_3_9 = 1,
		AGE_10_19 = 2,
		AGE_20_29 = 3,
		AGE_30_39 = 4,
		AGE_40_49 = 5,
		AGE_50_59 = 6,
		AGE_60_69 = 7,
		AGE_70_100 = 8,
	};
	enum FaceRace {
		WHITE = 0,
		BLACK = 1,
		LATINO = 2,
		ASIAN = 3,
		INDIAN = 4,
		ARABIC = 5,
	};

	FaceClassifier(std::string modelpath);
	std::vector<int> detect(cv::Mat srcimg, const std::vector<cv::Point2f> face_landmark_5);
private:
	void preprocess(cv::Mat img, const std::vector<cv::Point2f> face_landmark_5);
	std::vector<float> input_image;
	int input_height;
	int input_width;
    std::vector<cv::Point2f> normed_template;
	const float FAIRFACE_MODEL_MEAN[3] = {0.485, 0.456, 0.406};
	const float FAIRFACE_MODEL_STD[3] = {0.229, 0.224, 0.225};

	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Face Classifier");
	Ort::Session *ort_session = nullptr;
	Ort::SessionOptions sessionOptions = Ort::SessionOptions();
	std::vector<char*> input_names;
	std::vector<char*> output_names;
	std::vector<Ort::AllocatedStringPtr> input_names_ptrs;
    std::vector<Ort::AllocatedStringPtr> output_names_ptrs;
	std::vector<std::vector<int64_t>> input_node_dims; // >=1 outputs
	std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs
	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
};
#endif
