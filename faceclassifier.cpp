#include"faceclassifier.h"

using namespace cv;
using namespace std;
using namespace Ort;
using namespace FaceFusionUtils;

FaceClassifier::FaceClassifier(string model_path) : OnnxBase(model_path)
{
    this->input_height = input_node_dims[0][2];
    this->input_width = input_node_dims[0][3];
    ////在这里就直接定义了，没有像python程序里的那样normed_template = TEMPLATES.get(template) * crop_size
    //	[ 0.34191607, 0.46157411 ]
	//	[ 0.65653393, 0.45983393 ]
	//	[ 0.50022500, 0.64050536 ]
	//	[ 0.37097589, 0.82469196 ]
	//	[ 0.63151696, 0.82325089 ]
    this->normed_template.emplace_back(Point2f(0.34191607*224.0, 0.46157411*224.0));
    this->normed_template.emplace_back(Point2f(0.65653393*224.0, 0.45983393*224.0));
    this->normed_template.emplace_back(Point2f(0.50022500*224.0, 0.64050536*224.0));
    this->normed_template.emplace_back(Point2f(0.37097589*224.0, 0.82469196*224.0));
    this->normed_template.emplace_back(Point2f(0.63151696*224.0, 0.82325089*224.0));
}

void FaceClassifier::preprocess(Mat srcimg, const vector<Point2f> face_landmark_5)
{
    Mat crop_img;
    warp_face_by_face_landmark_5(srcimg, crop_img, face_landmark_5, this->normed_template, Size(224, 224));
    
    vector<cv::Mat> bgrChannels(3);
    split(crop_img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / (255.0*this->FAIRFACE_MODEL_STD[c]), -this->FAIRFACE_MODEL_MEAN[c]/this->FAIRFACE_MODEL_STD[c]);
    }

    const int image_area = this->input_height * this->input_width;
    this->input_image.resize(3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    memcpy(this->input_image.data(), (float *)bgrChannels[2].data, single_chn_size);
    memcpy(this->input_image.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->input_image.data() + image_area * 2, (float *)bgrChannels[0].data, single_chn_size);
}

vector<int> FaceClassifier::detect(Mat srcimg, const vector<Point2f> face_landmark_5)
{
    this->preprocess(srcimg, face_landmark_5);

    std::vector<int64_t> input_img_shape = {1, 3, this->input_height, this->input_width};
    Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_image.data(), this->input_image.size(), input_img_shape.data(), input_img_shape.size());

    Ort::RunOptions runOptions;
    vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), &input_tensor_, 1, this->output_names.data(), output_names.size());

    vector<int> face_classifier_id;
    face_classifier_id.push_back(*ort_outputs[0].GetTensorMutableData<int>()); //Race
    face_classifier_id.push_back(*ort_outputs[1].GetTensorMutableData<int>()); //Gender
    face_classifier_id.push_back(*ort_outputs[2].GetTensorMutableData<int>()); //Age

    return face_classifier_id;
}
