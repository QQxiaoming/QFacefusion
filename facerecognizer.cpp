#include"facerecognizer.h"

using namespace cv;
using namespace std;
using namespace Ort;
using namespace FaceFusionUtils;

FaceEmbdding::FaceEmbdding(string model_path) : OnnxBase(model_path)
{
    this->input_height = input_node_dims[0][2];
    this->input_width = input_node_dims[0][3];
    ////在这里就直接定义了，没有像python程序里的那样normed_template = TEMPLATES.get(template) * crop_size
    //	[ 0.34191607, 0.46157411 ]
	//	[ 0.65653393, 0.45983393 ]
	//	[ 0.50022500, 0.64050536 ]
	//	[ 0.37097589, 0.82469196 ]
	//	[ 0.63151696, 0.82325089 ]
    this->normed_template.emplace_back(Point2f(38.29459984, 51.69630032));
    this->normed_template.emplace_back(Point2f(73.53180016, 51.50140016));
    this->normed_template.emplace_back(Point2f(56.0252,     71.73660032));
    this->normed_template.emplace_back(Point2f(41.54929968, 92.36549952));
    this->normed_template.emplace_back(Point2f(70.72989952, 92.20409968));
}

void FaceEmbdding::preprocess(Mat srcimg, const vector<Point2f> face_landmark_5)
{
    Mat crop_img;
    warp_face_by_face_landmark_5(srcimg, crop_img, face_landmark_5, this->normed_template, Size(112, 112));
    /*vector<uchar> inliers(face_landmark_5.size(), 0);
    Mat affine_matrix = cv::estimateAffinePartial2D(face_landmark_5, this->normed_template, cv::noArray(), cv::RANSAC, 100.0);
    Mat crop_img;
    Size crop_size(112, 112);
    warpAffine(srcimg, crop_img, affine_matrix, crop_size, cv::INTER_AREA, cv::BORDER_REPLICATE);*/

    vector<cv::Mat> bgrChannels(3);
    split(crop_img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / 127.5, -1.0);
    }

    const int image_area = this->input_height * this->input_width;
    this->input_image.resize(3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    memcpy(this->input_image.data(), (float *)bgrChannels[2].data, single_chn_size);
    memcpy(this->input_image.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->input_image.data() + image_area * 2, (float *)bgrChannels[0].data, single_chn_size);
}

vector<float> FaceEmbdding::detect(Mat srcimg, const vector<Point2f> face_landmark_5)
{
    this->preprocess(srcimg, face_landmark_5);

    std::vector<int64_t> input_img_shape = {1, 3, this->input_height, this->input_width};
    Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_image.data(), this->input_image.size(), input_img_shape.data(), input_img_shape.size());

    Ort::RunOptions runOptions;
    vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), &input_tensor_, 1, this->output_names.data(), output_names.size());

    float *pdata = ort_outputs[0].GetTensorMutableData<float>(); /// 形状是(1, 512)
    const int len_feature = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];
    vector<float> embedding(len_feature);
    memcpy(embedding.data(), pdata, len_feature*sizeof(float));
    return embedding;
}
