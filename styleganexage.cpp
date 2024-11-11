#include "styleganexage.h"

using namespace cv;
using namespace std;
using namespace Ort;
using namespace FaceFusionUtils;

StyleganexAge::StyleganexAge(std::string modelpath) : OnnxBase(modelpath) 
{
    ////在这里就直接定义了，没有像python程序里的那样normed_template = TEMPLATES.get(template) * crop_size
	// 0.37691676, 0.46864664
	// 0.62285697, 0.46912813
	// 0.50123859, 0.61331904
	// 0.39308822, 0.72541100
	// 0.61150205, 0.72490465
    this->normed_template.emplace_back(Point2f(0.37691676*256.0, 0.46864664*256.0));
    this->normed_template.emplace_back(Point2f(0.62285697*256.0, 0.46912813*256.0));
    this->normed_template.emplace_back(Point2f(0.50123859*256.0, 0.61331904*256.0));
    this->normed_template.emplace_back(Point2f(0.39308822*256.0, 0.72541100*256.0));
    this->normed_template.emplace_back(Point2f(0.61150205*256.0, 0.72490465*256.0));
    this->normed_background_template.emplace_back(Point2f(0.37691676*512.0, 0.46864664*512.0));
    this->normed_background_template.emplace_back(Point2f(0.62285697*512.0, 0.46912813*512.0));
    this->normed_background_template.emplace_back(Point2f(0.50123859*512.0, 0.61331904*512.0));
    this->normed_background_template.emplace_back(Point2f(0.39308822*512.0, 0.72541100*512.0));
    this->normed_background_template.emplace_back(Point2f(0.61150205*512.0, 0.72490465*512.0));
}

void StyleganexAge::preprocessTarget(Mat srcimg, const vector<Point2f> face_landmark_5)
{
    Mat crop_img;
    warp_face_by_face_landmark_5(srcimg, crop_img, face_landmark_5, this->normed_template, Size(256, 256));
    
    vector<cv::Mat> bgrChannels(3);
    split(crop_img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / (255.0*this->MODEL_STD[c]), -this->MODEL_MEAN[c]/this->MODEL_STD[c]);
    }

    const int image_area = 256 * 256;
    this->input_image.resize(3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    memcpy(this->input_image.data(), (float *)bgrChannels[2].data, single_chn_size);
    memcpy(this->input_image.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->input_image.data() + image_area * 2, (float *)bgrChannels[0].data, single_chn_size);
}


void StyleganexAge::preprocessTargetWithBackground(Mat srcimg, const vector<Point2f> face_landmark_5)
{
    Mat crop_img;
    warp_face_by_face_landmark_5(srcimg, crop_img, face_landmark_5, this->normed_background_template, Size(512, 512));

    vector<cv::Mat> bgrChannels(3);
    split(crop_img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / (255.0*this->MODEL_STD[c]), -this->MODEL_MEAN[c]/this->MODEL_STD[c]);
    }

    const int image_area = 512 * 512;
    this->input_image_with_background.resize(3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    memcpy(this->input_image_with_background.data(), (float *)bgrChannels[2].data, single_chn_size);
    memcpy(this->input_image_with_background.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->input_image_with_background.data() + image_area * 2, (float *)bgrChannels[0].data, single_chn_size);
}

Mat StyleganexAge::process(Mat target_img, const vector<Point2f> face_landmark_5, float direction) //	[ 2.5, -2.5 ]
{
    this->preprocessTarget(target_img,face_landmark_5);
    this->preprocessTargetWithBackground(target_img,scale_face_landmark_5(face_landmark_5,2.0));

    std::vector<Ort::Value> inputs_tensor;
    std::vector<int64_t> input_with_background_img_shape = {1, 3, 512, 512};
    std::vector<int64_t> input_img_shape = {1, 3, 256, 256};
    int64_t direction_shape = 1;
    inputs_tensor.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, this->input_image_with_background.data(), this->input_image_with_background.size(), input_with_background_img_shape.data(), input_with_background_img_shape.size()));
    inputs_tensor.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, this->input_image.data(), this->input_image.size(), input_img_shape.data(), input_img_shape.size()));
    inputs_tensor.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, &direction, 1, &direction_shape, 1));
    
    Ort::RunOptions runOptions;
    std::vector<Ort::Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), inputs_tensor.data(), inputs_tensor.size(), this->output_names.data(), output_names.size());

    float* pdata = ort_outputs[0].GetTensorMutableData<float>();
    std::vector<int64_t> outs_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const int out_h = outs_shape[2];
    const int out_w = outs_shape[3];
    const int channel_step = out_h * out_w;
    cv::Mat rmat(out_h, out_w, CV_32FC1, pdata);
    cv::Mat gmat(out_h, out_w, CV_32FC1, pdata + channel_step);
    cv::Mat bmat(out_h, out_w, CV_32FC1, pdata + 2 * channel_step);
	rmat += 1;
	gmat += 1;
	bmat += 1;
    rmat *= 255.f/2.0f;
    gmat *= 255.f/2.0f;
    bmat *= 255.f/2.0f;
    rmat.setTo(0, rmat < 0);
    rmat.setTo(255, rmat > 255);
    gmat.setTo(0, gmat < 0);
    gmat.setTo(255, gmat > 255);
    bmat.setTo(0, bmat < 0);
    bmat.setTo(255, bmat > 255);

    std::vector<cv::Mat> channel_mats(3);
    channel_mats[0] = bmat;
    channel_mats[1] = gmat;
    channel_mats[2] = rmat;
    cv::Mat result;
    merge(channel_mats, result);
    return result;
}
