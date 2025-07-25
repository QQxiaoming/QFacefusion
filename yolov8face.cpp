#include "yolov8face.h"

using namespace cv;
using namespace std;
using namespace Ort;
using namespace FaceFusionUtils;

Yolov8Face::Yolov8Face(string model_path, const float conf_thres, const float iou_thresh) : OnnxBase(model_path)
{
    this->input_height = input_node_dims[0][2];
    this->input_width = input_node_dims[0][3];
    this->conf_threshold = conf_thres;
    this->iou_threshold = iou_thresh;
}

void Yolov8Face::preprocess(Mat srcimg)
{
    const int height = srcimg.rows;
    const int width = srcimg.cols;
    Mat temp_image = srcimg.clone();
    if (height > this->input_height || width > this->input_width)
    {
        const float scale = std::min((float)this->input_height / height, (float)this->input_width / width);
        Size new_size = Size(int(width * scale), int(height * scale));
        resize(srcimg, temp_image, new_size);
    }
    this->ratio_height = (float)height / temp_image.rows;
    this->ratio_width = (float)width / temp_image.cols;
    Mat input_img;
    copyMakeBorder(temp_image, input_img, 0, this->input_height - temp_image.rows, 0, this->input_width - temp_image.cols, BORDER_CONSTANT, 0);

    vector<cv::Mat> bgrChannels(3);
    split(input_img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / 128.0, -127.5 / 128.0);
    }

    const int image_area = this->input_height * this->input_width;
    this->input_image.resize(3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    memcpy(this->input_image.data(), (float *)bgrChannels[0].data, single_chn_size);
    memcpy(this->input_image.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->input_image.data() + image_area * 2, (float *)bgrChannels[2].data, single_chn_size);
}

////只返回检测框,因为在下游的模块里,置信度和5个关键点这两个信息在后续的模块里没有用到
void Yolov8Face::detect(Mat srcimg, std::vector<Bbox> &boxes)
{
    this->preprocess(srcimg);

    std::vector<int64_t> input_img_shape = {1, 3, this->input_height, this->input_width};
    Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_image.data(), this->input_image.size(), input_img_shape.data(), input_img_shape.size());

    Ort::RunOptions runOptions;
    vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), &input_tensor_, 1, this->output_names.data(), output_names.size());

    float *pdata = ort_outputs[0].GetTensorMutableData<float>(); /// 形状是(1, 20, 8400),不考虑第0维batchsize，每一列的长度20,前4个元素是检测框坐标(cx,cy,w,h)，第4个元素是置信度，剩下的15个元素是5个关键点坐标x,y和置信度
    const int num_box = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[2];
    vector<Bbox> bounding_box_raw;
    vector<float> score_raw;
    for (int i = 0; i < num_box; i++)
    {
        const float score = pdata[4 * num_box + i];
        if (score > this->conf_threshold)
        {
            //float cx        = pdata[0*num_box + i]
            //float cy        = pdata[1*num_box + i]
            //float w         = pdata[2*num_box + i]
            //float h         = pdata[3*num_box + i]
            //float score     = pdata[4*num_box + i]
            //float kp1_x     = pdata[5*num_box + i]
            //float kp1_y     = pdata[6*num_box + i]
            //float kp1_score = pdata[7*num_box + i]
            // ...
            float xmin = (pdata[i] - 0.5 * pdata[2 * num_box + i]) * this->ratio_width;            ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float ymin = (pdata[num_box + i] - 0.5 * pdata[3 * num_box + i]) * this->ratio_height; ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float xmax = (pdata[i] + 0.5 * pdata[2 * num_box + i]) * this->ratio_width;            ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float ymax = (pdata[num_box + i] + 0.5 * pdata[3 * num_box + i]) * this->ratio_height; ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            ////坐标的越界检查保护，可以添加一下
            bounding_box_raw.emplace_back(Bbox{xmin, ymin, xmax, ymax, 0});
            score_raw.emplace_back(score);
            /// 剩下的5个关键点坐标的计算,暂时不写,因为在下游的模块里没有用到5个关键点坐标信息
        }
    }
    vector<int> keep_inds = nms(bounding_box_raw, score_raw, this->iou_threshold);
    const int keep_num = keep_inds.size();
    boxes.clear();
    boxes.resize(keep_num);
    for (int i = 0; i < keep_num; i++)
    {
        const int ind = keep_inds[i];
        boxes[i] = bounding_box_raw[ind];
        boxes[i].score = score_raw[ind];
    }
}

void Yolov8Face::detect_with_kp5(Mat srcimg, std::vector<BboxWithKP5> &boxes)
{
    this->preprocess(srcimg);

    std::vector<int64_t> input_img_shape = {1, 3, this->input_height, this->input_width};
    Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_image.data(), this->input_image.size(), input_img_shape.data(), input_img_shape.size());

    Ort::RunOptions runOptions;
    vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), &input_tensor_, 1, this->output_names.data(), output_names.size());

    float *pdata = ort_outputs[0].GetTensorMutableData<float>(); /// 形状是(1, 20, 8400),不考虑第0维batchsize，每一列的长度20,前4个元素是检测框坐标(cx,cy,w,h)，第4个元素是置信度，剩下的15个元素是5个关键点坐标x,y和置信度
    const int num_box = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[2];
    vector<Bbox> bounding_box_raw;
    vector<float> score_raw;
    vector<FaceFusionUtils::KeyPoint> kp[5];
    vector<float> kpscore[5];
    for (int i = 0; i < num_box; i++)
    {
        const float score = pdata[4 * num_box + i];
        if (score > this->conf_threshold)
        {
            //float cx        = pdata[0*num_box + i]
            //float cy        = pdata[1*num_box + i]
            //float w         = pdata[2*num_box + i]
            //float h         = pdata[3*num_box + i]
            //float score     = pdata[4*num_box + i]
            //float kp1_x     = pdata[5*num_box + i]
            //float kp1_y     = pdata[6*num_box + i]
            //float kp1_score = pdata[7*num_box + i]
            // ...
            float xmin = (pdata[i] - 0.5 * pdata[2 * num_box + i]) * this->ratio_width;            ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float ymin = (pdata[num_box + i] - 0.5 * pdata[3 * num_box + i]) * this->ratio_height; ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float xmax = (pdata[i] + 0.5 * pdata[2 * num_box + i]) * this->ratio_width;            ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float ymax = (pdata[num_box + i] + 0.5 * pdata[3 * num_box + i]) * this->ratio_height; ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            ////坐标的越界检查保护，可以添加一下
            bounding_box_raw.emplace_back(Bbox{xmin, ymin, xmax, ymax, 0});
            score_raw.emplace_back(score);
            for(int j=0;j<5;j++) {
                float kpx = pdata[(5+3*j) * num_box + i] * this->ratio_width;
                float kpy = pdata[(6+3*j) * num_box + i] * this->ratio_height;
                float kps = pdata[(7+3*j) * num_box + i];
                kp[j].emplace_back(FaceFusionUtils::KeyPoint{kpx,kpy});
                kpscore[j].emplace_back(kps);
            }
        }
    }
    vector<int> keep_inds = nms(bounding_box_raw, score_raw, this->iou_threshold);
    const int keep_num = keep_inds.size();
    boxes.clear();
    boxes.resize(keep_num);
    for (int i = 0; i < keep_num; i++)
    {
        const int ind = keep_inds[i];
        boxes[i].xmin = bounding_box_raw[ind].xmin;
        boxes[i].ymin = bounding_box_raw[ind].ymin;
        boxes[i].xmax = bounding_box_raw[ind].xmax;
        boxes[i].ymax = bounding_box_raw[ind].ymax;
        boxes[i].score = score_raw[ind];
        for(int j=0;j<5;j++) {
            boxes[i].kp5[j] = kp[j][ind];
        }
    }
}
