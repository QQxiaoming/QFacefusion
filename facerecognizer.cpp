#include"facerecognizer.h"

using namespace cv;
using namespace std;
using namespace Ort;
using namespace FaceFusionUtils;

FaceEmbdding::FaceEmbdding(string model_path)
{
#if defined(CUDA_FACEFUSION_BUILD)
    try {
        OrtCUDAProviderOptions cuda_options;
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    } catch (const Ort::Exception& e) {
        std::cerr << "Error appending CUDA execution provider: " << e.what() << std::endl;
    }
#endif
#if defined(COREML_FACEFUSION_BUILD)
    OrtSessionOptionsAppendExecutionProvider_CoreML(sessionOptions,COREML_FLAG_ENABLE_ON_SUBGRAPH);
#endif

    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
#if defined(WINDOWS_FACEFUSION_BUILD)
    std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
    ort_session = new Session(env, widestr.c_str(), sessionOptions);
#endif
#if defined(LINUX_FACEFUSION_BUILD) || defined(MACOS_FACEFUSION_BUILD)
    ort_session = new Session(env, model_path.c_str(), sessionOptions);
#endif

    size_t numInputNodes = ort_session->GetInputCount();
    size_t numOutputNodes = ort_session->GetOutputCount();
    AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < numInputNodes; i++)
    {
        input_names_ptrs.push_back(ort_session->GetInputNameAllocated(i, allocator));
        input_names.push_back(input_names_ptrs.back().get());
        Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_node_dims.push_back(input_dims);
    }
    for (size_t i = 0; i < numOutputNodes; i++)
    {
        output_names_ptrs.push_back(ort_session->GetOutputNameAllocated(i, allocator));
        output_names.push_back(output_names_ptrs.back().get());
        Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
    }

    this->input_height = input_node_dims[0][2];
    this->input_width = input_node_dims[0][3];
    ////在这里就直接定义了，没有像python程序里的那样normed_template = TEMPLATES.get(template) * crop_size
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
