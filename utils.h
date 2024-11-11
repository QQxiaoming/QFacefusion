# ifndef UTILS
# define UTILS
#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include "opencv2/opencv.hpp"

namespace FaceFusionUtils {

typedef struct
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
} Bbox;

typedef struct
{
    float x;
    float y;
} KeyPoint;

typedef struct
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    KeyPoint kp5[5];
} BboxWithKP5;

float GetIoU(const Bbox box1, const Bbox box2);
std::vector<int> nms(std::vector<Bbox> boxes, std::vector<float> confidences, const float nms_thresh);
cv::Mat warp_face_by_face_landmark_5(const cv::Mat temp_vision_frame, cv::Mat &crop_img, const std::vector<cv::Point2f> face_landmark_5, const std::vector<cv::Point2f> normed_template, const cv::Size crop_size);
std::vector<cv::Point2f> scale_face_landmark_5(const std::vector<cv::Point2f> face_landmark_5, const float scale);
cv::Mat create_static_box_mask(const int *crop_size, const float face_mask_blur, const int *face_mask_padding);
cv::Mat paste_back(cv::Mat temp_vision_frame, cv::Mat crop_vision_frame, cv::Mat crop_mask, cv::Mat affine_matrix);
cv::Mat blend_frame(cv::Mat temp_vision_frame, cv::Mat paste_vision_frame, const int FACE_ENHANCER_BLEND=80);
float dot_product(const std::vector<float>& vec1, const std::vector<float>& vec2);
}

#endif