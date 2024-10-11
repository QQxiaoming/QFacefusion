#include "qfacefusion_api.h"

using namespace cv;
using namespace std;
using namespace FaceFusionUtils;

FaceFusion::FaceFusion(const std::string &model_path) 
	: m_model_path(model_path) {
	m_detect_face_net = new Yolov8Face(m_model_path+"/yoloface_8n.onnx");
	m_detect_68landmarks_net = new Face68Landmarks(m_model_path+"/2dfan4.onnx");
	m_face_embedding_net = new FaceEmbdding(m_model_path+"/arcface_w600k_r50.onnx");
	m_swap_face_net = new SwapFace(m_model_path+"/inswapper_128.onnx");
	m_enhance_face_net = new FaceEnhance(m_model_path+"/gfpgan_1.4.onnx");
}

FaceFusion::~FaceFusion() {
	delete m_detect_face_net;
	delete m_detect_68landmarks_net;
	delete m_face_embedding_net;
	delete m_swap_face_net;
	delete m_enhance_face_net;
}

template<typename T> void FaceFusion::sortBoxes(std::vector<T> &boxes, uint32_t order) {
	//根据order来排序
	// order = 0: 左右排序
	// order = 1: 右左排序
	// order = 2: 上下排序
	// order = 3: 下上排序
	if(order == 0) {
		sort(boxes.begin(), boxes.end(), [](const T &a, const T &b) {
			return a.xmin < b.xmin;
		});
	} else if(order == 1) {
		sort(boxes.begin(), boxes.end(), [](const T &a, const T &b) {
			return a.xmin > b.xmin;
		});
	} else if(order == 2) {
		sort(boxes.begin(), boxes.end(), [](const T &a, const T &b) {
			return a.ymin < b.ymin;
		});
	} else if(order == 3) {
		sort(boxes.begin(), boxes.end(), [](const T &a, const T &b) {
			return a.ymin > b.ymin;
		});
	}
}

int FaceFusion::runSwap(const cv::Mat &source_img, const cv::Mat &target_img, cv::Mat &output_img,
			uint32_t id, uint32_t order, bool multipleFace, std::function<void(uint64_t)> progress) {
	if(source_img.empty() || target_img.empty()){
        return -1;
    }
	if (source_img.channels() != 3 || target_img.channels() != 3) {
        return -1;
    }
	vector<Bbox> boxes;
    if(progress) progress(1);
    m_detect_face_net->detect(source_img, boxes);
	if(boxes.empty()) {
		return -1;
	}
    if(progress) progress(10);
	int position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
	vector<Point2f> face_landmark_5of68;
    if(progress) progress(20);
    vector<Point2f> face68landmarks = m_detect_68landmarks_net->detect(source_img, boxes[position], face_landmark_5of68);
    if(progress) progress(30);
    vector<float> source_face_embedding = m_face_embedding_net->detect(source_img, face_landmark_5of68);

    if(progress) progress(40);
    m_detect_face_net->detect(target_img, boxes);
	if(boxes.empty()) {
		return -1;
	}
	sortBoxes(boxes, order);
	if(multipleFace) {
    	if(progress) progress(50);
		output_img = target_img;
        for (size_t i = 0; i < boxes.size(); i++) {
			if(progress) progress(50+i*50/boxes.size());
			vector<Point2f> target_landmark_5;
            m_detect_68landmarks_net->detect(output_img, boxes[i], target_landmark_5);

			if(progress) progress(50+i*50/boxes.size()+10/boxes.size());
			Mat swapimg = m_swap_face_net->process(output_img, source_face_embedding, target_landmark_5);
			if(progress) progress(50+i*50/boxes.size()+40/boxes.size());
			output_img = m_enhance_face_net->process(swapimg, target_landmark_5);
			if(progress) progress(50+i*50/boxes.size()+50/boxes.size());
		}
		if(progress) progress(100);
	} else {
    	if(progress) progress(50);
		int position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
		if(id <= boxes.size()) position = id;
		vector<Point2f> target_landmark_5;
		m_detect_68landmarks_net->detect(target_img, boxes[position], target_landmark_5);

		if(progress) progress(60);
		Mat swapimg = m_swap_face_net->process(target_img, source_face_embedding, target_landmark_5);
		if(progress) progress(80);
		output_img = m_enhance_face_net->process(swapimg, target_landmark_5);
		if(progress) progress(100);
	}
	
    return 0;
}

void FaceFusion::clearSource(void) {
	m_source_face_embedding_arr.clear();
}

int FaceFusion::setSource(const cv::Mat &source_img, uint32_t id) {
	if(m_source_face_embedding_arr.size() < id) {
        return -1;
	}
	if(source_img.empty()){
        return -1;
    }
	if (source_img.channels() != 3) {
        return -1;
    }
	vector<Bbox> boxes;
    m_detect_face_net->detect(source_img, boxes);
	if(boxes.empty()) {
		return -1;
	}
	int position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
	vector<Point2f> face_landmark_5of68;
    m_detect_68landmarks_net->detect(source_img, boxes[position], face_landmark_5of68);
    vector<float> source_face_embedding = m_face_embedding_net->detect(source_img, face_landmark_5of68);
	if(m_source_face_embedding_arr.size() == id) {
		m_source_face_embedding_arr.push_back(source_face_embedding);
	} else {
		m_source_face_embedding_arr[id] = source_face_embedding;
	}
    return 0;
}

int FaceFusion::runSwap(const cv::Mat &target_img, cv::Mat &output_img,
			uint32_t id, uint32_t order, bool multipleFace, std::function<void(uint64_t)> progress) {
	if(m_source_face_embedding_arr.empty()){
		return -1;
	}
    if(target_img.empty()){
        return -1;
    }
	if (target_img.channels() != 3) {
        return -1;
    }
	
    if(progress) progress(1);
	vector<Bbox> boxes;
    m_detect_face_net->detect(target_img, boxes);
	if(boxes.empty()) {
		return -1;
	}
	sortBoxes(boxes, order);
	if(multipleFace) {
    	if(progress) progress(30);
		output_img = target_img;
        for (size_t i = 0; i < boxes.size(); i++) {
			if(progress) progress(30+i*70/boxes.size());
			vector<Point2f> target_landmark_5;
            m_detect_68landmarks_net->detect(output_img, boxes[i], target_landmark_5);

			if(progress) progress(30+i*70/boxes.size()+20/boxes.size());
			uint32_t source_id = i % m_source_face_embedding_arr.size();
            Mat swapimg = m_swap_face_net->process(output_img, m_source_face_embedding_arr[source_id], target_landmark_5);
			if(progress) progress(30+i*70/boxes.size()+50/boxes.size());
			output_img = m_enhance_face_net->process(swapimg, target_landmark_5);
			if(progress) progress(30+i*70/boxes.size()+70/boxes.size());
		}
		if(progress) progress(100);
	} else {
    	if(progress) progress(30);
		int position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
		if(id <= boxes.size()) position = id;
		vector<Point2f> target_landmark_5;
		m_detect_68landmarks_net->detect(target_img, boxes[position], target_landmark_5);

		if(progress) progress(50);
        Mat swapimg = m_swap_face_net->process(target_img, m_source_face_embedding_arr[0], target_landmark_5);
		if(progress) progress(80);
		output_img = m_enhance_face_net->process(swapimg, target_landmark_5);
		if(progress) progress(100);
	}
	
	return 0;
}

int FaceFusion::setDetect(const cv::Mat &source_img, cv::Mat &output_img, uint32_t order) {
	if(source_img.empty()){
        return -1;
    }
	if (source_img.channels() != 3) {
        return -1;
    }
	vector<BboxWithKP5> boxes;
    m_detect_face_net->detect_with_kp5(source_img, boxes);
	sortBoxes(boxes, order);
	cv::Mat temp_vision_frame = source_img.clone();
    for (size_t i = 0; i < boxes.size(); i++){
		cv::rectangle(temp_vision_frame, cv::Point(boxes[i].xmin, boxes[i].ymin), cv::Point(boxes[i].xmax, boxes[i].ymax), cv::Scalar(0, 255, 0), 2);
		for (int j = 0; j < 5; j++){
			cv::circle(temp_vision_frame, cv::Point(boxes[i].kp5[j].x, boxes[i].kp5[j].y), 2, cv::Scalar(0, 255, 0), 2);
		}
		cv::Point point = cv::Point(boxes[i].xmin, boxes[i].ymin);
		if (point.y < 3) point.y += 3; else point.y -= 3;
		cv::putText(temp_vision_frame, std::to_string(i), point, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
	}
	output_img = temp_vision_frame;
    return 0;
}

int FaceFusion::faceSwap(const string &source_path, 
						 const string &target_path, 
						 const string &output_path, 
						 uint32_t id, uint32_t order, bool multipleFace) {
	Mat source_img = imread(source_path);
	Mat target_img = imread(target_path);
	if(source_img.empty() || target_img.empty()){
        return -1;
    }
	if (source_img.channels() != 3 || target_img.channels() != 3) {
        return -1;
    }

	////图片路径和onnx文件的路径，要确保写正确，才能使程序正常运行的
	string root_mod_path = MODEL_PATH;
	Yolov8Face detect_face_net(root_mod_path+"/yoloface_8n.onnx");
	Face68Landmarks detect_68landmarks_net(root_mod_path+"/2dfan4.onnx");
	FaceEmbdding face_embedding_net(root_mod_path+"/arcface_w600k_r50.onnx");
	SwapFace swap_face_net(root_mod_path+"/inswapper_128.onnx");
	FaceEnhance enhance_face_net(root_mod_path+"/gfpgan_1.4.onnx");

    vector<Bbox> boxes;
	detect_face_net.detect(source_img, boxes);
	if(boxes.empty()) {
		return -1;
	}
	int position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
	vector<Point2f> face_landmark_5of68;
	vector<Point2f> face68landmarks = detect_68landmarks_net.detect(source_img, boxes[position], face_landmark_5of68);
	vector<float> source_face_embedding = face_embedding_net.detect(source_img, face_landmark_5of68);

	detect_face_net.detect(target_img, boxes);
	if(boxes.empty()) {
		return -1;
	}
	sortBoxes(boxes, order);
	Mat resultimg = target_img;
	if(multipleFace) {
        for (size_t i = 0; i < boxes.size(); i++) {
			vector<Point2f> target_landmark_5;
            detect_68landmarks_net.detect(resultimg, boxes[i], target_landmark_5);

			Mat swapimg = swap_face_net.process(resultimg, source_face_embedding, target_landmark_5);
			resultimg = enhance_face_net.process(swapimg, target_landmark_5);
		}
	} else {
		position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
		if(id <= boxes.size()) position = id;
		vector<Point2f> target_landmark_5;
		detect_68landmarks_net.detect(target_img, boxes[position], target_landmark_5);

		Mat swapimg = swap_face_net.process(target_img, source_face_embedding, target_landmark_5);
		resultimg = enhance_face_net.process(swapimg, target_landmark_5);
	}
	
	imwrite(output_path, resultimg);

    return 0;
}

int FaceFusion::faceSwap(const Mat &source_img, 
						 const Mat &target_img, 
						 Mat &output_img, 
						 uint32_t id, uint32_t order, bool multipleFace) {
	if(source_img.empty() || target_img.empty()){
        return -1;
    }
	if (source_img.channels() != 3 || target_img.channels() != 3) {
        return -1;
    }
	////图片路径和onnx文件的路径，要确保写正确，才能使程序正常运行的
	string root_mod_path = MODEL_PATH;
	Yolov8Face detect_face_net(root_mod_path+"/yoloface_8n.onnx");
	Face68Landmarks detect_68landmarks_net(root_mod_path+"/2dfan4.onnx");
	FaceEmbdding face_embedding_net(root_mod_path+"/arcface_w600k_r50.onnx");
	SwapFace swap_face_net(root_mod_path+"/inswapper_128.onnx");
	FaceEnhance enhance_face_net(root_mod_path+"/gfpgan_1.4.onnx");

    vector<Bbox> boxes;
	detect_face_net.detect(source_img, boxes);
	if(boxes.empty()) {
		return -1;
	}
	int position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
	vector<Point2f> face_landmark_5of68;
	vector<Point2f> face68landmarks = detect_68landmarks_net.detect(source_img, boxes[position], face_landmark_5of68);
	vector<float> source_face_embedding = face_embedding_net.detect(source_img, face_landmark_5of68);

	detect_face_net.detect(target_img, boxes);
	if(boxes.empty()) {
		return -1;
	}
	sortBoxes(boxes, order);
	if(multipleFace) {
		output_img = target_img;
        for (size_t i = 0; i < boxes.size(); i++) {
			vector<Point2f> target_landmark_5;
            detect_68landmarks_net.detect(output_img, boxes[i], target_landmark_5);

			Mat swapimg = swap_face_net.process(output_img, source_face_embedding, target_landmark_5);
			output_img = enhance_face_net.process(swapimg, target_landmark_5);
		}
	} else {
		position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
		if(id <= boxes.size()) position = id;
		vector<Point2f> target_landmark_5;
		detect_68landmarks_net.detect(target_img, boxes[position], target_landmark_5);

		Mat swapimg = swap_face_net.process(target_img, source_face_embedding, target_landmark_5);
		output_img = enhance_face_net.process(swapimg, target_landmark_5);
	}
	
    return 0;
}
