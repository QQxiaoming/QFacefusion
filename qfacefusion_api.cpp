#include <onnxruntime_cxx_api.h>
#if defined(COREML_FACEFUSION_BUILD)
#include <coreml_provider_factory.h>
#endif

#include "qfacefusion_api.h"

using namespace cv;
using namespace std;
using namespace FaceFusionUtils;

FaceFusion::FaceFusion(const std::string &model_path) 
	: m_model_path(model_path)
	, m_detect_face_net(m_model_path+"/yoloface_8n.onnx")
	, m_detect_68landmarks_net(m_model_path+"/2dfan4.onnx")
	, m_face_embedding_net(m_model_path+"/arcface_w600k_r50.onnx")
	, m_face_classifier_net(m_model_path+"/fairface.onnx")
	, m_swap_face_net(m_model_path+"/"+INSWAPPER_NAME)
	, m_enhance_face_net(m_model_path+"/gfpgan_1.4.onnx")
	, m_styleganexage_net(m_model_path+"/styleganex_age.onnx") {
}

FaceFusion::~FaceFusion() {
}

template<typename T> void FaceFusion::sortBoxes(std::vector<T> &boxes, uint32_t order) {
	//根据order来排序
	// order = 0: 左→右排序
	// order = 1: 右→左排序
	// order = 2: 上→下排序
	// order = 3: 下→上排序
	// order = 4: 大→小排序
	// order = 5: 小→大排序
	// order = 6: 好→差排序
	// order = 7: 差→好排序
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
	} else if(order == 4) {
		sort(boxes.begin(), boxes.end(), [](const T &a, const T &b) {
			float area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin);
			float area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin);
			return area_a > area_b;
		});
	} else if(order == 5) {
		sort(boxes.begin(), boxes.end(), [](const T &a, const T &b) {
			float area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin);
			float area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin);
			return area_a < area_b;
		});
	} else if(order == 6) {
		sort(boxes.begin(), boxes.end(), [](const T &a, const T &b) {
			return a.score > b.score;
		});
	} else if(order == 7) {
		sort(boxes.begin(), boxes.end(), [](const T &a, const T &b) {
			return a.score < b.score;
		});
	}
}

FaceClassifier::FaceGender FaceFusion::checkGender(const cv::Mat &source_img, const Bbox &box) {
    vector<Point2f> face_landmark_5of68;
	m_detect_68landmarks_net->detect(source_img, box, face_landmark_5of68);
	vector<int> face_classifier_id = m_face_classifier_net->detect(source_img, face_landmark_5of68);
	FaceClassifier::FaceGender gender = (FaceClassifier::FaceGender)face_classifier_id[1];
	return gender;
}

int FaceFusion::runSwap(const cv::Mat &source_img, const cv::Mat &target_img, cv::Mat &output_img,
			uint32_t id, uint32_t order, int multipleFace, int genderMask, std::function<void(uint64_t)> progress) {
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
	if(genderMask != 0) {
		vector<Bbox> boxes_tmp;
		for(auto &box : boxes) {
			FaceClassifier::FaceGender gender = checkGender(target_img, box);
			if((genderMask == 1) && gender == FaceClassifier::FEMALE) {
				boxes_tmp.push_back(box);
			} else if((genderMask == 2) && gender == FaceClassifier::MALE) {
				boxes_tmp.push_back(box);
			}
		}
		boxes = boxes_tmp;
	}
	if(boxes.empty()) {
		return -1;
	}
	sortBoxes(boxes, order);
	if(multipleFace == 1) {
    	if(progress) progress(50);
		output_img = target_img;
		int position = 0;
		if(id <= boxes.size()) position = id;
        for (size_t i = position; i < boxes.size(); i++) {
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
	} else if(multipleFace == 0) {
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

void FaceFusion::clearReference(void) {
	m_reference_face_embedding_arr.clear();
}

int FaceFusion::setReference(const cv::Mat &reference_img, uint32_t id) {
	if(m_reference_face_embedding_arr.size() < id) {
        return -1;
	}
	if(reference_img.empty()){
        return -1;
    }
	if (reference_img.channels() != 3) {
        return -1;
    }
	vector<Bbox> boxes;
    m_detect_face_net->detect(reference_img, boxes);
	if(boxes.empty()) {
		return -1;
	}
	int position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
	vector<Point2f> face_landmark_5of68;
    m_detect_68landmarks_net->detect(reference_img, boxes[position], face_landmark_5of68);
    vector<float> reference_face_embedding = m_face_embedding_net->detect(reference_img, face_landmark_5of68);
	if(m_reference_face_embedding_arr.size() == id) {
		m_reference_face_embedding_arr.push_back(reference_face_embedding);
	} else {
		m_reference_face_embedding_arr[id] = reference_face_embedding;
	}
    return 0;
}

int FaceFusion::runSwap(const cv::Mat &target_img, cv::Mat &output_img,
			uint32_t id, uint32_t order, int multipleFace, int genderMask, float similar_thres,
			std::function<void(uint64_t)> progress) {
	m_findFace = 0;
	m_Similarity.clear();
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
	if(genderMask != 0) {
		vector<Bbox> boxes_tmp;
		for(auto &box : boxes) {
			FaceClassifier::FaceGender gender = checkGender(target_img, box);
			if((genderMask == 1) && gender == FaceClassifier::FEMALE) {
				boxes_tmp.push_back(box);
			} else if((genderMask == 2) && gender == FaceClassifier::MALE) {
				boxes_tmp.push_back(box);
			}
		}
		boxes = boxes_tmp;
	}
	if(boxes.empty()) {
		return -1;
	}
	m_findFace = boxes.size();
    std::vector<std::vector<float>> source_face_embedding_arr;
	if(m_reference_face_embedding_arr.empty()) {
		// 按照order排序
		sortBoxes(boxes, order);
		source_face_embedding_arr = m_source_face_embedding_arr;
	} else {
		// 按照与reference_face_embedding的相似度排序
		struct BboxWithSimilarity {
			Bbox box;
			vector<float> similarity;
		};
		vector<BboxWithSimilarity> boxes_tmp;
		for(auto &box : boxes) {
			BboxWithSimilarity bbox;
			bbox.box.xmin = box.xmin;
    		bbox.box.ymin = box.ymin;
    		bbox.box.xmax = box.xmax;
    		bbox.box.ymax = box.ymax;
			vector<Point2f> target_landmark_5;
			m_detect_68landmarks_net->detect(target_img, box, target_landmark_5);
			std::vector<float> find_similarity;
			for(auto &reference_face_embedding : m_reference_face_embedding_arr) {
				vector<float> target_face_embedding = m_face_embedding_net->detect(target_img, target_landmark_5);
				float sim = dot_product(reference_face_embedding, target_face_embedding);
				bbox.similarity.push_back(sim);
				find_similarity.push_back(sim);
			}
			boxes_tmp.push_back(bbox);
			m_Similarity.push_back(find_similarity);
		}
		boxes.clear();
		for(size_t i = 0; i < m_reference_face_embedding_arr.size(); i++) {
			sort(boxes_tmp.begin(), boxes_tmp.end(), [i](const BboxWithSimilarity &a, const BboxWithSimilarity &b) {
                return a.similarity.at(i) < b.similarity.at(i);
			});
            BboxWithSimilarity temp = boxes_tmp.back();
			if(temp.similarity.at(i) > similar_thres) {
				boxes_tmp.pop_back();
				Bbox max;
				max.xmin = temp.box.xmin;
				max.ymin = temp.box.ymin;
				max.xmax = temp.box.xmax;
				max.ymax = temp.box.ymax;
				boxes.push_back(max);
				if(m_source_face_embedding_arr.size() > i) {
					source_face_embedding_arr.push_back(m_source_face_embedding_arr.at(i));
				}
			}
		}
		if(boxes.empty()) {
			return -1;
		}
	}
	if((multipleFace == 2) || (multipleFace == 1)) {
    	if(progress) progress(30);
		output_img = target_img;
		int position = 0;
		if(id <= boxes.size()) position = id;
        for (size_t i = position; i < boxes.size(); i++) {
			if(progress) progress(30+i*70/boxes.size());
			vector<Point2f> target_landmark_5;
            m_detect_68landmarks_net->detect(output_img, boxes[i], target_landmark_5);

			if(progress) progress(30+i*70/boxes.size()+20/boxes.size());
			uint32_t source_id = 0;
			if(multipleFace == 2) {
				source_id = i % source_face_embedding_arr.size();
			}
            Mat swapimg = m_swap_face_net->process(output_img, source_face_embedding_arr[source_id], target_landmark_5);
			if(progress) progress(30+i*70/boxes.size()+50/boxes.size());
			output_img = m_enhance_face_net->process(swapimg, target_landmark_5);
			if(progress) progress(30+i*70/boxes.size()+70/boxes.size());
		}
		if(progress) progress(100);
	} else if(multipleFace == 0) {
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
        cv::rectangle(output_img, cv::Point(boxes[position].xmin, boxes[position].ymin), cv::Point(boxes[position].xmax, boxes[position].ymax), cv::Scalar(0, 255, 0), 2);
	}
	
	return 0;
}

int FaceFusion::setDetect(const cv::Mat &source_img, cv::Mat &output_img, uint32_t order, int genderMask) {
	if(source_img.empty()){
        return -1;
    }
	if (source_img.channels() != 3) {
        return -1;
    }
	vector<BboxWithKP5> boxes;
    m_detect_face_net->detect_with_kp5(source_img, boxes);
	sortBoxes(boxes, order);

	vector<Bbox> boxesNotKp5;
	for(auto &box : boxes) {
		Bbox bbox;
		bbox.xmin = box.xmin;
		bbox.ymin = box.ymin;
		bbox.xmax = box.xmax;
		bbox.ymax = box.ymax;
		boxesNotKp5.push_back(bbox);
	}

	cv::Mat temp_vision_frame = source_img.clone();
	int face_count = 0;
    for (size_t i = 0; i < boxes.size(); i++){
		vector<Point2f> face_landmark_5of68;
		vector<Point2f> face68landmarks = m_detect_68landmarks_net->detect(source_img, boxesNotKp5[i], face_landmark_5of68);
		vector<int> face_classifier_id = m_face_classifier_net->detect(source_img, face_landmark_5of68);
		FaceClassifier::FaceRace reace = (FaceClassifier::FaceRace)face_classifier_id[0];
		FaceClassifier::FaceGender gender = (FaceClassifier::FaceGender)face_classifier_id[1];
		FaceClassifier::FaceAge age = (FaceClassifier::FaceAge)face_classifier_id[2];
		if(genderMask != 0) {
			if((genderMask == 1) && gender != FaceClassifier::FEMALE) {
				continue;
			} else if((genderMask == 2) && gender != FaceClassifier::MALE) {
				continue;
			}
		}
		cv::rectangle(temp_vision_frame, cv::Point(boxes[i].xmin, boxes[i].ymin), cv::Point(boxes[i].xmax, boxes[i].ymax), cv::Scalar(0, 255, 0), 2);
		for (int j = 0; j < 68; j++){
			Point2f face_landmark = face68landmarks.at(j);
			cv::circle(temp_vision_frame, face_landmark, 2, cv::Scalar(255, 255, 0), -1);		
		}
		for (int j = 0; j < 5; j++){
			Point2f face_landmark = face_landmark_5of68.at(j);
			cv::circle(temp_vision_frame, face_landmark, 4, cv::Scalar(0, 0, 255), 2);		
		}
		for (int j = 0; j < 5; j++){
			cv::circle(temp_vision_frame, cv::Point(boxes[i].kp5[j].x, boxes[i].kp5[j].y), 3, cv::Scalar(0, 255, 0), 2);
		}

		cv::Point point = cv::Point(boxes[i].xmin, boxes[i].ymin);
		if (point.y < 3) point.y += 3; else point.y -= 3;
		cv::putText(temp_vision_frame, std::to_string(face_count++)+":"+std::to_string(boxes[i].score), point, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
		
		switch(gender) {
			case FaceClassifier::MALE:
				cv::putText(temp_vision_frame, "male", cv::Point(boxes[i].xmin, boxes[i].ymin+20), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				break;
			case FaceClassifier::FEMALE:
				cv::putText(temp_vision_frame, "female", cv::Point(boxes[i].xmin, boxes[i].ymin+20), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				break;
			default:
				cv::putText(temp_vision_frame, "male", cv::Point(boxes[i].xmin, boxes[i].ymin+20), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				break;
		}
		switch(age) {
			case FaceClassifier::AGE_0_2:
				cv::putText(temp_vision_frame, "0-2", cv::Point(boxes[i].xmin, boxes[i].ymin+40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				break;
			case FaceClassifier::AGE_3_9:
				cv::putText(temp_vision_frame, "3-9", cv::Point(boxes[i].xmin, boxes[i].ymin+40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				break;
			case FaceClassifier::AGE_10_19:
				cv::putText(temp_vision_frame, "10-19", cv::Point(boxes[i].xmin, boxes[i].ymin+40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				break;
			case FaceClassifier::AGE_20_29:
				cv::putText(temp_vision_frame, "20-29", cv::Point(boxes[i].xmin, boxes[i].ymin+40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				break;
			case FaceClassifier::AGE_30_39:
				cv::putText(temp_vision_frame, "30-39", cv::Point(boxes[i].xmin, boxes[i].ymin+40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				break;
			case FaceClassifier::AGE_40_49:
				cv::putText(temp_vision_frame, "40-49", cv::Point(boxes[i].xmin, boxes[i].ymin+40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				break;
			case FaceClassifier::AGE_50_59:
				cv::putText(temp_vision_frame, "50-59", cv::Point(boxes[i].xmin, boxes[i].ymin+40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				break;
			case FaceClassifier::AGE_60_69:
				cv::putText(temp_vision_frame, "60-69", cv::Point(boxes[i].xmin, boxes[i].ymin+40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				break;
			default:
			case FaceClassifier::AGE_70_100:
				cv::putText(temp_vision_frame, "70-100", cv::Point(boxes[i].xmin, boxes[i].ymin+40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				break;
		}
		switch(reace) {
			default:
			case FaceClassifier::WHITE:
				cv::putText(temp_vision_frame, "white", cv::Point(boxes[i].xmin, boxes[i].ymin+60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				break;
			case FaceClassifier::BLACK:
				cv::putText(temp_vision_frame, "black", cv::Point(boxes[i].xmin, boxes[i].ymin+60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				break;
			case FaceClassifier::LATINO:
			case FaceClassifier::ASIAN:
				cv::putText(temp_vision_frame, "asian", cv::Point(boxes[i].xmin, boxes[i].ymin+60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				break;
			case FaceClassifier::INDIAN:
				cv::putText(temp_vision_frame, "indian", cv::Point(boxes[i].xmin, boxes[i].ymin+60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				break;
			case FaceClassifier::ARABIC:
				cv::putText(temp_vision_frame, "arabic", cv::Point(boxes[i].xmin, boxes[i].ymin+60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
				break;
		}
		vector<float> target_face_embedding = m_face_embedding_net->detect(source_img, face_landmark_5of68);
		int refIndex = 0;
		for(auto &reference_face_embedding : m_reference_face_embedding_arr) {
			float sim = dot_product(reference_face_embedding, target_face_embedding);
			cv::putText(temp_vision_frame, "sim"+std::to_string(refIndex)+":"+std::to_string(sim), cv::Point(boxes[i].xmin, boxes[i].ymin+80+refIndex*20), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
			refIndex++;
		}
	}
	output_img = temp_vision_frame;
    return 0;
}

int FaceFusion::setAgeModify(const cv::Mat &source_img, cv::Mat &output_img, float direction) {
	if(source_img.empty()){
        return -1;
    }
	if (source_img.channels() != 3) {
        return -1;
    }
	vector<Bbox> boxes;
    m_detect_face_net->detect(source_img, boxes);

	vector<Point2f> face_landmark_5of68;
	m_detect_68landmarks_net->detect(source_img, boxes[0], face_landmark_5of68);
	cv::Mat temp_frame = m_styleganexage_net->process(source_img, face_landmark_5of68, direction);
	
	vector<Bbox> temp_frame_boxes;
	m_detect_face_net->detect(temp_frame, temp_frame_boxes);
	if(temp_frame_boxes.empty()) {
        return -1;
	}
	vector<Point2f> temp_face_landmark_5of68;
	m_detect_68landmarks_net->detect(temp_frame, temp_frame_boxes[0], temp_face_landmark_5of68);
	vector<float> temp_face_embedding = m_face_embedding_net->detect(temp_frame, temp_face_landmark_5of68);
	
	Mat swapimg = m_swap_face_net->process(source_img, temp_face_embedding, face_landmark_5of68);
	output_img = m_enhance_face_net->process(swapimg, face_landmark_5of68);
    return 0;
}

int FaceFusion::faceSwap(const string &source_path, 
						 const string &target_path, 
						 const string &output_path, 
						 uint32_t id, uint32_t order, int multipleFace, int genderMask) {
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
	if(genderMask != 0) {
		FaceClassifier face_classifier_net(root_mod_path+"/fairface.onnx");
		vector<Bbox> boxes_tmp;
		for(auto &box : boxes) {
			vector<Point2f> face_landmark_5of68;
			detect_68landmarks_net.detect(target_img, box, face_landmark_5of68);
			vector<int> face_classifier_id = face_classifier_net.detect(target_img, face_landmark_5of68);
			FaceClassifier::FaceGender gender = (FaceClassifier::FaceGender)face_classifier_id[1];
			if((genderMask == 1) && gender == FaceClassifier::FEMALE) {
				boxes_tmp.push_back(box);
			} else if((genderMask == 2) && gender == FaceClassifier::MALE) {
				boxes_tmp.push_back(box);
			}
		}
		boxes = boxes_tmp;
	}
	if(boxes.empty()) {
		return -1;
	}
	sortBoxes(boxes, order);
	Mat resultimg = target_img;
	if(multipleFace == 1) {
		int position = 0;
		if(id <= boxes.size()) position = id;
        for (size_t i = position; i < boxes.size(); i++) {
			vector<Point2f> target_landmark_5;
            detect_68landmarks_net.detect(resultimg, boxes[i], target_landmark_5);

			Mat swapimg = swap_face_net.process(resultimg, source_face_embedding, target_landmark_5);
			resultimg = enhance_face_net.process(swapimg, target_landmark_5);
		}
	} else if(multipleFace == 0) {
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
						 uint32_t id, uint32_t order, int multipleFace, int genderMask) {
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
	if(genderMask != 0) {
		vector<Bbox> boxes_tmp;
		FaceClassifier face_classifier_net(root_mod_path+"/fairface.onnx");
		for(auto &box : boxes) {
			vector<Point2f> face_landmark_5of68;
			detect_68landmarks_net.detect(target_img, box, face_landmark_5of68);
			vector<int> face_classifier_id = face_classifier_net.detect(target_img, face_landmark_5of68);
			FaceClassifier::FaceGender gender = (FaceClassifier::FaceGender)face_classifier_id[1];
			if((genderMask == 1) && gender == FaceClassifier::FEMALE) {
				boxes_tmp.push_back(box);
			} else if((genderMask == 2) && gender == FaceClassifier::MALE) {
				boxes_tmp.push_back(box);
			}
		}
		boxes = boxes_tmp;
	}
	if(boxes.empty()) {
		return -1;
	}
	sortBoxes(boxes, order);
	if(multipleFace == 1) {
		output_img = target_img;
		int position = 0;
		if(id <= boxes.size()) position = id;
        for (size_t i = position; i < boxes.size(); i++) {
			vector<Point2f> target_landmark_5;
            detect_68landmarks_net.detect(output_img, boxes[i], target_landmark_5);

			Mat swapimg = swap_face_net.process(output_img, source_face_embedding, target_landmark_5);
			output_img = enhance_face_net.process(swapimg, target_landmark_5);
		}
	} else if(multipleFace == 0) {
		position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
		if(id <= boxes.size()) position = id;
		vector<Point2f> target_landmark_5;
		detect_68landmarks_net.detect(target_img, boxes[position], target_landmark_5);

		Mat swapimg = swap_face_net.process(target_img, source_face_embedding, target_landmark_5);
		output_img = enhance_face_net.process(swapimg, target_landmark_5);
	}
	
    return 0;
}

string FaceFusion::getModelInfo(string model_path) {
	stringstream info;
	
    using namespace Ort;
    SessionOptions sessionOptions = SessionOptions();
    Env env(ORT_LOGGING_LEVEL_ERROR, "getModelInfo");

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
    Session ort_session(env, widestr.c_str(), sessionOptions);
#endif
#if defined(LINUX_FACEFUSION_BUILD) || defined(MACOS_FACEFUSION_BUILD)
    Session ort_session(env, model_path.c_str(), sessionOptions);
#endif

	std::function<std::string(ONNXTensorElementDataType)> getDataTypeName = [](ONNXTensorElementDataType dataType) {
		switch (dataType) {
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
			return "UNDEFINED";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
			return "FLOAT";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
			return "UINT8";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
			return "INT8";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
			return "UINT16";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
			return "INT16";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
			return "INT32";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
			return "INT64";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
			return "STRING";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
			return "BOOL";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
			return "FLOAT16";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
			return "DOUBLE";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
			return "UINT32";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
			return "UINT64";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
			return "COMPLEX64";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
			return "COMPLEX128";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
			return "BFLOAT16";
		// float 8 types were introduced in onnx 1.14, see https://onnx.ai/onnx/technical/float8.html
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
			return "FLOAT8E4M3FN";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
			return "FLOAT8E4M3FNUZ";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
			return "FLOAT8E5M2";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
			return "FLOAT8E5M2FNUZ";
		// Int4 types were introduced in ONNX 1.16. See https://onnx.ai/onnx/technical/int4.html
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
			return "UINT4";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
			return "INT4";
		default:
			return "UNKNOWN";
		}
    };

    AllocatorWithDefaultOptions allocator;
	info << model_path << ":" << endl;
    try {
		ModelMetadata modelMetadata = ort_session.GetModelMetadata();
		info << "Producer Name: " << modelMetadata.GetProducerNameAllocated(allocator).get() << endl;
        info << "Model Version: " << modelMetadata.GetVersion() << endl;
		info << "Graph Name: " << modelMetadata.GetGraphNameAllocated(allocator).get() << endl;
		info << "Domain: " << modelMetadata.GetDomainAllocated(allocator).get() << endl;
		info << "Description: " << modelMetadata.GetDescriptionAllocated(allocator).get() << endl;
		info << "Graph Description: " << modelMetadata.GetGraphDescriptionAllocated(allocator).get() << endl;
		info << "Custom Metadata Map:" << endl;
		std::vector<AllocatedStringPtr> keys = modelMetadata.GetCustomMetadataMapKeysAllocated(allocator);
		for (const auto& key : keys) {
			info << "  " << key.get() << ": " << modelMetadata.LookupCustomMetadataMapAllocated(key.get(),allocator).get() << endl;
		}
	} catch (const Ort::Exception& e) {
		info << "Error getting model metadata: " << e.what() << endl;
	}

    size_t numInputNodes = ort_session.GetInputCount();
	info << "Number of inputs: " << numInputNodes << endl;
    for (size_t i = 0; i < numInputNodes; i++) {
        TypeInfo inputTypeInfo = ort_session.GetInputTypeInfo(i);

        ConstTensorTypeAndShapeInfo inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

		std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
        ONNXTensorElementDataType inputTensorInfoElementType = inputTensorInfo.GetElementType();

		info << "Input Name: " << ort_session.GetInputNameAllocated(i, allocator).get() << endl;
		info << "Type: " << getDataTypeName(inputTensorInfoElementType).c_str() << endl;
		info << "Num Dimensions: " << inputDims.size() << endl;
		for (size_t j = 0; j < inputDims.size(); j++) {
			const char *dimSymbolic = nullptr;
            inputTensorInfo.GetSymbolicDimensions(&dimSymbolic, j);
			if(dimSymbolic)
				info << "dim[" << j << "]: " << inputDims[j] << " " << dimSymbolic << endl;
			else
				info << "dim[" << j << "]: " << inputDims[j] << endl;
		}
    }
    size_t numOutputNodes = ort_session.GetOutputCount();
	info << "Number of outputs: " << numOutputNodes << endl;
    for (size_t i = 0; i < numOutputNodes; i++) {
		TypeInfo outputTypeInfo = ort_session.GetOutputTypeInfo(i);

		ConstTensorTypeAndShapeInfo outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

		std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
        ONNXTensorElementDataType outputTensorInfoElementType = outputTensorInfo.GetElementType();
		
		info << "Output Name: " << ort_session.GetOutputNameAllocated(i, allocator).get() << endl;
		info << "Type: " << getDataTypeName(outputTensorInfoElementType).c_str() << endl;
		info << "Num Dimensions: " << outputDims.size() << endl;
		for (size_t j = 0; j < outputDims.size(); j++) {
			const char *dimSymbolic = nullptr;
			outputTensorInfo.GetSymbolicDimensions(&dimSymbolic, j);
			if(dimSymbolic)
				info << "dim[" << j << "]: " << outputDims[j] << " " << dimSymbolic << endl;
			else
				info << "dim[" << j << "]: " << outputDims[j] << endl;
		}
    }

	return info.str();
}

