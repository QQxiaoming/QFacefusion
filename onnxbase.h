#ifndef ONNX_BASE
#define ONNX_BASE

#include <fstream>
#include <sstream>
#include <onnxruntime_cxx_api.h>
#if defined(COREML_FACEFUSION_BUILD)
#include <coreml_provider_factory.h>
#endif 

class OnnxBase {
public:
    OnnxBase(std::string model_path);
    ~OnnxBase();

public:
    Ort::Env env;
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


#endif // ONNX_BASE
