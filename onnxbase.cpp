#include "onnxbase.h"

using namespace Ort;

OnnxBase::OnnxBase(std::string model_path) {
    env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, model_path.c_str());

#if defined(CUDA_FACEFUSION_BUILD)
    try {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0; // 设置 GPU 设备 ID
        cuda_options.arena_extend_strategy = 0; // 使用默认的内存分配策略
        cuda_options.gpu_mem_limit = SIZE_MAX; // 设置 GPU 内存限制
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE; // 使用最优的卷积算法
        cuda_options.do_copy_in_default_stream = 1; // 在默认流中进行数据复制
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
}

OnnxBase::~OnnxBase() {
    delete ort_session;
}
