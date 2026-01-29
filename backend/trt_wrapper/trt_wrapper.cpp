#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <memory>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstring> // for strcpy if needed, but we use std::string

namespace {
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

thread_local std::string g_last_error;

bool set_error(const std::string& msg) {
    g_last_error = msg;
    return false;
}

std::vector<char> read_file(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        buffer.clear();
    }
    return buffer;
}
} // namespace

struct TrtRunner {
    TrtLogger logger;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;

    // We store names now instead of indices
    std::string input_name;
    std::string output_name;

    nvinfer1::Dims input_dims{};
    nvinfer1::Dims output_dims{};
    size_t input_bytes = 0;
    size_t output_bytes = 0;
    void* d_input = nullptr;
    void* d_output = nullptr;
    cudaStream_t stream = nullptr;
};

extern "C" {
const char* trt_last_error() { return g_last_error.c_str(); }

int trt_build_engine(const char* onnx_path, const char* engine_path) {
    g_last_error.clear();
    
    TrtLogger logger;
    initLibNvInferPlugins(&logger, "");
    
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder) {
        set_error("Failed to create builder.");
        return -1;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        set_error("Failed to create network.");
        return -1;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        set_error("Failed to create builder config.");
        return -1;
    }

    // Set memory pool size (workspace) - 4GB
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 4096ULL * 1024 * 1024);

    // Use FP16 if available
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser) {
        set_error("Failed to create ONNX parser.");
        return -1;
    }

    if (!parser->parseFromFile(onnx_path, static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::stringstream ss;
        ss << "Failed to parse ONNX file: " << onnx_path;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            ss << "\n" << parser->getError(i)->desc();
        }
        set_error(ss.str());
        return -1;
    }

    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
        set_error("Failed to build serialized network.");
        return -1;
    }

    std::ofstream outfile(engine_path, std::ios::binary);
    if (!outfile) {
        set_error("Failed to open output engine file for writing.");
        return -1;
    }

    outfile.write(static_cast<const char*>(plan->data()), plan->size());
    return 0;
}

// Forward declaration
void trt_destroy(TrtRunner* runner);

TrtRunner* trt_create(const char* engine_path) {
    g_last_error.clear();
    auto data = read_file(engine_path);
    if (data.empty()) {
        set_error("Failed to read engine file.");
        return nullptr;
    }

    auto* runner = new TrtRunner();
    // initLibNvInferPlugins returns boolean in some versions, but usually void or we ignore
    initLibNvInferPlugins(&runner->logger, "");
    
    runner->runtime = nvinfer1::createInferRuntime(runner->logger);
    if (!runner->runtime) {
        set_error("Failed to create TensorRT runtime.");
        delete runner;
        return nullptr;
    }

    runner->engine = runner->runtime->deserializeCudaEngine(data.data(), data.size());
    if (!runner->engine) {
        set_error("Failed to deserialize TensorRT engine.");
        trt_destroy(runner);
        return nullptr;
    }

    runner->context = runner->engine->createExecutionContext();
    if (!runner->context) {
        set_error("Failed to create TensorRT execution context.");
        trt_destroy(runner);
        return nullptr;
    }

    // Inspect V3 IOTensors
    int nb_io = runner->engine->getNbIOTensors();
    
    // Naive search for 1 input and 1 output if names not matching "images"/"output0"
    std::string found_input;
    std::string found_output;

    // Prefer "images" and "output0"
    bool has_images = false;
    bool has_output0 = false;
    
    // First pass loop to find them or defaults
    for (int i = 0; i < nb_io; ++i) {
        const char* name = runner->engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = runner->engine->getTensorIOMode(name);
        
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            if (std::string(name) == "images") {
                runner->input_name = name;
                has_images = true;
            } else if (found_input.empty()) {
                found_input = name;
            }
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
             if (std::string(name) == "output0") {
                runner->output_name = name;
                has_output0 = true;
            } else if (found_output.empty()) {
                found_output = name;
            }
        }
    }

    if (runner->input_name.empty()) runner->input_name = found_input;
    if (runner->output_name.empty()) runner->output_name = found_output;

    if (runner->input_name.empty() || runner->output_name.empty()) {
        set_error("Failed to resolve input/output tensors (expecting 1 input, 1 output).");
        trt_destroy(runner);
        return nullptr;
    }

    runner->input_dims = runner->engine->getTensorShape(runner->input_name.c_str());
    runner->output_dims = runner->engine->getTensorShape(runner->output_name.c_str());

    auto calc_bytes = [](const nvinfer1::Dims& dims) -> size_t {
        size_t count = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            count *= static_cast<size_t>(dims.d[i] < 0 ? 1 : dims.d[i]);
        }
        return count * sizeof(float);
    };

    runner->input_bytes = calc_bytes(runner->input_dims);
    runner->output_bytes = calc_bytes(runner->output_dims);

    if (cudaStreamCreate(&runner->stream) != cudaSuccess) {
        set_error("Failed to create CUDA stream.");
        trt_destroy(runner);
        return nullptr;
    }

    if (cudaMalloc(&runner->d_input, runner->input_bytes) != cudaSuccess ||
        cudaMalloc(&runner->d_output, runner->output_bytes) != cudaSuccess) {
        set_error("Failed to allocate CUDA buffers.");
        trt_destroy(runner);
        return nullptr;
    }

    return runner;
}

void trt_destroy(TrtRunner* runner) {
    if (!runner) return;
    if (runner->d_input) cudaFree(runner->d_input);
    if (runner->d_output) cudaFree(runner->d_output);
    if (runner->stream) cudaStreamDestroy(runner->stream);
    
    // In TRT 10 C++ API, we use delete
    delete runner->context;
    delete runner->engine;
    delete runner->runtime;
    delete runner;
}

int trt_get_input_dims(TrtRunner* runner, int* n, int* c, int* h, int* w) {
    if (!runner || !n || !c || !h || !w) return -1;
    auto dims = runner->input_dims;
    if (dims.nbDims == 3) {
        *n = 1;
        *c = dims.d[0];
        *h = dims.d[1];
        *w = dims.d[2];
        return 0;
    }
    if (dims.nbDims == 4) {
        *n = dims.d[0];
        *c = dims.d[1];
        *h = dims.d[2];
        *w = dims.d[3];
        return 0;
    }
    return -1;
}

int trt_get_output_dims(TrtRunner* runner, int* n, int* c, int* h, int* w) {
    if (!runner || !n || !c || !h || !w) return -1;
    auto dims = runner->output_dims;
    if (dims.nbDims == 2) {
        *n = 1;
        *c = dims.d[0];
        *h = dims.d[1];
        *w = 1;
        return 0;
    }
    if (dims.nbDims == 3) {
        *n = dims.d[0];
        *c = dims.d[1];
        *h = dims.d[2];
        *w = 1;
        return 0;
    }
    if (dims.nbDims == 4) {
        *n = dims.d[0];
        *c = dims.d[1];
        *h = dims.d[2];
        *w = dims.d[3];
        return 0;
    }
    return -1;
}

size_t trt_get_output_count(TrtRunner* runner) {
    if (!runner) return 0;
    size_t count = 1;
    for (int i = 0; i < runner->output_dims.nbDims; ++i) {
        count *= static_cast<size_t>(runner->output_dims.d[i] < 0 ? 1 : runner->output_dims.d[i]);
    }
    return count;
}

int trt_infer(TrtRunner* runner, const float* input, float* output) {
    if (!runner || !input || !output) return -1;
    if (cudaMemcpyAsync(runner->d_input, input, runner->input_bytes, cudaMemcpyHostToDevice, runner->stream) != cudaSuccess) {
        set_error("Failed to copy input to device.");
        return -1;
    }

    if (!runner->context->setTensorAddress(runner->input_name.c_str(), runner->d_input)) {
         set_error("Failed to set input tensor address");
         return -1;
    }
    if (!runner->context->setTensorAddress(runner->output_name.c_str(), runner->d_output)) {
         set_error("Failed to set output tensor address");
         return -1;
    }

    if (!runner->context->enqueueV3(runner->stream)) {
        set_error("TensorRT enqueueV3 failed.");
        return -1;
    }

    if (cudaMemcpyAsync(output, runner->d_output, runner->output_bytes, cudaMemcpyDeviceToHost, runner->stream) != cudaSuccess) {
        set_error("Failed to copy output to host.");
        return -1;
    }

    if (cudaStreamSynchronize(runner->stream) != cudaSuccess) {
        set_error("CUDA sync failed.");
        return -1;
    }

    return 0;
}
}
