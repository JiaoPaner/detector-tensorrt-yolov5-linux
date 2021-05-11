//
// Created by jiaopan on 2020-10-30.
//

//s ->> gd=0.33,gw = 0.50
//m ->> gd=0.67,gw = 0.75
//l ->> gd=1.0,gw = 1.0
//x ->> gd=1.33,gw = 1.25
#include <iostream>
#include <chrono>
#include <detector.h>
#include <vector>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"
#include "labels.h"
#include "cJSON.h"

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

static Logger gLogger;
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1

void Detector::loadWeightsToEngineFile(std::string weightsFile, std::string engineFileName) {
    std::cout << "starting loadWeightsToEngineFile" << std::endl;
    cudaSetDevice(DEVICE);
    float gd = 0.33f, gw = 0.50f;
    IHostMemory* modelStream{ nullptr };
    APIToModel(BATCH_SIZE, &modelStream, gd, gw, weightsFile);
    assert(modelStream != nullptr);
    std::ofstream p(engineFileName, std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();

}
void Detector::init(std::string engineFile) {
    cudaSetDevice(DEVICE);
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engineFile << " error!" << std::endl;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    this->runtime = createInferRuntime(gLogger);
    assert(this->runtime != nullptr);
    this->engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    this->context = engine->createExecutionContext();
    assert(this->context != nullptr);
    delete[] trtModelStream;
    assert(this->engine->getNbBindings() == 2);

}

//doInference(*context, stream, buffers, data, prob, BATCH_SIZE);

char* Detector::doInference(cv::Mat image) {
    cudaSetDevice(DEVICE);
    // Create GPU buffers on device
    const int inputIndex = this->engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = this->engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    CUDA_CHECK(cudaMalloc(&this->buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&this->buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    CUDA_CHECK(cudaStreamCreate(&this->stream));

    float input[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    this->createInputData(input,image);
    float output[BATCH_SIZE * OUTPUT_SIZE];

    CUDA_CHECK(cudaMemcpyAsync(this->buffers[0], input, BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, this->stream));
    this->context->enqueue(BATCH_SIZE, this->buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, this->buffers[1], BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, this->stream));
    cudaStreamSynchronize(this->stream);

    std::vector<Yolo::Detection> batch_res;
    nms(batch_res, output, CONF_THRESH, NMS_THRESH);

    cudaStreamDestroy(this->stream);
    CUDA_CHECK(cudaFree(this->buffers[inputIndex]));
    CUDA_CHECK(cudaFree(this->buffers[outputIndex]));

    cJSON  *result = cJSON_CreateObject(), *items = cJSON_CreateArray();
    for (int i = 0; i < batch_res.size(); ++i) {
        cJSON  *item = cJSON_CreateObject();
        cv::Rect rect = get_rect(image, batch_res[i].bbox);
        int labelIndex = batch_res[i].class_id;
        cJSON_AddStringToObject(item, "label",labels.at(labelIndex).c_str());
        cJSON_AddNumberToObject(item, "score", batch_res[i].conf);
        cJSON  *location = cJSON_CreateObject();
        cJSON_AddNumberToObject(location, "x", rect.x);
        cJSON_AddNumberToObject(location, "y", rect.y);
        cJSON_AddNumberToObject(location, "width", rect.width);
        cJSON_AddNumberToObject(location, "height", rect.height);
        cJSON_AddItemToObject(item, "location", location);
        cJSON_AddItemToArray(items, item);
    }
    cJSON_AddNumberToObject(result, "code", 0);
    cJSON_AddStringToObject(result, "msg", "success");
    cJSON_AddItemToObject(result, "data", items);
    char *resultJson = cJSON_PrintUnformatted(result);
    return resultJson;
}

void Detector::unload() {
    cudaStreamDestroy(this->stream);
    const int inputIndex = this->engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = this->engine->getBindingIndex(OUTPUT_BLOB_NAME);
    CUDA_CHECK(cudaFree(this->buffers[inputIndex]));
    CUDA_CHECK(cudaFree(this->buffers[outputIndex]));
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
}

ICudaEngine * Detector::buildEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt,
                                    float &gd, float &gw, std::string &weightsFile) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(weightsFile);

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, get_width(64, gw), 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, 9, 13, "model.8");

    /* ------ yolov5 head ------ */
    auto bottleneck_csp9 = C3(network, weightMap, *spp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

    auto upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(ResizeMode::kNEAREST);
    upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());

    ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

    auto upsample15 = network->addResize(*conv14->getOutput(0));
    assert(upsample15);
    upsample15->setResizeMode(ResizeMode::kNEAREST);
    upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());

    ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

    // yolo layer 0
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
    //yolo layer 1
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, det0, det1, det2);
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap){
        free((void*)(mem.second.values));
    }

    return engine;
}

void Detector::APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream, float &gd, float &gw,std::string &wts_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = buildEngine(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();

}

int Detector::get_depth(int x, float gd) {
    if (x == 1) return 1;
    else return round(x * gd) > 1 ? round(x * gd) : 1;
}

int Detector::get_width(int x, float gw, int divisor) {
    //return math.ceil(x / divisor) * divisor
    if (int(x * gw) % divisor == 0) return int(x * gw);
    return (int(x * gw / divisor) + 1) * divisor;
}

void Detector::createInputData(float *input,cv::Mat image) {
    cv::Mat pr_img = preprocess_img(image, INPUT_W, INPUT_H); // letterbox BGR to RGB
    int i = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < INPUT_W; ++col) {
            input[i] = (float)uc_pixel[2] / 255.0;
            input[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
            input[i + 2 * Yolo::INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }
}

