//
// Created by jiaopan on 2020-10-30.
//

#ifndef DETECTOR_YOLOV5_H
#define DETECTOR_YOLOV5_H

#include "NvInfer.h"
#include <string>
#include <opencv2/opencv.hpp>
using namespace nvinfer1;

#define NET s  // s m l x
#define NETSTRUCT(str) createEngine_##str
#define CREATENET(net) NETSTRUCT(net)

class Yolov5 {
    public:
        void loadWeightsToEngineFile(std::string weightsFile,std::string engineFileName);
        void init(std::string engineFile);
        void unload();
        char * doInference(cv::Mat image,float confThresh = 0.5f,int channels = 3);
    private:
        ICudaEngine* createEngine_s(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt,std::string weights);
        ICudaEngine* createEngine_m(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt,std::string weights);
        ICudaEngine* createEngine_l(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt,std::string weights);
        ICudaEngine* createEngine_x(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt,std::string weights);

        IRuntime* runtime{nullptr};
        ICudaEngine* engine{nullptr};
        IExecutionContext* context{nullptr};
        cudaStream_t stream;
        void* buffers[2];

};


#endif //DETECTOR_YOLOV5_H
