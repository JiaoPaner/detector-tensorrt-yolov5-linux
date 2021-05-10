//
// Created by jiaopan on 2020-10-30.
//

#ifndef DETECTOR_DETECTOR_H
#define DETECTOR_DETECTOR_H



#include "NvInfer.h"
#include <string>
#include <opencv2/opencv.hpp>
using namespace nvinfer1;

class Detector {
    public:
        void loadWeightsToEngineFile(std::string weightsFile,std::string engineFileName);
        void init(std::string engineFile);
        char * doInference(cv::Mat image);
        void unload();
    private:
        static int get_width(int x, float gw, int divisor = 8);
        static int get_depth(int x, float gd);
        void createInputData(float *data,cv::Mat image);
        void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, float& gd, float& gw, std::string& wts_name);
        ICudaEngine* buildEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& weightsFile);
        IRuntime* runtime{nullptr};
        ICudaEngine* engine{nullptr};
        IExecutionContext* context{nullptr};
        cudaStream_t stream;
        void* buffers[2];
};


#endif //DETECTOR_DETECTOR_H
