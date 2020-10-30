//
// Created by jiaopan on 2020-10-30.
//

#ifndef DETECTOR_YOLOV5_H
#define DETECTOR_YOLOV5_H

#include "NvInfer.h"
#include <string>
using namespace nvinfer1;

class Yolov5 {
    public:
        void loadWeightsToEngineFile(std::string weightsFile,std::string engineFileName,int height,int width);
        //void init(std::string engineFile);
        //void unload();
        //void doInference(cudaStream_t& stream, void **buffers, float* input, float* output,int height,int width,int channels);

    private:
        ICudaEngine* createEngine_s(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt,std::string weights ,int height,int width);
        //IRuntime* runtime{nullptr};
        //ICudaEngine* engine{nullptr};
        //IExecutionContext* context{nullptr};
};


#endif //DETECTOR_YOLOV5_H
