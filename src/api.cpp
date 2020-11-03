//
// Created by jiaopan on 2020-10-30.
//

#include "api.h"
#include "yolov5.h"
#include "cJSON.h"
#include "utils.h"

static  Yolov5 yolov5;

void loadWeightsToEngineFile(const char* weightsFile,const char* engineFileName){
    yolov5.loadWeightsToEngineFile(weightsFile,engineFileName);
}
void init(const char* model_path){
    std::cout << "loading model:" << model_path << std::endl;
    yolov5.init(model_path);
    std::cout << "this is a detector lib by jiaopaner@qq.com" << std::endl;
}
void unload(){
    yolov5.unload();
}

/**
 * detection api
 */
char* detectByBase64(const char* base64_data, float min_score){
    try {
        std::string data(base64_data);
        cv::Mat image = base64ToMat(data);
        return yolov5.doInference(image,min_score);
    }
    catch (const char* msg) {
        cJSON* result = cJSON_CreateObject(), * data = cJSON_CreateArray();;
        cJSON_AddNumberToObject(result, "code", 1);
        cJSON_AddStringToObject(result, "msg", msg);
        cJSON_AddItemToObject(result, "data", data);
        return cJSON_PrintUnformatted(result);
    }
}
char* detectByFile(const char* file, float min_score){
    try {
        cv::Mat image = cv::imread(file);
        return yolov5.doInference(image,min_score);
    }
    catch (const char* msg) {
        cJSON* result = cJSON_CreateObject(), * data = cJSON_CreateArray();;
        cJSON_AddNumberToObject(result, "code", 1);
        cJSON_AddStringToObject(result, "msg", msg);
        cJSON_AddItemToObject(result, "data", data);
        return cJSON_PrintUnformatted(result);
    }
}