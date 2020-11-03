//
// Created by jiaopan on 2020-10-30.
//

#ifndef DETECTOR_API_H
#define DETECTOR_API_H


extern "C" {

/**
 * common api
 */
void loadWeightsToEngineFile(const char* weightsFile,const char* engineFileName);
void init(const char* model_path);
void unload();

/**
 * detection api
 */
char* detectByBase64(const char* base64_data, float min_score = 0.5);
char* detectByFile(const char* file, float min_score = 0.5);

}

#endif //DETECTOR_API_H
