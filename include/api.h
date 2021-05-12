//
// Created by jiaopan on 2020-10-30.
//

#ifndef DETECTOR_API_H
#define DETECTOR_API_H


extern "C" {


/**
 * common api
 */
void loadWeightsToEngineFile(const char* weightsFile,const char* engineFile);
void init(const char* engineFile);
void unload();

/**
 * detection api
 */
char* detectByBase64(const char* base64_data);
char* detectByFile(const char* file);

}

#endif //DETECTOR_API_H
