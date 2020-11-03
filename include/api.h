//
// Created by jiaopan on 2020-10-30.
//

#ifndef DETECTOR_API_H
#define DETECTOR_API_H


extern "C" {

/**
 * common api
 */
int init(const char* model_path);
int unload();

/**
 * detection api
 */
char* detectByBase64(const char* base64_data, float min_score = 0.9);
char* detectByFile(const char* file, float min_score = 0.9);

}

#endif //DETECTOR_API_H
