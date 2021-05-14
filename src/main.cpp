//
// Created by jiaopan on 2020-10-30.
//

#include <iostream>
#include <chrono>
#include "api.h"
int main(){

    //loadWeightsToEngineFile("../video_analysis_2021_04_27_v4.wts","video_analysis.engine");

    init("video_analysis.engine");
    auto start = std::chrono::system_clock::now();
    char* result = detectByFile("test.jpg","[{\"x\":0,\"y\":0},{\"x\":500,\"y\":200},{\"x\":500,\"y\":600},{\"x\":100,\"y\":600}]");
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::cout << result;
    return 0;
}