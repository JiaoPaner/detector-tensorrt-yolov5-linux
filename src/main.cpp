//
// Created by jiaopan on 2020-10-30.
//

#include "api.h"
#include <iostream>
#include <chrono>

int main(){

    std::cout << "hello world" << std::endl;
    //loadWeightsToEngineFile("../model/yolov5s.wts","yolov5s.engine");
    init("yolov5s.engine");

    auto start = std::chrono::system_clock::now();

    char* result = detectByFile("/home/adminpc/jiaopan/resource/bus.jpg");
    std::cout << result << std::endl;

    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    auto begin = std::chrono::system_clock::now();

    char* code = detectByFile("/home/adminpc/jiaopan/resource/bus.jpg");
    std::cout << code << std::endl;

    auto ending = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(ending - begin).count() << "ms" << std::endl;

    return 0;
}