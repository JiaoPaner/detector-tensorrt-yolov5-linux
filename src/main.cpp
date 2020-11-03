//
// Created by jiaopan on 2020-10-30.
//

#include "api.h"
#include <iostream>
#include <chrono>

int main(){

    std::cout << "hello world" << std::endl;

    init("yolov5s.engine");

    auto start = std::chrono::system_clock::now();

    char* result = detectByFile("../samples/PartA_00456.jpg");
    std::cout << result << std::endl;

    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    return 0;
}