tensorrt's version is TensorRT-7.0.0.11.Ubuntu-16.04.x86_64-gnu.cuda-10.2.cudnn7.6.tar.gz<br>
**yolov5 version is v3.0**
#### requirements
* cuda 10.2+ 
* cudnn 7.6+
* opencv 


### how to build
1.add model.pt file to model dir <br>
2.use tools/gen_wts.py generate model.wts file<br>
3.use loadWeightsToEngineFile(model.wts,engine_name) generate model.engine file(the example is in main.cpp) or you can call loadWeightsToEngineFile by python,java,etc.<br>
4.please call init() method once before doing anything <br>
5.the test/test.py is a example that how python call .so

