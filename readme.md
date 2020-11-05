this repo tensorrt's version is TensorRT-7.0.0.11.Ubuntu-16.04.x86_64-gnu.cuda-10.2.cudnn7.6.tar.gz
#### requirements
* cuda 10.2+ 
* cudnn 7.6+
* opencv 


### how to build
1.add model.pt file to model dir <br>
2.use tools/gen_wts.py generate model.wts file
3.use loadWeightsToEngineFile generate model.engine file(the example is in main.cpp) or you can call loadWeightsToEngineFile by python,java,etc.
4.please do init() method once before do anything 

