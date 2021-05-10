tensorrt's version is TensorRT-7.0.0.11.Ubuntu-16.04.x86_64-gnu.cuda-10.2.cudnn7.6.tar.gz<br>
**this repo is based on yolov5 v4.0**
#### requirements
* cuda 10.2+ 
* cudnn 7.6+
* opencv 
* tensorrt


### how to build
1.generate .wts from pytorch with .pt<br>
  download yolov5s.pt,address:https://github.com/ultralytics/yolov5/releases/tag/v4.0
```bash
git clone detector-tensorrt-yolov5-linux
git clone -b v4.0 https://github.com/ultralytics/yolov5.git
cp detector-tensorrt-yolov5-linux/gen_wts.py {dir}/yolov5
cd {dir}/yolov5
python gen_wts.py {dir}/yolov5s.pt 
```
a file 'yolov5s.wts' will be generated.

2.build<br>
  update CLASS_NUM in include/yololayer.h if your model is trained on custom dataset
```bash
cd detector-tensorrt-yolov5-linux
mkdir build
cd build
cp {dir}/yolov5/yolov5s.wts detector-tensorrt-yolov5-linux/build
cmake ..
make
```
3.run<br>
  add test.jpg to build dir and run
```bash
#

./detector 

# the build dir will generated detected jpg.
```
use loadWeightsToEngineFile(model.wts,engine_name) method in include/api.h generate model.engine file(the example is in main.cpp)<br>
init() method in include/api.h is to load engine file ,please call it once before doing anything <br>
the test/test.py is a example that how python call .so


