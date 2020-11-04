import cv2
import ctypes
import base64
from PIL import Image
from io import BytesIO
import json
from datetime import datetime


if __name__ == '__main__':
    so = ctypes.cdll.LoadLibrary("/home/adminpc/jiaopan/projects/detector-tensorrtx-linux/build/libdetector.so")
    model_path = bytes("/home/adminpc/jiaopan/projects/detector-tensorrtx-linux/build/yolov5s.engine", "utf-8")
    so.init(model_path)
    image = bytes("/home/adminpc/jiaopan/resource/bus.jpg", "utf-8")
    detectFile = so.detectByFile
    detectFile.restype = ctypes.c_char_p
    start = datetime.now()
    result = detectFile(image, ctypes.c_float(0.5))
    end = datetime.now()
    print("time cost:",(end-start).total_seconds())

    result = ctypes.string_at(result, -1).decode("utf-8")
    print(result)

    image = cv2.imread("/home/adminpc/jiaopan/resource/bus.jpg")
    result = json.loads(result)
    data = result["data"]
    for box in data:
        x = int(box["location"]["x"])
        y = int(box["location"]["y"])
        width = int(box["location"]["width"])
        height = int(box["location"]["height"])
        cv2.putText(image, box["label"], (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
        cv2.rectangle(image,(x,y),(x+width,y+height),(255,0,0),2)
    cv2.imwrite("output.jpg",image)
