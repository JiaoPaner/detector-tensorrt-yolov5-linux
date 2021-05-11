import cv2
import ctypes
import base64
from PIL import Image
from io import BytesIO
import json
import subprocess as sp
import multiprocessing as mp
from datetime import datetime


def image_to_base64(image):
    image = cv2.imencode('.jpg', image)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code


#test stream
def pushFrame(queue):
    cap = cv2.VideoCapture("/home/adminpc/jiaopan/projects/videos/dongjiaojiyi.mp4")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    command = ['ffmpeg',
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-pix_fmt', 'bgr24',
               '-s', "{}x{}".format(width, height),
               '-r', str(fps),
               '-re',
               '-i', '-',
               '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p',
               '-preset', 'ultrafast',
               '-f', 'flv',
               "rtmp://172.28.28.168/live/test"]
    while True:
        if len(command) > 0:
            # 管道配置
            p = sp.Popen(command, stdin=sp.PIPE)
            break
    while True:
        frame = queue.get()
        p.stdin.write(frame.tostring())
        cv2.waitKey(200)

def readFrame(queue):

    so = ctypes.cdll.LoadLibrary("libdetector.so")
    model_path = bytes("video_analysis.engine", "utf-8")
    so.init(model_path)
    detect = so.detectByBase64
    detect.restype = ctypes.c_char_p

    cap = cv2.VideoCapture("/home/adminpc/jiaopan/projects/videos/dongjiaojiyi.mp4")
    index = 0

    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            #frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_AREA)
            if index % 1 == 0:
                image = image_to_base64(frame)
                image = bytes(image, "utf-8")
                result = detect(image)

                result = ctypes.string_at(result, -1).decode("utf-8")
                result = json.loads(result)
                data = result["data"]
                for box in data:
                    x = int(box["location"]["x"])
                    y = int(box["location"]["y"])
                    width = int(box["location"]["width"])
                    height = int(box["location"]["height"])
                    cv2.putText(frame, box["label"], (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.rectangle(frame,(x,y),(x+width,y+height),(255,0,0),2)
            queue.put(frame)
            k = cv2.waitKey(20)
            index += 1
            # q键退出
            if k & 0xff == ord('q'):
                break

def run():
    mp.set_start_method(method='spawn')  # init
    queue = mp.Queue(maxsize=30)
    processes = [mp.Process(target=readFrame, args=(queue,)),
                 mp.Process(target=pushFrame, args=(queue,))]

    [process.start() for process in processes]
    [process.join() for process in processes]



#test file
def file_test():
    so = ctypes.cdll.LoadLibrary("libdetector.so")
    model_path = bytes("video_analysis.engine", "utf-8")
    so.init(model_path)
    image = bytes("test.jpg", "utf-8")
    detectFile = so.detectByFile
    detectFile.restype = ctypes.c_char_p
    start = datetime.now()
    result = detectFile(image)
    end = datetime.now()
    print("time cost:",(end-start).total_seconds())

    result = ctypes.string_at(result, -1).decode("utf-8")
    print(result)

    image = cv2.imread("test.jpg")
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

if __name__ == '__main__':
    file_test()
    run();