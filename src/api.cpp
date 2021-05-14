//
// Created by jiaopan on 2020-10-30.
//

#include "api.h"
#include "detector.h"
#include "cJSON.h"

static  Detector detector;

cv::Mat base64ToMat(std::string &base64_data);
std::string base64Decode(const char *data, int dataByte);

/**
 * detection api
 */
void loadWeightsToEngineFile(const char* weightsFile,const char* engineFileName){
    detector.loadWeightsToEngineFile(weightsFile,engineFileName);
}
void init(const char* engineFile){
    std::cout << "loading engineFile:" << engineFile << std::endl;
    detector.init(engineFile);
}
void unload(){
    detector.unload();
}

char* detectByBase64(const char* base64_data,const char* rois){
    try {
        std::string data(base64_data);
        cv::Mat image = base64ToMat(data);
        if(rois != ""){
            cv::Mat dst;
            cv::Mat roi = cv::Mat::zeros(image.size(),CV_8U);
            std::vector<std::vector<cv::Point>> contour;
            std::vector<cv::Point> pts;

            cJSON *root;
            root = cJSON_Parse(rois);
            cJSON *data = cJSON_GetObjectItem(root, "rois");
            int size = cJSON_GetArraySize(data);

            for (int i = 0; i < size; ++i) {
                cJSON *item = cJSON_GetArrayItem(data, i);
                int x = cJSON_GetObjectItem(item, "x")->valueint;
                int y = cJSON_GetObjectItem(item, "y")->valueint;
                pts.push_back(cv::Point(x,y));
            }

            contour.push_back(pts);
            drawContours(roi,contour,0,cv::Scalar::all(255),-1);
            image.copyTo(dst,roi);

            return detector.doInference(dst);
        }
        return detector.doInference(image);
    }
    catch (const char* msg) {
        cJSON* result = cJSON_CreateObject(), * data = cJSON_CreateArray();;
        cJSON_AddNumberToObject(result, "code", 1);
        cJSON_AddStringToObject(result, "msg", msg);
        cJSON_AddItemToObject(result, "data", data);
        return cJSON_PrintUnformatted(result);
    }
}
char* detectByFile(const char* file,const char* rois){
    try {
        cv::Mat image = cv::imread(file);
        if(rois != ""){
            cv::Mat dst;
            cv::Mat roi = cv::Mat::zeros(image.size(),CV_8U);
            std::vector<std::vector<cv::Point>> contour;
            std::vector<cv::Point> pts;

            cJSON *root;
            root = cJSON_Parse(rois);
            cJSON *data = cJSON_GetObjectItem(root, "rois");
            int size = cJSON_GetArraySize(data);

            for (int i = 0; i < size; ++i) {
                cJSON *item = cJSON_GetArrayItem(data, i);
                int x = cJSON_GetObjectItem(item, "x")->valueint;
                int y = cJSON_GetObjectItem(item, "y")->valueint;
                pts.push_back(cv::Point(x,y));
            }

            contour.push_back(pts);
            drawContours(roi,contour,0,cv::Scalar::all(255),-1);
            image.copyTo(dst,roi);

            return detector.doInference(dst);
        }
        return detector.doInference(image);
    }
    catch (const char* msg) {
        cJSON* result = cJSON_CreateObject(), * data = cJSON_CreateArray();;
        cJSON_AddNumberToObject(result, "code", 1);
        cJSON_AddStringToObject(result, "msg", msg);
        cJSON_AddItemToObject(result, "data", data);
        return cJSON_PrintUnformatted(result);
    }
}

/**
 * detection api end
 */
cv::Mat base64ToMat(std::string &base64_data) {
    cv::Mat img;
    std::string s_mat;
    s_mat = base64Decode(base64_data.data(), base64_data.size());
    std::vector<char> base64_img(s_mat.begin(), s_mat.end());
    img = cv::imdecode(base64_img,1);//CV_LOAD_IMAGE_COLOR
    return img;
}

std::string base64Decode(const char *Data, int DataByte) {
    const char DecodeTable[] ={
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            62, // '+'
            0, 0, 0,
            63, // '/'
            52, 53, 54, 55, 56, 57, 58, 59, 60, 61, // '0'-'9'
            0, 0, 0, 0, 0, 0, 0,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, // 'A'-'Z'
            0, 0, 0, 0, 0, 0,
            26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
            39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, // 'a'-'z'
    };
    std::string strDecode;
    int nValue;
    int i = 0;
    while (i < DataByte){
        if (*Data != '\r' && *Data != '\n'){
            nValue = DecodeTable[*Data++] << 18;
            nValue += DecodeTable[*Data++] << 12;strDecode += (nValue & 0x00FF0000) >> 16;
            if (*Data != '='){
                nValue += DecodeTable[*Data++] << 6;strDecode += (nValue & 0x0000FF00) >> 8;
                if (*Data != '='){
                    nValue += DecodeTable[*Data++];
                    strDecode += nValue & 0x000000FF;
                }
            }
            i += 4;
        }
        else{
            Data++;
            i++;
        }
    }
    return strDecode;
}