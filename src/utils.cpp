//
// Created by jiaopan on 2020-11-02.
//


#include "utils.h"



void createInputImage(float* input, cv::Mat image) {
    cv::Mat pr_img = preprocess_img(image); // letterbox BGR to RGB
    int i = 0;
    for (int row = 0; row < Yolo::INPUT_HEIGHT; ++row) {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < Yolo::INPUT_WIDTH; ++col) {
            input[i] = (float)uc_pixel[2] / 255.0;
            input[i + Yolo::INPUT_HEIGHT * Yolo::INPUT_WIDTH] = (float)uc_pixel[1] / 255.0;
            input[i + 2 * Yolo::INPUT_HEIGHT * Yolo::INPUT_WIDTH] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }
}


cv::Mat preprocess_img(cv::Mat& img) {
    int w, h, x, y;
    float r_w = Yolo::INPUT_WIDTH / (img.cols*1.0);
    float r_h = Yolo::INPUT_HEIGHT / (img.rows*1.0);
    if (r_h > r_w) {
        w = Yolo::INPUT_WIDTH;
        h = r_w * img.rows;
        x = 0;
        y = (Yolo::INPUT_HEIGHT - h) / 2;
    }
    else {
        w = r_h * img.cols;
        h = Yolo::INPUT_HEIGHT;
        x = (Yolo::INPUT_WIDTH - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(Yolo::INPUT_HEIGHT, Yolo::INPUT_WIDTH, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    int l, r, t, b;
    float r_w = Yolo::INPUT_WIDTH / (img.cols * 1.0);
    float r_h = Yolo::INPUT_HEIGHT / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (Yolo::INPUT_HEIGHT - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (Yolo::INPUT_HEIGHT - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (Yolo::INPUT_WIDTH - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (Yolo::INPUT_WIDTH - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
            (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
            (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
            (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
            (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) {
    return a.conf > b.conf;
}



void nms(std::vector<Yolo::Detection>& res, float *output, float conf_thresh, float nms_thresh) {
    int det_size = sizeof(Yolo::Detection) / sizeof(float);
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}


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