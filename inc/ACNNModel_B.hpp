#pragma once
#include <string.h>
#include <string>
#include <vector>
#include <dirent.h>
#include "acl/acl.h"
#include <cstring> // for memcpy
#include <stdexcept> // for std::runtime_error
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 

#define INFO_LOG(fmt, ...) fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__);fflush(stdout)
#define ERROR_LOG(fmt, ...)fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

namespace {
    const float min_chn_0 = 123.675;
    const float min_chn_1 = 116.28;
    const float min_chn_2 = 103.53;
    const float var_reci_chn_0 = 0.0171247538316637;
    const float var_reci_chn_1 = 0.0175070028011204;
    const float var_reci_chn_2 = 0.0174291938997821;
}


class ACNNModel_B
{
public:
    ACNNModel_B(const char* modelPath);
    ~ACNNModel_B();
    Result backbone_initDatasets();
    Result backbone_ProcessInput(cv::Mat& img);
    Result backbone_Inference();
    Result backbone_GetResults(std::vector<std::vector<float>> &output);

    Result runACNN_B(std::vector<std::vector<float>> &output, cv::Mat& img);
private:
    const char* modelPath_;
    uint32_t modelId_;
    aclmdlDesc *modelDesc_;

    aclmdlDataset *inputDataset_b;
    aclmdlDataset *outputDataset_b;
    void* inputBuffer_b;
    void *outputBuffer_b;
    size_t inputBufferSize_b;
    size_t modelOutputSize_b; 
    float* imageBytes;
};