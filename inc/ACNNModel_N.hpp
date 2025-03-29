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
#include "ACNNModel_B.hpp"

#define INFO_LOG(fmt, ...) fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__);fflush(stdout)
#define ERROR_LOG(fmt, ...)fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)



class ACNNModel_N
{
public:
    ACNNModel_N(const char* modelPath);
    ~ACNNModel_N();
    Result head_initDatasets();
    Result head_Inference(float* input_data0,float* input_data1);
    Result head_GetResults(std::vector<std::vector<float>> &output);
    Result runACNN_N(std::vector<std::vector<float>> &output, float* input_data0, float* input_data1);

private:
    const char* modelPath_;
    uint32_t modelId_;
    aclmdlDesc *modelDesc_;

    aclmdlDataset *inputDataset_n;
    aclmdlDataset *outputDataset_n;
    void* inputBuffer_n1;
    void *outputBuffer_n1;
    size_t inputBufferSize_n1;
    size_t modelOutputSize_n1; 
    void* inputBuffer_n2;
    void *outputBuffer_n2;
    size_t inputBufferSize_n2;
    size_t modelOutputSize_n2; 
};