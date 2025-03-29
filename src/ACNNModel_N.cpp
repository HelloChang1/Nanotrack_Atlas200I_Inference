#include "ACNNModel_N.hpp"
#include <iostream>
using namespace std;
extern bool DEBUG;
ACNNModel_N::ACNNModel_N(const char* modelPath):modelPath_(modelPath)
{
    aclError ret;
    // load model from file
    ret = aclmdlLoadFromFile(modelPath_, &modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclmdlLoadFromFile failed, errorCode is %d", ret);
    }
    INFO_LOG("modelid is %d\n",modelId_);
    // create description of model
    modelDesc_ = aclmdlCreateDesc();
    ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclmdlGetDesc failed, errorCode is %d", ret);
    }
}
ACNNModel_N::~ACNNModel_N()
{
    aclError ret;
  // release resource includes acl resource, data set and unload model
    aclrtFree(inputBuffer_n1);
    inputBuffer_n1= nullptr;
    aclrtFree(inputBuffer_n2);
    inputBuffer_n2= nullptr;
    (void)aclmdlDestroyDataset(inputDataset_n);
    inputDataset_n = nullptr;

    aclrtFree(outputBuffer_n1);
    outputBuffer_n1 = nullptr;
    aclrtFree(outputBuffer_n2);
    outputBuffer_n2 = nullptr;
    (void)aclmdlDestroyDataset(outputDataset_n);
    outputDataset_n = nullptr;

    ret = aclmdlDestroyDesc(modelDesc_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("destroy description failed, errorCode is %d", ret);
    }

    ret = aclmdlUnload(modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("unload model failed, errorCode is %d", ret);
    }     

}

Result ACNNModel_N::head_initDatasets()
{
    aclError ret;
    // create data set of input
    inputDataset_n = aclmdlCreateDataset();
    
    inputBufferSize_n1 = aclmdlGetInputSizeByIndex(modelDesc_, 0);
    const char* inputname_1=aclmdlGetInputNameByIndex(modelDesc_, 0);
    cout<<"head_model inputname_1: "<<inputname_1<<endl;
    aclrtMalloc(&inputBuffer_n1, inputBufferSize_n1, ACL_MEM_MALLOC_HUGE_FIRST);
    aclDataBuffer *inputData_n1 = aclCreateDataBuffer(inputBuffer_n1, inputBufferSize_n1);
    ret = aclmdlAddDatasetBuffer(inputDataset_n, inputData_n1);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("head_initDatasets aclmdlAddDatasetBuffer n1 failed, errorCode is %d", ret);
        return FAILED;
    }else{
        INFO_LOG("head_initDatasets aclmdlAddDatasetBuffer n1 success");

    }

    inputBufferSize_n2 = aclmdlGetInputSizeByIndex(modelDesc_, 1);
    const char* inputname_2=aclmdlGetInputNameByIndex(modelDesc_, 1);
    cout<<"head_model inputname_2: "<<inputname_2<<endl;
    aclrtMalloc(&inputBuffer_n2, inputBufferSize_n2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclDataBuffer *inputData_n2 = aclCreateDataBuffer(inputBuffer_n2, inputBufferSize_n2);
     ret = aclmdlAddDatasetBuffer(inputDataset_n, inputData_n2);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("head_initDatasets aclmdlAddDatasetBuffer n2 failed, errorCode is %d", ret);
        return FAILED;
    }else{
        INFO_LOG("head_initDatasets aclmdlAddDatasetBuffer n2 success");

    }     
     

    // create data set of output
    outputDataset_n = aclmdlCreateDataset();

    modelOutputSize_n1 = aclmdlGetOutputSizeByIndex(modelDesc_, 0);
    aclrtMalloc(&outputBuffer_n1, modelOutputSize_n1, ACL_MEM_MALLOC_HUGE_FIRST);
    aclDataBuffer *outputData_n1 = aclCreateDataBuffer(outputBuffer_n1, modelOutputSize_n1);
    ret = aclmdlAddDatasetBuffer(outputDataset_n, outputData_n1);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclmdlAddDatasetBuffer n1 failed, errorCode is %d", ret);
        return FAILED;
    }else{
        INFO_LOG("aclmdlAddDatasetBuffer n1  success");

    }
    
    modelOutputSize_n2 = aclmdlGetOutputSizeByIndex(modelDesc_, 1);
    aclrtMalloc(&outputBuffer_n2, modelOutputSize_n2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclDataBuffer *outputData_n2 = aclCreateDataBuffer(outputBuffer_n2, modelOutputSize_n2);
    ret = aclmdlAddDatasetBuffer(outputDataset_n, outputData_n2);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclmdlAddDatasetBuffer n2 failed, errorCode is %d", ret);
        return FAILED;
    }else{
        INFO_LOG("aclmdlAddDatasetBuffer n2 success");
    }

    return SUCCESS;
}

Result ACNNModel_N::head_Inference(float* input_data0,float* input_data1)
{
    // copy host datainputs to device
    aclError ret = aclrtMemcpy(inputBuffer_n1, inputBufferSize_n1, input_data0 , inputBufferSize_n1, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("memcpy  failed, errorCode is %d", ret);
        return FAILED;
    }
    ret = aclrtMemcpy(inputBuffer_n2, inputBufferSize_n2, input_data1, inputBufferSize_n2, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("memcpy  failed, errorCode is %d", ret);
        return FAILED;
    }
    // inference
    ret = aclmdlExecute(modelId_, inputDataset_n, outputDataset_n);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("execute model failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result ACNNModel_N::head_GetResults(std::vector<std::vector<float>> &output)
{
    aclError ret;
    void *outHostData = nullptr;
    float *outData = nullptr;
    uint32_t output_num=aclmdlGetNumOutputs(modelDesc_);
    if(DEBUG)
        cout<<"the output_num of model is  "<<output_num<<endl;
    output.resize(output_num); // 假设第一个维度是 batch size
    for(int i=0;i<output_num;++i)
    {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(outputDataset_n, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        uint32_t output_length = aclGetDataBufferSizeV2(dataBuffer);
   
        //查询模型输出数据的维度与数据类型
        aclmdlIODims dims;
        ret = aclmdlGetOutputDims(modelDesc_, i, &dims);// 创建一个aclmdlIODims实例
        if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclmdlGetOutputDims failed, errorCode is %d", ret);
        }
        std::vector<int> dimensions(4); // 创建一个大小为 4 的 vector
        if(DEBUG)
            cout<<"dims.dimCount of the output: "<<dims.dimCount<<endl;
        for (size_t i = 0; i < dims.dimCount; ++i) {
            dimensions[i] = static_cast<int>(dims.dims[i]); // 将int64_t转换为int并存储到vector中
        }
        if(DEBUG)        
            cout<<"dims.dims[i] of the output is : "<<dimensions[0]<<" ; "<<dimensions[1]<<" ; "<<dimensions[2]<<" ; "<<dimensions[3]<<endl;


        aclDataType datatype= aclmdlGetOutputDataType(modelDesc_, i);
        if(DEBUG)
            cout<<"datatype of the output "<<i<< " is "<<datatype<<endl;

        // copy device output data to host
        aclrtMallocHost(&outHostData, output_length);
        ret = aclrtMemcpy(outHostData, output_length, data, output_length, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("memcpy  failed, errorCode is %d", ret);
            return FAILED;
        }
        outData = reinterpret_cast<float*>(outHostData);
        if(DEBUG){
            //输出数据正确性检查
            cout<<"data check of in head_model model output: "<< i <<endl;
            std::vector<float> first_channel_first_row;
            first_channel_first_row.resize(dimensions[3]); // 列的数量等于最后一个维度
            
            // 计算偏移量
            size_t offset = (0 * dimensions[1] + 0) * dimensions[2] * dimensions[3]; // 第一个通道，第一行
            for (size_t col = 0; col < dimensions[3]; ++col) {
            first_channel_first_row[col] = outData[offset + col];
            }

            // 输出结果进行验证
            for (float value : first_channel_first_row) {
            std::cout << value << " ";
            }
        }

        // 计算总元素数
        size_t total_elements = 1;
        for (int dim : dimensions) {
            total_elements *= dim;
        }
        if(DEBUG)
        cout<<"the num of the outdata in float_dtype:  "<< total_elements<<"  in output:  "<<i<<endl;

        output[i].resize(total_elements / dimensions[0]);
        std::memcpy(output[i].data(), outData, (total_elements / dimensions[0]) * sizeof(float));

    }
    return SUCCESS;  

}



Result ACNNModel_N::runACNN_N(std::vector<std::vector<float>> &output, float* input_data0, float* input_data1)
{
    Result ret;

    //推理
    ret = head_Inference(input_data0,input_data1);
    if (ret != SUCCESS) {
        ERROR_LOG("Inference  failed");
        return FAILED;
    }
    ret = head_GetResults(output);
    if (ret != SUCCESS) {
        ERROR_LOG("Inference  failed");
        return FAILED;
    }  
    return SUCCESS;  
}

