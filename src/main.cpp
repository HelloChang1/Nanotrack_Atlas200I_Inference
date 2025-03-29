#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  

#include "acl/acl.h"

#include "nanotrack.hpp"

using namespace cv;
using namespace std;

constexpr bool INFER_ONCE = false;

bool DEBUG = false;

Result ACNNInit(int32_t& deviceId, aclrtContext& context, aclrtStream& stream)
{
    // init acl resource
    const char *aclConfigPath = "";
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclInit failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtSetDevice failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtCreateContext(&context, deviceId);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtCreateContext failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtCreateStream failed, errorCode is %d", ret);
        return FAILED;
    }

    return SUCCESS;
}

Result ACNNDeInit(int32_t& deviceId, aclrtContext& context, aclrtStream& stream)
{
    aclError ret;

    if (stream != nullptr) {
        ret = aclrtDestroyStream(stream);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("aclrtDestroyStream failed, errorCode is %d", ret);
        }
        stream = nullptr;
    }

    if (context != nullptr) {
        ret = aclrtDestroyContext(context);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("aclrtDestroyContext failed, errorCode is %d", ret);
        }
        context = nullptr;
    }

    ret = aclrtResetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtResetDevice failed, errorCode is %d", ret);
    }

    ret = aclFinalize();
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclFinalize failed, errorCode is %d", ret);
    }
}

bool fileExists(const std::string& filePath) {
    std::ifstream file(filePath);
    return file.good();
}

int main(int argc, char** argv)
{
    /*---------------------------------------------------ACNNInit---------------------------------------------------*/
    int32_t deviceId_=0;
    aclrtContext context_=nullptr;
    aclrtStream stream_=nullptr;
    Result ret = ACNNInit(deviceId_, context_, stream_);
    if (ret != SUCCESS) {
        ERROR_LOG("ACNNInit  failed");
        return FAILED;
    }
    else{INFO_LOG("ACNNInit SUCCES");}


    /*---------------------------------------------------Nanotrack 加载模型---------------------------------------------------*/

    const char* T_backbone_model = "../model/nanotrack_backbone_127.om";
    const char* X_backbone_model = "../model/nanotrack_backbone_255.om";
    const char* head_model = "../model/nanotrack_head.om"; 
     if (!fileExists(T_backbone_model)) {
        std::cerr << "Model file does not exist: " << T_backbone_model << std::endl;
        return 1;
    }
    if (!fileExists(X_backbone_model)) {
        std::cerr << "Model file does not exist: " << X_backbone_model << std::endl;
        return 1;
    }

    if (!fileExists(head_model)) {
        std::cerr << "Model file does not exist: " << head_model << std::endl;
        return 1;
    }

    NanoTrack nanotrack(T_backbone_model, X_backbone_model, head_model);
    INFO_LOG(" nanotrack create SUCCES");
    nanotrack.initsource();
    INFO_LOG(" nanotrack initsource SUCCES");

    
    if(INFER_ONCE)
    {
        // 指定推理图片路径
        string image_path0 = "../image/frame_0000.png";
        string image_path2 = "../image/frame_0002.png";

        // 使用 OpenCV 读取图片
        Mat first_frame = cv::imread(image_path0);
        Mat track_frame = cv::imread(image_path2);
        Mat temp_frame=first_frame;
        std::vector<float> img_bbox;
        // 检查图片是否正确读取
        if (first_frame.empty()) {
            std::cerr << "Error: image_path0 image not found or unable to load." << std::endl;
            return -1;
        }
            Rect bbox(275, 149, 62, 60); //gir_dance.mp4
            // Rect bbox(706, 679, 155, 141);
            // 绘制边界框
            rectangle(temp_frame, bbox, cv::Scalar(0, 255, 0), 2);
            // 保存第一帧到本地
            string output_path = "../results/INFER_ONCE_first_frame_with_bbox.jpg";
            if (!imwrite(output_path, temp_frame)) {
                cerr << "Error: Could not save the INFER_ONCE first frame to " << output_path << endl;
                return -1;
            }
            cout << "First frame with bounding box saved to " << output_path << endl;
            nanotrack.init(first_frame, bbox);
            INFO_LOG(" nanotrack.init SUCCES");
            double t1 = getTickCount();
            img_bbox = nanotrack.track(track_frame);
            double t2 = getTickCount();
            double process_time_ms = (t2 - t1) * 1000 / getTickFrequency();
            double fps_value = getTickFrequency() / (t2 - t1);
            cout << "每帧处理时间: " << process_time_ms << " ms, FPS: " << fps_value << endl;
            cv::Rect resultBox(img_bbox[0], img_bbox[1],img_bbox[2],img_bbox[3]);
            // 绘制边界框
            rectangle(track_frame,resultBox, cv::Scalar(0, 255, 0), 3);
            imwrite("../results/INFER_ONCE_result.jpg",track_frame);
            INFO_LOG(" Rectangle frame SUCCES");
    }
    else
    {
        /*------------------------------------------------------视频帧初始化-------------------------------------------------------*/
        const string video_path = "../data/girl_dance.mp4";
        int frame_count=1;
        // const string video_path = "../data/girl_dance.mp4";
        VideoCapture cap(video_path);
        // 检查视频是否成功打开
        if (!cap.isOpened()) {
            cerr << "Error: Could not open video." << endl;
            return -1;
        }
        // 获取视频帧率和帧尺寸
        double fps = cap.get(CAP_PROP_FPS);
        int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

        // 创建视频写入器
        VideoWriter video_writer("../results/output_video.avi", VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, Size(width, height));
        // 读取一帧初始帧
        Mat frame;
        if (!cap.read(frame)) {
            cerr << "Error: Could not read the first frame." << endl;
            return -1;
        }
        // [706, 679, 155, 141]
        Rect bbox(275, 149, 62, 60);
        // Rect bbox(706, 679, 155, 141);
        // 绘制边界框
        rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);
        // 保存第一帧到本地
        string output_path = "../results/first_frame_with_bbox.jpg";
        if (!imwrite(output_path, frame)) {
            cerr << "Error: Could not save the first frame to " << output_path << endl;
            return -1;
        }
        cout << "First frame with bounding box saved to " << output_path << endl;
        nanotrack.init(frame, bbox);
        INFO_LOG(" nanotrack.init SUCCES");
    
        std::vector<float> video_bbox;
    /*------------------------------------------------------视频帧初始化-------------------------------------------------------*/

    /*------------------------------------------------------视频帧追踪-------------------------------------------------------*/
        while (true) 
        {
            if (!cap.read(frame)) {
                cout << "End of video or unable to read frame." << endl;
                break;
            }


            double t1 = getTickCount();
            video_bbox = nanotrack.track(frame);
            double t2 = getTickCount();
            double process_time_ms = (t2 - t1) * 1000 / getTickFrequency();
            double fps_value = getTickFrequency() / (t2 - t1);
            cout << "每帧处理时间: " << process_time_ms << " ms, FPS: " << fps_value << endl;
            cv::Rect resultBox(video_bbox[0], video_bbox[1], video_bbox[2], video_bbox[3]);
            if (resultBox.x < 0 || resultBox.y < 0 || resultBox.x + resultBox.width > frame.cols || resultBox.y + resultBox.height > frame.rows) {
                std::cerr << "Error: ROI is out of frame bounds!" << std::endl;
                return -1;
            }
            // 绘制边界框
            rectangle(frame,resultBox, cv::Scalar(0, 255, 0), 3);
            INFO_LOG(" Rectangle frame SUCCES");

            // 写入视频
            video_writer.write(frame);
            frame_count++;
            cout<<"now frame  is "<<frame_count<<endl;
        }
        // 释放资源
        video_writer.release();
        cap.release();
    }


    /*------------------------------------------------------视频帧追踪-------------------------------------------------------*/
    ret = ACNNDeInit(deviceId_, context_, stream_);
    if (ret != SUCCESS) {
        ERROR_LOG("ACNNDeInit  failed");
        return FAILED;
    }
    else{INFO_LOG("ACNNDeInit SUCCES");}

    return 0;
}
 