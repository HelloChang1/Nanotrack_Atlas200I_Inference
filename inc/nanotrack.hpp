#ifndef NANOTRACK_H
#define NANOTRACK_H

#include <vector> 
#include <map>  
 #include <cmath> 

#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 

#include <string>

#include "acl/acl.h"
#include "ACNNModel_B.hpp"
#include "ACNNModel_N.hpp"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <iostream>
#include <dirent.h>
#include <algorithm>
#include <fstream>


#define PI 3.1415926 

using namespace cv;

struct Config{ 
    
    std::string windowing = "cosine";
    std::vector<float> window;

    int stride = 16;
    float penalty_k = 0.138;
    float window_influence = 0.455;
    float lr = 0.348;
    int exemplar_size=127;
    int instance_size=255;
    int total_stride=16;
    int score_size=15;
    float context_amount = 0.5;
};

struct State { 
    float im_h; 
    float im_w;  
    cv::Scalar channel_ave; 
    cv::Point2f target_pos= {0.f, 0.f}; 
    cv::Point2f target_sz = {0.f, 0.f}; 
    float cls_score_max; 
};

class NanoTrack {

public: 
    
    NanoTrack(const char* modelPath_1,const char* modelPath_2,const char* modelPath_3);   
    ~NanoTrack(); 
 
    void init(cv::Mat img, cv::Rect bbox);
    void initsource();
        
    std::vector<float> track(cv::Mat& im);
    
    std::vector<float> result_T, result_X;

    // state  dynamic
    State state;
    
    // config static
    Config cfg; 

    const float mean_vals[3] = { 0.485f*255.f, 0.456f*255.f, 0.406f*255.f };  
    const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};
    
    ACNNModel_B module_T127;
    ACNNModel_B module_X255;
    ACNNModel_N net_head;

private:

    const char* g_modelPath_1;
    const char* g_modelPath_2;
    const char* g_modelPath_3;

    void create_grids(); 
    void create_window();  
    cv::Mat get_subwindow_tracking(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz,cv::Scalar channel_ave);

    std::vector<float> grid_to_search_x;
    std::vector<float> grid_to_search_y;
    std::vector<float> window;
};

#endif 