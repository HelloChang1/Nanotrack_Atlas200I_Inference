#include <iostream>
#include <cstdlib>
#include <string>
#include "nanotrack.hpp"
#include <numeric> // for std::accumulate
#include <cmath> // for std::max and std::sqrt

using namespace std;
extern bool DEBUG;


std::vector<float> convert_score(const std::vector<float> &input) {
    size_t input_size = input.size();
    size_t cal_nums = input_size / 2;
    if(DEBUG)
    {
        std::cout << "the convert_score nums in cls_result: " << cal_nums << std::endl;
    }
    std::vector<float> output(cal_nums); // 预分配内存

    for (size_t i = 0; i < cal_nums; ++i) {
        float exp1 = std::exp(input[i]); // 通道 1 的得分
        float exp2 = std::exp(input[i + cal_nums]); // 通道 2 的得分
        float sum_exp = exp1 + exp2; // 两个通道的 exp 值之和

        output[i] = exp2 / sum_exp; // 计算通道 2 的 softmax 值
    }

    return output;
}


NanoTrack::NanoTrack(const char* modelPath_1,const char* modelPath_2,const char* modelPath_3):
                                g_modelPath_1 (modelPath_1), g_modelPath_2 (modelPath_2), g_modelPath_3 (modelPath_3),
                                module_T127(modelPath_1), module_X255(modelPath_2), net_head(modelPath_3)
{   
  
}


NanoTrack::~NanoTrack()
{

}


void NanoTrack::initsource()
{   
    Result ret;
        //模型类构造时就进行模型加载，输入输出内存分配


    ret = module_T127.backbone_initDatasets();
    if (ret != SUCCESS) {
        ERROR_LOG("module_T127.backbone_initDatasets failed ");
    }
    ret = module_X255.backbone_initDatasets();
    if (ret != SUCCESS) {
        ERROR_LOG("module_X255.backbone_initDatasets failed ");
    }
    ret=net_head.head_initDatasets();
    if (ret != SUCCESS) {
        ERROR_LOG("net_head.head_initDatasets failed ");
    }
}


//  bbox(275, 149, 62, 60);
void NanoTrack::init(cv::Mat img, cv::Rect bbox) 
{
    create_window(); 

    create_grids(); 

    cv::Point2f target_pos ={0.f, 0.f}; // cx, cy
    cv::Point2f target_sz = {0.f, 0.f}; //w,h
    
    // 目标框中心点坐标
    // 305.5
    target_pos.x = bbox.x + ((bbox.width - 1.0f) / 2.0f);    
    // 178.5
    target_pos.y = bbox.y + ((bbox.height - 1.0f) / 2.0f); 
    // 目标框的宽高
    // 62
    target_sz.x=bbox.width;
    // 60
    target_sz.y=bbox.height;

    cout<<"bbox"<<bbox<<endl;
    cout<<"target_pos"<<target_pos<<endl;
    cout<<"target_sz"<<target_sz<<endl;
    // 303
    float wc_z = target_sz.x + cfg.context_amount * (target_sz.x + target_sz.y);
    //
    float hc_z = target_sz.y + cfg.context_amount * (target_sz.x + target_sz.y);
    //122
    float s_z = round(sqrt(wc_z * hc_z));  
    if(DEBUG){
        cout<<"the data check of the wc_z, hc_z, s_z: "<<wc_z<<hc_z<<s_z<<endl;
        
        cout<<"the data check of the img_127"<<endl;
        for (int i = 0; i < img.cols; ++i) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(80, i);  // 第一行第 i 列
            uchar b_value = pixel[0];  // B 通道
            std::cout << static_cast<int>(b_value) << " ";
        }
        std::cout << std::endl;
    }


    // 对于多通道图像（如 RGB 图像），  cv::Scalar   的值是一个包含多个通道平均值的向量。
    cv::Scalar avg_chans = cv::mean(img);
    cv::Mat z_crop;
    // (img,(784,250),127,296,avg_chans(3))
    z_crop  = get_subwindow_tracking(img, target_pos, cfg.exemplar_size, int(s_z),avg_chans); //cv::Mat BGR order 
    if(DEBUG){    
        cout<<"the data check of the z_crop"<<endl;
        int zount=0;
        assert(z_crop.rows == 127 && z_crop.cols == 127 && z_crop.channels() == 3);
        for (int i = 0; i < z_crop.cols; ++i) {
            cv::Vec3b pixel = z_crop.at<cv::Vec3b>(0, i);  // 第一行第 i 列
            uchar b_value = pixel[0];  // B 通道
            std::cout << static_cast<int>(b_value) << " ";
            zount++;
        }
        cout<<"the data num  of the check of z_crop is :"<<zount<<endl;
        std::cout << std::endl;
    }

    vector<vector<float>> acnnOutputs;
    INFO_LOG("START module_T127.runACNN  run ");
    Result ret = module_T127.runACNN_B(acnnOutputs, z_crop);
    if (ret != SUCCESS) {
        ERROR_LOG("module_T127.runACNN  run  failed");
    }    
    INFO_LOG("FINISH module_T127.runACNN  run ");


    this->result_T = acnnOutputs[0];
    this->state.channel_ave=avg_chans;
    this->state.im_h=img.rows;
    this->state.im_w=img.cols;
    this->state.target_pos=target_pos;
    this->state.target_sz= target_sz; 
}


// 实现 change 函数
float change(float r) {
    return std::max(r, 1.0f / r);
}

// 实现 sz 函数
float sz(float w, float h) {
    float pad = (w + h) * 0.5f;
    return std::sqrt((w + pad) * (h + pad));
}

std::vector<float> NanoTrack::track(cv::Mat& im)
{
    if(DEBUG){cout<<" this->state.target_pos: "<< this->state.target_pos<<endl;}
    cv::Point2f target_pos = this->state.target_pos;
    cv::Point2f target_sz = this->state.target_sz;
    // context_amount：0.5
    float wc_z = target_sz.x + cfg.context_amount * (target_sz.x + target_sz.y);
    float hc_z = target_sz.y + cfg.context_amount * (target_sz.x + target_sz.y);
    float s_z = sqrt(wc_z * hc_z);  
    if(DEBUG){cout<<"track s_z: "<<s_z<<endl;}
    // exemplar_size:127
    float scale_z = cfg.exemplar_size / s_z;  

    float d_search = (cfg.instance_size - cfg.exemplar_size) / 2; 
    float pad = d_search / scale_z; 
    float s_x = s_z + 2*pad;
    cv::Mat x_crop; 
    if(DEBUG){
        cout<<"track s_x: "<<s_x<<endl;
        cout<<"track target_pos: "<<target_pos<<endl;
    }
    x_crop  = get_subwindow_tracking(im, target_pos, cfg.instance_size, std::round(s_x),state.channel_ave);

    // 图像255的输入
    vector<vector<float>> acnnOutputs;
    INFO_LOG("START module_X255.runACNN  run ");
    Result ret = module_X255.runACNN_B(acnnOutputs, x_crop);
    if (ret != SUCCESS) {
        ERROR_LOG("module_X255.runACNN  run  failed");
    }    
    INFO_LOG("FINISH module_X255.runACNN  run ");
    

    this->result_X = acnnOutputs[0];
    if(DEBUG){
        //输出数据正确性检查
        cout<<"data check of this->result_X:"<<endl;
        std::vector<float> first_channel_first_row;
        first_channel_first_row.resize(16); // 列的数量等于最后一个维度
        // 计算偏移量
        for (size_t col = 0; col < 16; ++col) {
        first_channel_first_row[col] = result_X[col];
        }
        // 输出结果进行验证
        for (float value : first_channel_first_row) {
        std::cout << value << " ";
        }
    }

    // 创建指向 result_T_transposedVec 的指针
    float* ptr_T = result_T.data();
    // 创建指向 result_X_transposedVec 的指针
    float* ptr_X = result_X.data();


    vector<vector<float>> acnnOutputs_2;
    cv::Mat emptyMat; // 创建一个空的 cv::Mat 对象
    INFO_LOG("START net_head.runACNN_Head  run ");
    ret = net_head.runACNN_N(acnnOutputs_2,ptr_T,ptr_X);
    if (ret != SUCCESS) {
        ERROR_LOG("net_head.runACNN_Head  run  failed");
    }
    INFO_LOG("FINISH net_head.runACNN_Head  run ");


    vector<float> cls_score_result = acnnOutputs_2[0];
    vector<float> bbox_pred_result = acnnOutputs_2[1];

    if(DEBUG){    
        //输出数据正确性检查
        cout<<"data check of cls_score_result:"<<endl;
        std::vector<float> cls_first_channel_first_row;
        cls_first_channel_first_row.resize(15); // 列的数量等于最后一个维度
        // 计算偏移量
        for (size_t col = 0; col < 15; ++col) {
        cls_first_channel_first_row[col] = cls_score_result[col];
        }
        // 输出结果进行验证
        for (float value : cls_first_channel_first_row) {
        std::cout << value << " ";
        }
        std::cout << std::endl;


        //输出数据正确性检查
        cout<<"data check of bbox_score_result:"<<endl;
        std::vector<float> bbox_first_channel_first_row;
        bbox_first_channel_first_row.resize(15); // 列的数量等于最后一个维度
        // 计算偏移量
        for (size_t col = 0; col < 15; ++col) {
        bbox_first_channel_first_row[col] = bbox_pred_result[col];
        }
        // 输出结果进行验证
        for (float value : bbox_first_channel_first_row) {
        std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    int cols = 15; 
    int rows = 15; 

    vector<float> cls_scores = convert_score(cls_score_result);
    if(DEBUG){    
        cout<<"data check of cls_scores:"<<endl;
        // 输出结果进行验证
        for (float value : cls_scores) {
        std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    std::vector<float> pred_x1(cols*rows, 0), pred_y1(cols*rows, 0), pred_x2(cols*rows, 0), pred_y2(cols*rows, 0);
    std::vector<float> pred_xc(cols*rows, 0), pred_yc(cols*rows, 0), pred_w(cols*rows, 0), pred_h(cols*rows, 0);

    // {1,4,15,15}
    float* bbox_pred_data = (float*) bbox_pred_result.data();    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // 计算 xyxy 格式的坐标
            pred_x1[i * cols + j] = this->grid_to_search_x[i * cols + j] - bbox_pred_data[i * cols + j];
            pred_y1[i * cols + j] = this->grid_to_search_y[i * cols + j] - bbox_pred_data[i * cols + j + 15 * 15 * 1];
            pred_x2[i * cols + j] = this->grid_to_search_x[i * cols + j] + bbox_pred_data[i * cols + j + 15 * 15 * 2];
            pred_y2[i * cols + j] = this->grid_to_search_y[i * cols + j] + bbox_pred_data[i * cols + j + 15 * 15 * 3];

            // 转换为 xywh 格式
            float x_c = (pred_x1[i * cols + j] + pred_x2[i * cols + j]) / 2.0f;
            float y_c = (pred_y1[i * cols + j] + pred_y2[i * cols + j]) / 2.0f;
            float w = pred_x2[i * cols + j] - pred_x1[i * cols + j];
            float h = pred_y2[i * cols + j] - pred_y1[i * cols + j];

            // 存储 xywh 格式的坐标
            pred_xc[i * cols + j] = x_c;  // 中心点 x 坐标
            pred_yc[i * cols + j] = y_c;  // 中心点 y 坐标
            pred_w[i * cols + j] = w;     // 宽度
            pred_h[i * cols + j] = h;     // 高度
        }
    }
    if(DEBUG){    
        cout<<"data check of pred_xc:"<<endl;
        // 输出结果进行验证
        for (float value : pred_xc) {
        std::cout << value << " ";
        }
        std::cout << std::endl;
        cout<<"target_sz.x,target_sz.y: "<<target_sz.x<<target_sz.y<<scale_z<<endl;
    }

    std::vector<float> s_c;
    std::vector<float> r_c;
    float radio = target_sz.x/target_sz.y;
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {        
            s_c.push_back(change(sz(pred_w[i*cols+j],pred_h[i*cols+j])/sz(target_sz.x * scale_z,target_sz.y * scale_z )));
            
            r_c.push_back(change(radio / (pred_w[i*cols+j]/pred_h[i*cols+j])));
        }
    }

    if(DEBUG){   
        cout<<"data check of s_c:"<<endl;
        // 输出结果进行验证
        for (float value : s_c) {
        std::cout << value << " ";
        }
        std::cout << std::endl; 

        cout<<"data check of r_c:"<<endl;
        // 输出结果进行验证
        for (float value : r_c) {
        std::cout << value << " ";
        }
        std::cout << std::endl; 
    }

    std::vector<float> penalty(rows*cols,0);
    for (int i = 0; i < rows * cols; i++)
    {
        penalty[i] = std::exp(-1 * (s_c[i] * r_c[i]-1) * cfg.penalty_k);
    } 

    if(DEBUG){   
        cout<<"data check of penalty:"<<endl;
        // 输出结果进行验证
        for (float value : penalty) {
        std::cout << value << " ";
        }
        std::cout << std::endl; 
    }

    // window penalty 窗口惩罚
     std::vector<float> pscore(rows*cols,0);
    // int r_max = 0, c_max = 0; 
    float maxScore = 0; 

    int max_idx = 0;
    for (int i = 0; i < rows * cols; i++)
    {
        pscore[i] = (penalty[i] * cls_scores[i]) * (1 - cfg.window_influence) + this->window[i] * cfg.window_influence; 
        if (pscore[i] > maxScore) 
        {
            // get max 
            maxScore = pscore[i]; 
            max_idx = i;
        }
    } 
    // 存储 xywh 格式的坐标
    float max_predxc=pred_xc[max_idx];  // 中心点 x 坐标
    float max_predyc=pred_yc[max_idx];  // 中心点 y 坐标
    float max_predw=pred_w[max_idx];     // 宽度
    float max_predh=pred_h[max_idx];     // 高度
    float lr = penalty[max_idx] * cls_scores[max_idx] * cfg.lr;
    
    if(DEBUG){   
        cout<<"max_idx: "<<max_idx<<endl;
        cout<<"max bbox:"<<max_predxc<<" , "<<max_predyc<<" , "<<max_predw<<" , "<<max_predh<<endl;
        // size learning rate
        cout<<"lr"<<lr<<endl;
    }
    
    float cx = max_predxc+target_pos.x;
    float cy = max_predyc+target_pos.y;

    float width = target_sz.x * (1 - lr) + max_predw * lr;
    float height = target_sz.x * (1 - lr) + max_predh * lr;
    
    if(DEBUG){   
        cout<<"data check of cx, cy ,width ,height:"<<endl;
        cout<<"new bbox:"<<cx<<" , "<<cy<<" , "<<width<<" , "<<height<<endl;
    }


    // box_clip
    cx = std::max(0.0f, min(state.im_w-1, cx));
    cy = std::max(0.0f, min(state.im_h-1, cy));
    width = float(std::max(10.0f, min(state.im_w, width)));
    height = float(std::max(10.0f, min(state.im_h, height)));
    
    // cout<<"new bbox cliped:"<<cx<<" , "<<cy<<" , "<<width<<" , "<<height<<endl;


    target_pos.x = cx;
    target_pos.y = cy;
    target_sz.x = width;
    target_sz.y = height;

    state.target_pos = target_pos;
    state.target_sz = target_sz;

    // 计算 bbox 的左上角坐标
    float x1 = std::max(0.0f, cx - width / 2);
    float y1 = std::max(0.0f, cy - height / 2);
    float x2 = std::min(state.im_w, cx + width / 2);
    float y2 = std::min(state.im_h, cy + height / 2);

    // 确保宽度和高度至少为 10
    width = std::max(10.0f, x2 - x1);
    height = std::max(10.0f, y2 - y1);

    cout<<"new bbox cliped:"<<x1<<" , "<<y1<<" , "<<width<<" , "<<height<<endl;

    // 更新 bbox
    std::vector<float> bbox;
    bbox.push_back(x1);
    bbox.push_back(y1);
    bbox.push_back(width);
    bbox.push_back(height);

    float cls_score_max = cls_scores[max_idx];

    return bbox;
}

void NanoTrack::create_window()
{
    int score_size= cfg.score_size;
    std::vector<float> hanning(score_size,0);
    this->window.resize(score_size*score_size, 0);

    for (int i = 0; i < score_size; i++)
    {
        float w = 0.5f - 0.5f * std::cos(2 * 3.1415926535898f * i / (score_size - 1));
        hanning[i] = w;
    } 
    for (int i = 0; i < score_size; i++)
    {
        for (int j = 0; j < score_size; j++)
        {
            this->window[i*score_size+j] = hanning[i] * hanning[j]; 
        }
    }  

}

// 生成每一个格点的坐标 
void NanoTrack::create_grids()
{
    /*
    each element of feature map on input search image
    :return: H*W*2 (position for each element)
    */
    int sz = cfg.score_size;   //16x16
    int ori=0;
    ori = ori-round(sz/2)*cfg.total_stride;
    cout<<"ori:"<<ori<<endl;
    this->grid_to_search_x.resize(sz * sz, 0);
    this->grid_to_search_y.resize(sz * sz, 0);

    for (int i = 0; i < sz; i++)
    {
        for (int j = 0; j < sz; j++)
        {
            this->grid_to_search_x[i*sz+j] = ori+j*cfg.total_stride;   
            this->grid_to_search_y[i*sz+j] = ori+i*cfg.total_stride;
        }
    }
    if(DEBUG){
        cout<<"data check of grid_to_search_x: "<<endl;
        for(float value:grid_to_search_x)
        {
            cout<<value<<" ";
        }
        cout << endl;
        cout<<"data check of grid_to_search_y: "<<endl;
        for(float value:grid_to_search_y)
        {
            cout<<value<<" ";
        }
        cout << endl;
    }
}
// 其目的是从输入图像im中提取初始目标子窗口，并将其调整到指定的大小 model_sz。
/*
• 输入参数：
        •   cv::Mat im  ：输入图像。
        •   cv::Point2f pos  ：目标中心点的位置。
        •   int model_sz  ：最终输出的子窗口大小。
        •   int original_sz  ：原始子窗口的大小。
        •   cv::Scalar channel_ave  ：用于填充的通道平均值。
*/
cv::Mat NanoTrack::get_subwindow_tracking(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz,cv::Scalar channel_ave)
{
    float c = (float)(original_sz + 1) / 2;
    int context_xmin = std::round(pos.x - c);
    int context_xmax = context_xmin + original_sz - 1;
    int context_ymin = std::round(pos.y - c);
    int context_ymax = context_ymin + original_sz - 1;

    int left_pad = int(std::max(0, -context_xmin));
    int top_pad = int(std::max(0, -context_ymin));
    int right_pad = int(std::max(0, context_xmax - im.cols + 1));
    int bottom_pad = int(std::max(0, context_ymax - im.rows + 1));
   
    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;

    cv::Mat im_path_original;

    if (top_pad > 0 || left_pad > 0 || right_pad > 0 || bottom_pad > 0)
    {
        cv::Mat te_im = cv::Mat::zeros(im.rows + top_pad + bottom_pad, im.cols + left_pad + right_pad, CV_8UC3);
       
        cv::copyMakeBorder(im, te_im, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, channel_ave);
        im_path_original = te_im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }
    else
    {
        if(DEBUG)
            std::cout << "top_pad left_pad right_pad bottom_pad==0" << std::endl;
        im_path_original = im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }
    if(DEBUG)
    {
        // 打印 im_path_original 的形状信息
        if (!im_path_original.empty()) {
            std::cout << "Shape (height, width, channels): (" << im_path_original.rows
                    << ", " << im_path_original.cols
                    << ", " << im_path_original.channels()
                    << ")" << std::endl;
        } else {
            std::cout << "im_path_original is empty." << std::endl;
        }
    }
    cv::Mat im_path;
    cv::resize(im_path_original, im_path, cv::Size(model_sz, model_sz));

    return im_path; 
}