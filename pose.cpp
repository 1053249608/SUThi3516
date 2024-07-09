#include <stdlib.h>
#include <string.h>

#include <errno.h>
#include <termios.h>

#include <iostream>
#include <sstream>
#include <cstdio>

#include <iomanip>
#include <string>
#include "hand_classify.h"
#include "sample_comm_nnie.h"
#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include "hi_comm_video.h"
#include "sample_comm_nnie.h"
#include "sample_media_ai.h"
#include "ai_infer_process.h"
#include "yolov2_hand_detect.h"
#include "vgs_img.h"
#include "ive_img.h"
#include "misc_util.h"
#include "hisignalling.h"
#ifdef __cplusplus
}
#endif


#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include "crop.h"
#include "tennis_detect.h"

#include "hand_classify.h"

#include "detect.h"
#include "detectg.h"

#define HAND_FRM_WIDTH 216
#define HAND_FRM_HEIGHT 384
#define DETECT_OBJ_MAX 32
#define RET_NUM_MAX 4
    // Draw the width of the line
#define WIDTH_LIMIT 32
#define HEIGHT_LIMIT 32
// #define IMAGE_WIDTH        224  // The resolution of the model IMAGE sent to the classification is 224*224
// #define IMAGE_HEIGHT       224
#define DRAW_RETC_THICK 2
#define YOLO_MIN(a, b) ((a) > (b) ? (b) : (a))
#define YOLO_MAX(a, b) ((a) < (b) ? (b) : (a))
// 假设的相机参数  
#define FOCAL_LENGTH 3.6 // 焦距，单位：毫米  
#define PIXEL_SIZE 0.00285 // 像素的物理尺寸，单位：毫米/像素  

// 物品的实际尺寸（这里假设为100毫米，例如一个10厘米的物体）  
#define REAL_DISTANCE_OF_OBJECT 135.0 // 单位：毫米 


    static int biggestBoxIndex;

    static DetectObjInfo objs[DETECT_OBJ_MAX] = {0};
    static RectBox boxs[DETECT_OBJ_MAX] = {0};
    static RectBox objBoxs[DETECT_OBJ_MAX] = {0};
    static RectBox remainingBoxs[DETECT_OBJ_MAX] = {0};
    static RectBox cnnBoxs[DETECT_OBJ_MAX] = {0}; // Store the results of the classification network
    static RecogNumInfo numInfo[RET_NUM_MAX] = {0};
    static IVE_IMAGE_S imgIn;
    static IVE_IMAGE_S imgDst;
    static VIDEO_FRAME_INFO_S frmIn;
    static VIDEO_FRAME_INFO_S frmDst;
    int uartFd = 0;
    extern int num_distance;
    extern int threshold_distance;

    static IVE_IMAGE_S img;
    static HI_S32 s32FrmCnt;

#ifdef __cplusplus
extern "C" {
#endif
struct DetectedPoints {
    std::vector<cv::Point> points;
};

DetectedPoints g_detectedPoints;
#ifdef __cplusplus
}
#endif
using namespace cv; 




std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat>  pose(const cv::Mat& img_pts, const cv::Mat& world_pts, double fx, double fy, double cx, double cy) {
    // 计算内参矩阵
    cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    // 计算内参矩阵的逆
    cv::Mat intrinsic_inv = intrinsic.inv();
 
    // 打印内参矩阵和其逆矩阵
    std::cout << "Intrinsic Matrix :\n" << intrinsic << "\n\n";
    std::cout << "Inverse of Intrinsic Matrix :\n" << intrinsic_inv << "\n\n";
    // 将图像点从像素坐标转换为归一化坐标
    cv::Mat img_pts_prime(img_pts.size(), CV_64F);
    for (int i = 0; i < img_pts.rows; ++i) {
        cv::Mat point = img_pts.row(i).t();
        cv::Mat new_point = intrinsic_inv * point;
        img_pts_prime.row(i) = new_point.t();
    }
    // 打印归一化坐标的图像点
    std::cout << "Normalized Image Points:\n";
    for (int i = 0; i < img_pts_prime.rows; ++i) {
        for (int j = 0; j < img_pts_prime.cols; ++j) {
            std::cout << img_pts_prime.at<double>(i, j) << " ";
        }
        std::cout << "\n";
    }
    // 计算单应性矩阵
    cv::Mat homography_matrix = cv::findHomography(img_pts_prime, world_pts);
    cv::Mat homography_matrix_normalised = homography_matrix / cv::norm(homography_matrix);
    // 打印单应性矩阵
    std::cout << "Homography Matrix:\n" << homography_matrix << "\n\n";
    // 打印归一化的单应性矩阵
    std::cout << "Normalized Homography Matrix:\n" << homography_matrix_normalised << "\n\n";

    // 计算旋转矩阵
    cv::Mat rotation_matrix = cv::Mat::zeros(3, 3, CV_64F);
    cv::Mat r1 = homography_matrix_normalised.col(0).clone();
    cv::Mat r2 = homography_matrix_normalised.col(1).clone();
    cv::Mat r3 = r1.cross(r2);

    std::cout << "r1:\n" << r1 << "\n\n";
    std::cout << "r2:\n" << r2 << "\n\n";
    std::cout << "r3:\n" << r3 << "\n\n";

    r1.copyTo(rotation_matrix.col(0));
    r2.copyTo(rotation_matrix.col(1));
    r3.copyTo(rotation_matrix.col(2));

    // 计算平移矩阵
    cv::Mat translation_matrix = -(rotation_matrix.inv() * homography_matrix_normalised.col(2));

    // 打印旋转矩阵和平移矩阵
    std::cout << "ROTATION MATRIX :\n" << rotation_matrix << "\n\n";
    std::cout << "TRANSLATION MATRIX :\n" << translation_matrix << "\n\n";

    // 计算单应性矩阵的逆
    // cv::Mat homography_matrix_inv = homography_matrix.inv();

    // 调用 reproject 函数（需要你提供具体实现）
    // reproject(world_pts, img_pts, homography_matrix_inv, intrinsic);
    return std::make_tuple(rotation_matrix, translation_matrix, intrinsic_inv, homography_matrix);
}


// 计算两点之间的距离
double distance(const cv::Point& p1, const cv::Point& p2) {
    return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}



int cnt;
HI_S32 Yolo2HandDetectResnetClassifyCal2(uintptr_t model, VIDEO_FRAME_INFO_S *srcFrm, VIDEO_FRAME_INFO_S *dstFrm, int num_1, char *uartRead_1, char *uartRead_2)
{
    SAMPLE_SVP_NNIE_CFG_S *self = (SAMPLE_SVP_NNIE_CFG_S *)model;
    HI_S32 resLen = 0;
    int objNum;
    int ret;
    int result;
    float left_x = 160.0f;
    float up_y = 85.0f;
    float right_x = 220.0f;
    float down_y = 120.0f;
    // int num_2=0;
    float center_x = 0.0;
    float center_y = 0.0;
    float size = 0;
    // int size =0;
    int distance_pixels = 0;

    unsigned char uartReadBuff[6] = {0};
    unsigned char uartsend[50] = {0};
    unsigned char threshold_value[6] = {0};
    int readLen = 0;
    unsigned int RecvLen = strlen("write");

    unsigned char distance_char[5];
    unsigned char left_up_char[11];
    unsigned char left_down_char[11];
    unsigned char right_up_char[11];
    unsigned char right_down_char[12];
    


    yolo_result *output_result = NULL;

    HI_S32 s32Ret = HI_SUCCESS;
    ISP_DEHAZE_ATTR_S stDehazeAttr;

   
    // printf("copiedFrame = %p\n", copiedFrame);


    ret = FrmToOrigImg((VIDEO_FRAME_INFO_S *)srcFrm, &img);
    SAMPLE_CHECK_EXPR_RET(ret != HI_SUCCESS, ret, "hand detect for YUV Frm to Img FAIL, ret=%#x\n", ret);
    ret = YOLOV5CalImg(self, &img, &output_result);
    if (output_result != NULL)
    {
        yolo_result_sort(output_result); // 输出结果排序，方便下面的 nms 处理

        yolo_nms(output_result, 0.001f); // nms
    }
    if (num_1==1)
        mat_init();
        std::vector<cv::Point> P_old;
   
    if ((g_fresh==1)&&(output_result != NULL)&&(cnt>40))
    {
        Dp_left_up_x = output_result->left_up_x;
        Dp_left_up_y = output_result->left_up_y ;
        Dp_right_down_x = output_result->right_down_x ;
        Dp_right_down_y = output_result->right_down_y ; 
        RECT_S rect_cut={0,0,1920,1080};
        //ret = mpp_frame_copy_yxx420sp(dstFrm,&CropFrame.stVideoFrameInfo,rect_cut);
        cnt=0;

        int result;
        printf("xl=%d,yl=%d,xr=%d,yr=%d\n",Dp_left_up_x,Dp_left_up_y,Dp_right_down_x,Dp_right_down_y);
                //mpp_frame_yxx_deinit(&CropFrame);
        result = point_detect(dstFrm, Dp_left_up_x*5, Dp_left_up_y*5, Dp_right_down_x*5, Dp_right_down_y*5);
        // if (result == 1) {
        // printf("Detected points\n");}
        //copiedFrame = copyVideoFrameInfo(srcFrm);
         
       
        if (result == 1) {
            std::cout << "Detected points:" << std::endl;
            for (const cv::Point& pt : g_detectedPoints.points) {
                std::cout << "(" << pt.x << ", " << pt.y << ")" << std::endl;
            }
        } 
        else {
            std::cout << "No points detected.\n";
        }
        
        if (g_detectedPoints.points.size() >= 4) {
        
        int x1 = Dp_left_up_x*5; // 这里假设 x1 和 y1 已经初始化，实际值应根据你的逻辑设置
        int y1 = Dp_left_up_y*5;
        cv::Point pt_in_frame = {x1, y1};
        // 将 cnt 中的点转换为相对于框架的坐标
        std::vector<cv::Point> pt_transformed;
        for (const auto& point : g_detectedPoints.points) {
            pt_transformed.push_back({point.x + pt_in_frame.x, point.y + pt_in_frame.y});
        }

        // 获取各点在框架中的坐标
        cv::Point A_in_frame = pt_transformed[0];
        cv::Point B_in_frame = (g_detectedPoints.points.size() > 3) ? pt_transformed[3] : cv::Point{};
        cv::Point C_in_frame = (g_detectedPoints.points.size() > 2) ? pt_transformed[2] : cv::Point{};
        cv::Point D_in_frame = pt_transformed[1];

        // 将点存储在向量中
        P_old = {A_in_frame, B_in_frame, C_in_frame, D_in_frame};
        // std::cout << "P_old points:" << std::endl;
        // for (const auto& point : P_old) {
        //     std::cout << "(" << point.x << ", " << point.y << ")" << std::endl;
        // }

        // 打印结果以验证
        std::cout << "A_in_frame: (" << A_in_frame.x << ", " << A_in_frame.y << ")\n";
        std::cout << "B_in_frame: (" << B_in_frame.x << ", " << B_in_frame.y << ")\n";
        std::cout << "C_in_frame: (" << C_in_frame.x << ", " << C_in_frame.y << ")\n";
        std::cout << "D_in_frame: (" << D_in_frame.x << ", " << D_in_frame.y << ")\n";
        }
        else {
        cv::Point A(1355, 422);
        cv::Point B(1387, 643);
        cv::Point C(1152, 671);
        cv::Point D(1126, 436);    
        // cv::Point A, B, C, D; // 声明 A、B、C 和 D
        for (const auto& point : g_detectedPoints.points) {
            std::vector<double> error;
            for (const auto& P_ : P_old) {
                error.push_back(distance(point, P_));
            }
            auto index = std::distance(error.begin(), std::min_element(error.begin(), error.end()));
            switch (index) {
                case 0:
                    A = point;
                    break;
                case 1:
                    B = point;
                    break;
                case 2:
                    C = point;
                    break;
                default:
                    D = point;
                    break;
                }
            }
            P_old = {A, B, C, D};
            // 打印更新后的点
            std::cout << "Updated P_old points:\n";
            std::cout << "A: (" << A.x << ", " << A.y << ")\n";
            std::cout << "B: (" << B.x << ", " << B.y << ")\n";
            std::cout << "C: (" << C.x << ", " << C.y << ")\n";
            std::cout << "D: (" << D.x << ", " << D.y << ")\n";

            
        }
        
        cv::Mat world_pts = (cv::Mat_<double>(4, 3) <<
        0, 0, 1,
        2.7, 0, 1,
        2.7, 2.7, 1,
        0, 2.7, 1);
        double fx = 1177.43167830732;
        double fy = 1119.27730595604;
        double cx = 1044.30793166953;
        double cy = 528.294868893910;  // 内参数

        // 将 std::vector<cv::Point2d> 转换为 cv::Mat
        cv::Mat img_pts(P_old.size(), 2, CV_64F);
        for (size_t i = 0; i < P_old.size(); ++i) {
            img_pts.at<double>(i, 0) = static_cast<double>(P_old[i].x);
            img_pts.at<double>(i, 1) = static_cast<double>(P_old[i].y);
        }

        std::cout << "img_pts matrix:" << std::endl;
        for (int i = 0; i < img_pts.rows; ++i) {
            for (int j = 0; j < img_pts.cols; ++j) {
                std::cout << img_pts.at<double>(i, j) << " ";
            }
            std::cout << std::endl;
        }

        // 将 img_pts 转换为齐次坐标形式
        cv::Mat img_pts_homog(img_pts.rows, img_pts.cols + 1, CV_64F);
        // 将 img_pts_homog 的最后一列设置为1.0（齐次坐标的w分量）  
        for (int i = 0; i < img_pts_homog.rows; ++i) {  
            img_pts_homog.at<double>(i, img_pts_homog.cols - 1) = 1.0;  
        } 
        img_pts.copyTo(img_pts_homog(cv::Rect(0, 0, img_pts.cols, img_pts.rows)));

        // 打印结果
        std::cout << "img_pts_homog: " << img_pts_homog << std::endl;

        // 调用 pose 函数
        auto [rotation_matrix, translation_matrix, intrinsic_inv, homography_matrix] = pose(img_pts_homog, world_pts, fx, fy, cx, cy);
        std::cout << "pose function called." << std::endl;
        

        //g_fresh=0;
        output_result = output_result->next;
    }
    cnt++;
    MppFrmDrawRects(dstFrm, output_result, 1, RGB888_RED, DRAW_RETC_THICK);
    // readLen = UartRead_1(uartFd, uartReadBuff, RecvLen, 0); /* 1000 :time out */
    printf("num_1----------%d\n", num_1);

    


    while (output_result != nullptr)//output_result != nullptr false
    {
        // int uartFd = 0;
        int left_up_x = output_result->left_up_x ;
        int left_up_y = output_result->left_up_y ;
        int right_down_x = output_result->right_down_x ;
        int right_down_y = output_result->right_down_y ;

        // 假设 VIDEO_FRAME_INFO_S 是视频帧的结构体，包含了帧的信息和像素数据
        //MPP_VI_FRAME_INFO_S CropFrame;

        // VIDEO_FRAME_INFO_S videoFrameInfo;
        
        // int CropW = right_down_x - left_up_x;
        // int CropH = right_down_y - left_up_y;
        // int CropX = (left_up_x + right_down_x)/2;
        // int CropY = (left_up_y + right_down_y)/2;
        HI_U32 CropW = static_cast<HI_U32>(right_down_x - left_up_x);
        HI_U32 CropH = static_cast<HI_U32>(right_down_y - left_up_y);
        HI_S32 CropX = static_cast<HI_S32>((left_up_x + right_down_x) / 2);
        HI_S32 CropY = static_cast<HI_S32>((left_up_y + right_down_y) / 2);

        // mpp_frame_yxx_init(&CropFrame,CropW,CropH,dstFrm->stVFrame.enPixelFormat);
        // RECT_S rect_cut={CropX,CropY,CropW,CropH};
      
        


        // int ret = mpp_frame_cutout_yxx420sp(dstFrm,&CropFrame.stVideoFrameInfo,rect_cut);
        char buffer[100];
        
        left_up_x = output_result->left_up_x*5 ;
        left_up_y = output_result->left_up_y*5 ;
        right_down_x = output_result->right_down_x*5 ;
        right_down_y = output_result->right_down_y*5 ;
        distance_pixels = (int)(output_result->right_down_x - output_result->left_up_x)*5;
       
        // 计算相机到物品的实际距离  
        int real_distance = calculateRealDistance(distance_pixels);  
        
        
        std::ostringstream oss;
        oss << std::setw(4) << std::setfill('0') << real_distance
        << "a" << std::setw(4) << std::setfill('0') << left_up_x << "," << std::setw(4) << std::setfill('0') << left_up_y
        << "b" << std::setw(4) << std::setfill('0') << left_up_x << "," << std::setw(4) << std::setfill('0') << right_down_y
        << "c" << std::setw(4) << std::setfill('0') << right_down_x << "," << std::setw(4) << std::setfill('0') << left_up_y
        << "d" << std::setw(4) << std::setfill('0') << right_down_x << "," << std::setw(4) << std::setfill('0') << right_down_y << "e";
        std::string uartsend = oss.str();
        // 打印结果以验证
        std::ostringstream oss_buffer;

        oss_buffer << "distance_pixels = " << std::setw(4) << std::setfill('0') << distance_pixels << "\n";
        std::cout << oss_buffer.str();
        oss_buffer.str("");
        oss_buffer.clear();

        oss_buffer << "real_distance = " << std::setw(4) << std::setfill('0') << real_distance << "\n";
        std::cout << oss_buffer.str();
        oss_buffer.str("");
        oss_buffer.clear();

        oss_buffer << "left_up (x, y) = (" << std::setw(4) << std::setfill('0') << left_up_x << "," << std::setw(4) << std::setfill('0') << left_up_y << ")\n";
        std::cout << oss_buffer.str();
        oss_buffer.str("");
        oss_buffer.clear();

        oss_buffer << "left_down (x, y) = (" << std::setw(4) << std::setfill('0') << left_up_x << "," << std::setw(4) << std::setfill('0') << right_down_y << ")\n";
        std::cout << oss_buffer.str();
        oss_buffer.str("");
        oss_buffer.clear();

        oss_buffer << "right_up (x, y) = (" << std::setw(4) << std::setfill('0') << right_down_x << "," << std::setw(4) << std::setfill('0') << left_up_y << ")\n";
        std::cout << oss_buffer.str();
        oss_buffer.str("");
        oss_buffer.clear();

        oss_buffer << "right_down (x, y) = (" << std::setw(4) << std::setfill('0') << right_down_x << "," << std::setw(4) << std::setfill('0') << right_down_y << ")\n";
        std::cout << oss_buffer.str();
        oss_buffer.str("");
        oss_buffer.clear();

        

        char* uartsend_cstr = const_cast<char*>(uartsend.c_str());
        // const char *uartsend_cstr = uartsend.c_str();
        std::cout << "uartsend: " << uartsend << std::endl;
        std::cout << "uartsend_cstr: " << uartsend_cstr << std::endl;
        
        

        if (num_1 % 10 == 0)
        {
            //发送数据到串口  
            
            ret = UartSend_1(uartFd, uartsend_cstr, 50);
            std::cout << "ret=" << ret << std::endl;
            // printf("ret=%d\n", ret);  
            if (ret < 0) 
                {  
                // 处理发送错误  
                std::sprintf(buffer, "Error sending UART data: %d\n", ret);  
                std::cout << buffer;
                }
            
        } 
        // freeVideoFrameInfo(copiedFrame);
        output_result = output_result->next;
    }
    
    
    return ret;
}






