/*
 * Copyright (c) 2022 HiSilicon (Shanghai) Technologies CO., LIMITED.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * 该cpp文件基于OpenCV实现了网球检测功能。为了保证FPS的帧数，
 * 我们设计的原则是IVE（Intelligent Video Engine）+AI CPU结合使用，即IVE不支持的算子
 * 通过AI CPU进行计算，否则走IVE硬件加速模块进行处理。并将检测的结果通过VGS标记出来。
 *
 * The cpp file implements the tennis ball detection function based on OpenCV.
 * In order to ensure the number of FPS frames,
 * The principle of our design is the combination of IVE (Intelligent Video Engine) + AI CPU,
 * that is, operators not supported by IVE are calculated by AI CPU, otherwise,
 * the IVE hardware acceleration module is used for processing.
 * And the detection results are marked by VGS.
 */

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "tennis_detect.h"
#include <opencv2/calib3d.hpp>

#include "sample_comm_nnie.h"
#include "sample_comm_ive.h"
#include "sample_media_ai.h"
#include "vgs_img.h"
#include "misc_util.h"
//#include "trackerbyframe.hpp"
using namespace std;  // 使用标准命名空间 
using namespace cv;   // 使用OpenCV命名空间  

static IVE_SRC_IMAGE_S pstSrc; // IVE源图像结构体，用于存储原始图像信息  
static IVE_DST_IMAGE_S pstDst; // IVE目标图像结构体，用于存储处理后的图像信息  
static IVE_CSC_CTRL_S stCscCtrl; // IVE颜色空间转换控制结构体，可能用于控制颜色空间转换的参数
IVE_DST_IMAGE_S pstDstn;





struct DetectedPoints {
    std::vector<cv::Point> points;
};
extern DetectedPoints g_detectedPoints;
// 配置IVE的源图像和目标图像参数  
static HI_VOID IveImageParamCfg(IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
    VIDEO_FRAME_INFO_S *srcFrame)
{
    pstSrc->enType = IVE_IMAGE_TYPE_YUV420SP;
    pstSrc->au64VirAddr[0] = srcFrame->stVFrame.u64VirAddr[0];
    pstSrc->au64VirAddr[1] = srcFrame->stVFrame.u64VirAddr[1];
    pstSrc->au64VirAddr[2] = srcFrame->stVFrame.u64VirAddr[2]; // 2: Image data virtual address

    pstSrc->au64PhyAddr[0] = srcFrame->stVFrame.u64PhyAddr[0];
    pstSrc->au64PhyAddr[1] = srcFrame->stVFrame.u64PhyAddr[1];
    pstSrc->au64PhyAddr[2] = srcFrame->stVFrame.u64PhyAddr[2]; // 2: Image data physical address

    pstSrc->au32Stride[0] = srcFrame->stVFrame.u32Stride[0];
    pstSrc->au32Stride[1] = srcFrame->stVFrame.u32Stride[1];
    pstSrc->au32Stride[2] = srcFrame->stVFrame.u32Stride[2]; // 2: Image data span

    pstSrc->u32Width = srcFrame->stVFrame.u32Width;
    pstSrc->u32Height = srcFrame->stVFrame.u32Height;

    pstDst->enType = IVE_IMAGE_TYPE_U8C3_PACKAGE;
    pstDst->u32Width = pstSrc->u32Width;
    pstDst->u32Height = pstSrc->u32Height;
    pstDst->au32Stride[0] = pstSrc->au32Stride[0];
    pstDst->au32Stride[1] = 0;
    pstDst->au32Stride[2] = 0; // 2: Image data span
}
// 将YUV格式的视频帧转换为RGB格式  
HI_S32 yuvFrame2DHSV(VIDEO_FRAME_INFO_S *srcFrame,IPC_IMAGE *dstImage)
{
    IVE_HANDLE hIveHandle;
    IVE_SRC_IMAGE_S pstSrc;
    IVE_DST_IMAGE_S pstDst;
    IVE_CSC_CTRL_S stCscCtrl;
    HI_S32 s32Ret = 0;
    stCscCtrl.enMode = IVE_CSC_MODE_PIC_BT709_YUV2RGB;//IVE_CSC_MODE_VIDEO_BT601_YUV2RGB;
    pstSrc.enType = IVE_IMAGE_TYPE_YUV420SP;
    pstSrc.au64VirAddr[0]=srcFrame->stVFrame.u64VirAddr[0];
    pstSrc.au64VirAddr[1]=srcFrame->stVFrame.u64VirAddr[1];
    pstSrc.au64VirAddr[2]=srcFrame->stVFrame.u64VirAddr[2];

    pstSrc.au64PhyAddr[0]=srcFrame->stVFrame.u64PhyAddr[0];
    pstSrc.au64PhyAddr[1]=srcFrame->stVFrame.u64PhyAddr[1];
    pstSrc.au64PhyAddr[2]=srcFrame->stVFrame.u64PhyAddr[2];

    pstSrc.au32Stride[0]=srcFrame->stVFrame.u32Stride[0];
    pstSrc.au32Stride[1]=srcFrame->stVFrame.u32Stride[1];
    pstSrc.au32Stride[2]=srcFrame->stVFrame.u32Stride[2];

    pstSrc.u32Width = srcFrame->stVFrame.u32Width;
    pstSrc.u32Height = srcFrame->stVFrame.u32Height;



   /* HI_BOOL bInstant = HI_TRUE;
    s32Ret = HI_MPI_IVE_CSC(&hIveHandle,&pstSrc,&pstDst,&stCscCtrl,bInstant);
    if(HI_SUCCESS != s32Ret)
    {
        HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
        std::cout<<"HI_MPI_IVE_CSC Failed with 0x"<<s32Ret<<std::endl;
        return s32Ret;
    }
    if (HI_TRUE == bInstant)
    {
        HI_BOOL bFinish = HI_TRUE;
        HI_BOOL bBlock = HI_TRUE;
        s32Ret = HI_MPI_IVE_Query(hIveHandle,&bFinish,bBlock);
        while(HI_ERR_IVE_QUERY_TIMEOUT == s32Ret)
        {
            usleep(230);
            s32Ret = HI_MPI_IVE_Query(hIveHandle,&bFinish,bBlock);
        }
    }*/
    dstImage->u64PhyAddr = pstDst.au64PhyAddr[0];
    dstImage->u64VirAddr = pstDst.au64VirAddr[0];
    dstImage->u32Width = pstDst.u32Width;
    dstImage->u32Height = pstDst.u32Height;
    return s32Ret;
}
static HI_S32 yuvFrame2rgb(VIDEO_FRAME_INFO_S *srcFrame, IPC_IMAGE *dstImage)
{
    IVE_HANDLE hIveHandle; // IVE句柄  
    HI_S32 s32Ret = 0;// 返回值，初始化为成功  
    // stCscCtrl.enMode = IVE_CSC_MODE_PIC_BT709_YUV2RGB; // IVE_CSC_MODE_VIDEO_BT601_YUV2RGB// 设置颜色空间转换模式为YUV420SP（BT.709）到RGB  
    stCscCtrl.enMode = IVE_CSC_MODE_PIC_BT601_YUV2HSV;
    // 假设它们已经被定义并指向IVE_SRC_IMAGE_S和IVE_DST_IMAGE_S类型的结构体  
    IveImageParamCfg(&pstSrc, &pstDst, srcFrame);// 配置IVE的源图像和目标图像参数  
    // 为目标图像分配内存（缓存的）  
    s32Ret = HI_MPI_SYS_MmzAlloc(&pstDst.au64PhyAddr[0], (void **)&pstDst.au64VirAddr[0],
        "User", HI_NULL, pstDst.u32Height*pstDst.au32Stride[0] * 3); // 3: multiple
    if (HI_SUCCESS != s32Ret) {
        // 如果分配失败，释放内存并返回错误  
        HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
        SAMPLE_PRT("HI_MPI_SYS_MmzFree err\n");
        return s32Ret;
    }
    // 刷新内存缓存  
    // s32Ret = HI_MPI_SYS_MmzFlushCache(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0],
    //     pstDst.u32Height*pstDst.au32Stride[0] * 3); // 3: multiple// 3: RGB三通道的倍数  
    // if (HI_SUCCESS != s32Ret) {
    //      // 如果刷新失败，释放内存并返回错误  
    //     HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
    //     return s32Ret;
    // }
    // 3: multiple
    // 将目标图像的内存区域清零  
    memset_s((void *)pstDst.au64VirAddr[0], pstDst.u32Height*pstDst.au32Stride[0] * 3,
        0, pstDst.u32Height*pstDst.au32Stride[0] * 3); // 3: multiple// 3: RGB三通道的倍数  
    HI_BOOL bInstant = HI_TRUE;  // 设置是否立即处理
    /* 执行颜色空间转换  
    s32Ret = HI_MPI_IVE_CSC(&hIveHandle, &pstSrc, &pstDst, &stCscCtrl, bInstant);
    if (HI_SUCCESS != s32Ret) {
        // 如果转换失败，释放内存并返回错误  
        HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
        return s32Ret;
    }
    // 如果设置为立即处理，则等待处理完成  
    if (HI_TRUE == bInstant) {
        HI_BOOL bFinish = HI_TRUE;
        HI_BOOL bBlock = HI_TRUE;
        s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, bBlock);
        while (HI_ERR_IVE_QUERY_TIMEOUT == s32Ret) {
            usleep(100); // 100: usleep time// 等待100微秒  
            s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, bBlock);
        }
    }*/
    // 将结果保存到IPC_IMAGE结构体中  
    dstImage->u64PhyAddr = pstDst.au64PhyAddr[0];// 将目标图像的物理地址保存到IPC_IMAGE结构体的物理地址字段中
    dstImage->u64VirAddr = pstDst.au64VirAddr[0];// 将目标图像的虚拟地址保存到IPC_IMAGE结构体的虚拟地址字段中
    dstImage->u32Width = pstDst.u32Width;// 将目标图像的宽度保存到IPC_IMAGE结构体的宽度字段中  
    dstImage->u32Height = pstDst.u32Height;// 将目标图像的高度保存到IPC_IMAGE结构体的高度字段中  
  
    return HI_SUCCESS;// 返回成功状态码
}
// 定义一个函数frame2Mat，用于将YUV帧转换为OpenCV的Mat对象
HI_S32 frame2Mat(VIDEO_FRAME_INFO_S *srcFrame, Mat &dstMat)
{
    // 从srcFrame结构体中获取视频的宽度和高度  
    HI_U32 w = srcFrame->stVFrame.u32Width;
    HI_U32 h = srcFrame->stVFrame.u32Height;
    int bufLen = w * h * 3;// 计算RGB缓冲区的大小（每个像素3个字节，即红绿蓝三个通道）  
    HI_U8 *srcRGB = NULL;// 定义一个指向RGB数据的指针，初始化为NULL  
    IPC_IMAGE dstImage;// 定义一个IPC_IMAGE结构体，用于保存YUV到RGB转换后的图像信息  
    // 调用yuvFrame2rgb函数进行YUV到RGB的转换，如果转换失败则返回错误  
    
    IVE_HANDLE hIveHandle;
    IVE_SRC_IMAGE_S pstSrcn;
    //IVE_DST_IMAGE_S pstDst;
    IVE_CSC_CTRL_S stCscCtrl;
    HI_S32 s32Ret = 0;
    stCscCtrl.enMode = IVE_CSC_MODE_PIC_BT709_YUV2HSV;//IVE_CSC_MODE_VIDEO_BT601_YUV2RGB;
    pstSrcn.enType = IVE_IMAGE_TYPE_YUV420SP;
    pstSrcn.au64VirAddr[0]=srcFrame->stVFrame.u64VirAddr[0];
    pstSrcn.au64VirAddr[1]=srcFrame->stVFrame.u64VirAddr[1];
    pstSrcn.au64VirAddr[2]=srcFrame->stVFrame.u64VirAddr[2];

    pstSrcn.au64PhyAddr[0]=srcFrame->stVFrame.u64PhyAddr[0];
    pstSrcn.au64PhyAddr[1]=srcFrame->stVFrame.u64PhyAddr[1];
    pstSrcn.au64PhyAddr[2]=srcFrame->stVFrame.u64PhyAddr[2];

    pstSrcn.au32Stride[0]=srcFrame->stVFrame.u32Stride[0];
    pstSrcn.au32Stride[1]=srcFrame->stVFrame.u32Stride[1];
    pstSrcn.au32Stride[2]=srcFrame->stVFrame.u32Stride[2];

    pstSrcn.u32Width = srcFrame->stVFrame.u32Width;
    pstSrcn.u32Height = srcFrame->stVFrame.u32Height;
    //pstDstn.u32Width=srcFrame->stVFrame.u32Width;
    //pstDstn.u32Height=srcFrame->stVFrame.u32Height;
    //pstDstn.au32Stride[0]=srcFrame->stVFrame.u32Stride[0];
    memset((void *)pstDstn.au64VirAddr[0], 0, pstDstn.u32Height*pstDstn.au32Stride[0]*3);
    HI_BOOL bInstant = HI_TRUE;
    HI_BOOL bFinish = HI_TRUE;
    s32Ret = HI_MPI_IVE_CSC(&hIveHandle,&pstSrcn,&pstDstn,&stCscCtrl,bInstant);
    //HI_MPI_IVE_Query_Timeout(hIveHandle, -1,&bFinish);
    if(HI_SUCCESS != s32Ret)
    {
        //HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
        std::cout<<"HI_MPI_IVE_CSC Failed with 0x"<<s32Ret<<std::endl;
        return s32Ret;
    }
    if (HI_TRUE == bInstant)
    {
        HI_BOOL bFinish = HI_TRUE;
        HI_BOOL bBlock = HI_TRUE;
        s32Ret = HI_MPI_IVE_Query(hIveHandle,&bFinish,bBlock);
        while(HI_ERR_IVE_QUERY_TIMEOUT == s32Ret)
        {
            usleep(100);
            s32Ret = HI_MPI_IVE_Query(hIveHandle,&bFinish,bBlock);
        }
    }
    // 从dstImage中获取RGB数据的虚拟地址  
    srcRGB = (HI_U8 *)pstDstn.au64VirAddr[0];
    // 创建一个OpenCV的Mat对象，大小为h x w，类型为CV_8UC3（即8位无符号整型，三通道）  
    //dstMat.create(h, w, CV_8UC3);
    // 使用memcpy_s安全地将RGB数据从源地址复制到Mat对象的内部缓冲区  
    // 注意：这里假设了srcRGB指向的数据和dstMat.data指向的缓冲区大小都是bufLen  
    memcpy_s(dstMat.data, bufLen * sizeof(HI_U8), srcRGB, bufLen * sizeof(HI_U8));
    imwrite("test.jpg", dstMat);
    // 释放dstImage中使用的物理内存（假设u64PhyAddr是物理地址，u64VirAddr是对应的虚拟地址） 
    //HI_MPI_SYS_MmzFree(dstImage.u64PhyAddr, (void *)&(dstImage.u64VirAddr));
     //s32Ret =HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
     if(HI_SUCCESS != s32Ret)
        {
            std::cout<<"HI_MPI_SYS_MmzFree Failed with 0x"<<s32Ret<<std::endl;
            return s32Ret;
        }
    return HI_SUCCESS;    // 返回成功状态码 
}
// tennis_detect类中的一个成员函数，用于加载网球检测模型
HI_S32 tennis_detect::TennisDetectLoad(uintptr_t* model)
{
    // 定义一个返回值ret，并初始化为1（通常，在C/C++中，0表示成功，非0表示错误，但这里的约定可能不同） 
    HI_S32 ret = 1;
    // 将传入的model指针所指向的值设置为1（这看起来并不是一个有效的模型句柄或地址，可能只是一个占位符）  
    *model = 1;
    // 打印一条消息到控制台，表明"TennisDetectLoad"函数已经成功执行（尽管从函数实现来看，它并没有真正加载任何模型）  
    SAMPLE_PRT("TennisDetectLoad success\n");
    // 返回ret的值，即1（可能表示某种错误或占位符状态，而不是真正的成功）  
    return ret;
}

HI_S32 tennis_detect::TennisDetectUnload(uintptr_t model)
{
    // 这里只是将传入的model参数的值设置为0，但这并不会影响原始传入的值  
    // 因为model是按值传递的，不是引用或指针的引用 
    model = 0;

    return HI_SUCCESS;// 返回HI_SUCCESS，表示函数执行成功  
}

/*
 * 网球检测推理
 * Tennis detect calculation
 */
HI_S32 tennis_detect::TennisDetectCal(uintptr_t model, VIDEO_FRAME_INFO_S *srcFrm, VIDEO_FRAME_INFO_S *dstFrm)
{
    (void)model;
    int ret = 0;
    RectBox boxs[32] = {0}; // 32: TENNIS_OBJ_MAX
    int j = 0;

    Mat image;
    //frame2Mat(srcFrm, image);
    if (image.size == 0) {
        SAMPLE_PRT("image is null\n");
        return HI_FAILURE;
    }

    Mat src = image;
    Mat src1 = src.clone();
    Mat dst, edge, gray, hsv;

    dst.create(src1.size(), src1.type()); // Create a matrix of the same type and size as src (dst)

    /*
     * cvtColor运算符用于将图像从一个颜色空间转换到另一个颜色空间
     * The cvtColor operator is used to convert an image from one color space to another color space
     */
    cvtColor(src1, hsv, COLOR_BGR2HSV); // Convert original image to HSV image

    /*
     * 二值化hsv图像，这里是对绿色背景进行二值化，
     * 这个参数可以根据需要调整
     *
     * Binarize the hsv image, here is to binarize the green background,
     * this parameter can be adjusted according to requirements
     */
    inRange(hsv, Scalar(31, 82, 68), Scalar(65, 248, 255), gray); // 31: B, 82: G, 68:R / 65: B, 248:G, 255:R

    /*
     * 使用canny算子进行边缘检测
     * Use canny operator for edge detection
     */
    Canny(gray, gray, 3, 9, 3); // 3: threshold1, 9: threshold2, 3: apertureSize
    vector<vector<Point>> contours;
    findContours(gray, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
    SAMPLE_PRT("contours.size():%d\n", contours.size());

    for (int i = 0; i < (int)contours.size(); i++) {
        if (contours.size() > 40) { // 40: contours.size() extremes
            continue;
        }

        Rect ret1 = boundingRect(Mat(contours[i]));
        ret1.x -= 5; // 5: x coordinate translation
        ret1.y -= 5; // 5: y coordinate translation
        ret1.width += 10; // 10: Rectangle width plus 10
        ret1.height += 10; // 10: Rectangle height plus 10

        // 20: Rectangle width and height pixel extremes
        if ((ret1.width > 20) && (ret1.height > 20)) {
            boxs[j].xmin = ret1.x * 3; // 3: optimized value
            boxs[j].ymin = (int)(ret1.y * 2.25); // 2.25: optimized value
            boxs[j].xmax = boxs[j].xmin + ret1.width * 3; // 3: optimized value
            boxs[j].ymax = boxs[j].ymin + (int)ret1.height * 2.25; // 2.25: optimized value
            j++;
        }
    }
    // 25: detect boxesNum
    if (j > 0 && j <= 25) {
        SAMPLE_PRT("box num:%d\n", j);
        //MppFrmDrawRects(dstFrm, boxs, j, RGB888_RED, 2); // 2: DRAW_RETC_THICK
    }

    return ret;
}
Mat imageN;
// ColorSpace_T CHSV;
void mat_init()
{
    imageN.create(1080, 1920, CV_8UC3);
    pstDstn.enType = IVE_IMAGE_TYPE_U8C3_PACKAGE;
    pstDstn.u32Width   = 1920;
    pstDstn.u32Height  = 1080;
    pstDstn.au32Stride[0]  = 1920;
    pstDstn.au32Stride[1]  = 0;
    pstDstn.au32Stride[2]  = 0;
    HI_S32 s32Ret = 0;
    s32Ret = HI_MPI_SYS_MmzAlloc(&pstDstn.au64PhyAddr[0], (void **)&pstDstn.au64VirAddr[0], "User", HI_NULL, pstDstn.u32Height*pstDstn.au32Stride[0]*3);
    if(HI_SUCCESS != s32Ret)
    {
        HI_MPI_SYS_MmzFree(pstDstn.au64PhyAddr[0], (void *)pstDstn.au64VirAddr[0]);
        std::cout<<"HI_MPI_SYS_MmzAlloc Failed with 0x"<<s32Ret<<std::endl;
    }
}

// extern "C"  {
HI_S32 point_detect(VIDEO_FRAME_INFO_S *copiedFrame, HI_U16 left_up_x, HI_U16 left_up_y, HI_U16 right_down_x, HI_U16 right_down_y)
{
    // Mat image;
    frame2Mat(copiedFrame, imageN);
    // frame2Mat(&CropFrame.stVideoFrameInfo,img,ColorMode);
// 绘制左上角点 (红色)
    cv::circle(imageN, cv::Point(left_up_x, left_up_y), 5, cv::Scalar(0, 0, 255), -1);

    // 绘制左下角点 (绿色)
    cv::circle(imageN, cv::Point(left_up_x, right_down_y), 5, cv::Scalar(0, 255, 0), -1);

    // 绘制右上角点 (蓝色)
    cv::circle(imageN, cv::Point(right_down_x, left_up_y), 5, cv::Scalar(255, 0, 0), -1);

    // 绘制右下角点 (黄色)
    cv::circle(imageN, cv::Point(right_down_x, right_down_y), 5, cv::Scalar(0, 255, 255), -1);
    imwrite("imageN.jpg",imageN);
    if (imageN.empty()) {
        SAMPLE_PRT("image is null\n");
        return HI_FAILURE;
    }
    int width = right_down_x - left_up_x;
    int height = right_down_y - left_up_y;
    SAMPLE_PRT("Final values: left_up_x: %d, left_up_y: %d, right_down_x: %d, right_down_y: %d, width: %d, height: %d\n",
           left_up_x, left_up_y, right_down_x, right_down_y, width, height);

    if (left_up_x < 0 || left_up_y < 0 || right_down_x > imageN.cols || right_down_y > imageN.rows || width <= 0 || height <= 0) {
        SAMPLE_PRT("Invalid crop region\n");
        return HI_FAILURE;
    }
    cv::Rect cropRegion(left_up_x, left_up_y, width, height);
    cv::Mat croppedImage = imageN(cropRegion);

    if (croppedImage.empty()) {
        printf("croppedImage is null\n");
        return HI_FAILURE;
    }
    imwrite("crop.jpg",croppedImage);
    cv::Scalar lower_green = cv::Scalar(20, 20, 40);  
    cv::Scalar upper_green = cv::Scalar(140, 255, 120); 
    // 创建一个与roi_hsv相同大小的Mat对象来存储掩码  
    cv::Mat mask1;  
    mask1.create(croppedImage.size(), CV_8UC1); // 8位无符号单通道图像  
    cv::inRange(croppedImage, lower_green, upper_green, mask1);
    imwrite("mask.jpg",mask1);

    // 创建结构元素
    int morph_size = 2; // 可以根据实际情况调整
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                                                cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
                                                cv::Point(morph_size, morph_size));

    // 先进行闭运算
    cv::Mat mask1_closed;
    cv::morphologyEx(mask1, mask1_closed, cv::MORPH_CLOSE, element);

    // 再进行开运算
    cv::Mat mask1_opened;
    cv::morphologyEx(mask1_closed, mask1_opened, cv::MORPH_OPEN, element);
    imwrite("mask1_opened.jpg",mask1_opened);


    vector<vector<Point>> contours;
    vector<cv::Vec4i> hierarchy; 
    findContours(mask1_opened, contours, cv::RETR_LIST, CHAIN_APPROX_SIMPLE, cv::Point());
    SAMPLE_PRT("contours.size():%d\n", contours.size());
    if (contours.size()==0)
    {
        return HI_FAILURE;
    }

    // 找到面积最大的轮廓
    auto maxContourIt = std::max_element(contours.begin(), contours.end(), [](const vector<cv::Point>& cnt1, const vector<cv::Point>& cnt2) {
        return cv::contourArea(cnt1, true) > cv::contourArea(cnt2, true);
    });
    // 获取最大的轮廓
    vector<cv::Point> maxContour = *maxContourIt;
    // 对最大轮廓进行多边形逼近
    vector<cv::Point> approx;
    cv::approxPolyDP(maxContour, approx, cv::arcLength(maxContour, true) * 0.02, true);
   
    // 检查多边形是否为四边形并确保其角点不相同
    if (approx.size() == 4) {
        bool allPointsDifferent = true;
        for (size_t i = 0; i < approx.size(); ++i) {
            for (size_t j = i + 1; j < approx.size(); ++j) {
                if (approx[i] == approx[j]) {
                    allPointsDifferent = false;
                    break;
                }
            }
        }

        if (allPointsDifferent) {
            std::cout << "Approximated quadrilateral points:" << std::endl;
            for (const cv::Point& pt : approx) {
                std::cout << "(" << pt.x << ", " << pt.y << ")" << std::endl;
            }
            // 保存逼近后的点信息
            g_detectedPoints.points = approx;
        } else {
            std::cout << "Quadrilateral points are not unique." << std::endl;
        }
    } else {
        std::cout << "No valid quadrilateral found." << std::endl;
    }



    

    return 1;
   
}

// } // extern "C"