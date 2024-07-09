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

#ifndef DETECT2_H
#define DETECT2_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include "hi_comm_video.h"



// void yolo_result_sort(yolo_result *output_result);
// void yolo_nms(yolo_result *output_result, float iou_threshold);
// int calculateRealDistance(int distance_pixels);
// HI_S32 Yolo2HandDetectResnetClassifyCal(uintptr_t model, VIDEO_FRAME_INFO_S *srcFrm,VIDEO_FRAME_INFO_S *dstFrm,int num_1,char uartRead_1[],char uartRead_2[]);
HI_S32 Yolo2HandDetectResnetClassifyCal2(uintptr_t model, VIDEO_FRAME_INFO_S *srcFrm, VIDEO_FRAME_INFO_S *dstFrm, int num_1, char *uartRead_1, char *uartRead_2);
// void frame2VideoFrameInfo(const MPP_VI_FRAME_INFO_S *srcFrame, VIDEO_FRAME_INFO_S *dstFrame);
// HI_S32 point_detect( VIDEO_FRAME_INFO_S *dstFrm, int left_up_x, int left_up_y, int right_down_x, int right_down_y);

int UartSend_1(int fd, char *buf, int len);
#ifdef __cplusplus
}
#endif
#endif