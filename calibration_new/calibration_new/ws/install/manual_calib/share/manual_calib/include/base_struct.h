#ifndef BASE_STRUCT_H
#define BASE_STRUCT_H

#include <opencv2/opencv.hpp>

// 内参结构体
struct IntrParams {
    cv::Mat K_;      // 相机内参矩阵
    cv::Mat dist_;   // 畸变系数
};

// 外参结构体
struct ExtParams {
    double x, y, z;        // 平移
    double roll, pitch, yaw;  // 欧拉角
};

// 2D角点结构体
struct Corners2D {
    cv::Point2f corn_left_top;
    cv::Point2f corn_right_top;
    cv::Point2f corn_right_bottom;
    cv::Point2f corn_left_bottom;
};

// 3D点结构体
struct Point3D {
    float x, y, z;
};

// 3D角点云结构体
struct CornersCloud {
    Point3D corn_left_top_3d;
    Point3D corn_right_top_3d;
    Point3D corn_right_bottom_3d;
    Point3D corn_left_bottom_3d;
};

// 标定板信息结构体
struct BoardInfo {
    Corners2D corners_2d;
    CornersCloud corners_cloud;
};

#endif // BASE_STRUCT_H

