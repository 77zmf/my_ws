#ifndef MANUAL_CALIB_NODE_H
#define MANUAL_CALIB_NODE_H

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include "cvui.h"
#include "base_struct.h"
#include "calibrate.h"


class ManualCalibNode : public rclcpp::Node
{
public:
    ManualCalibNode();

private:
    // Configuration loading
    void loadConfigFromYaml(const std::string& config_file);
    void loadImage();
    
    // Callbacks
    void click_points_callback(const geometry_msgs::msg::PointStamped::SharedPtr msg);
    void timer_callback();
    
    // Point cloud operations
    void loadAndPublishPointCloud();
    void publishTargetPointCloud();
    void clearMarkers();
    
    // Configuration parameters
    std::string data_path_;
    std::string parameters_path_;
    std::string pointcloud_topic_;
    std::string pointcloud_target_topic_;
    std::string frame_id_;
    std::vector<int> camera_list_;
    std::vector<std::string> camera_name_list_;
    int calib_camera_id_;
    std::string calib_camera_name_;
    double outlier_range_;
    
    // Image
    cv::Mat image_;
    
    // Clicked points
    std::vector<cv::Point3f> clicked_points_;
    
    // Image clicked points (pixel coordinates)
    std::vector<cv::Point2f> clicked_pixels_;
    
    // Publishers and subscribers
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_target_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr click_points_sub_;
    
    // Marker operations
    void publishTextMarkers();
    
    // Static instance for mouse callback
    static ManualCalibNode* instance_;
    
    // Draw image points
    void drawImagePoints(cv::Mat& frame);

private:
    CalibFunc* calibInstance;
    double scale_height_ = 1.0;
    double scale_width_ = 1.0;
};

#endif // MANUAL_CALIB_NODE_H

