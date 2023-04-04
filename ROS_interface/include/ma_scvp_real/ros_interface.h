#ifndef ROS_INTERFACE_H
#define ROS_INTERFACE_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>

#include <ros/ros.h>

#include <std_msgs/String.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseStamped.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl/PCLPointCloud2.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>
#include <moveit_msgs/ExecuteTrajectoryAction.h>

#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Path.h>

#include "ma_scvp_real/planner.hpp"

class RosInterface
{
    public:
        RosInterface(ros::NodeHandle& nh);

        virtual ~RosInterface();

        ros::NodeHandle nh_;

        // target frame
        std::string target_frame_;

        // tf listener
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
        std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

        // pose visualization
        geometry_msgs::PoseArray pose_array_;
        ros::Publisher pose_array_pub_;

        nav_msgs::Path path_;
        ros::Publisher path_pub_;

        nav_msgs::Path global_path_;
        ros::Publisher global_path_pub_;

        // pointcloud subscriber
        ros::Subscriber point_cloud_sub_;

        bool is_point_cloud_received_ = false;

        // moveit client
        std::string PLANNING_GROUP_ARM = "manipulator";
        std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm_;
        moveit::planning_interface::MoveGroupInterface::Plan arm_plan_;

        void run();

        // callbacks
        void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& point_cloud_msg);

        ros::Publisher pc_pub_;

        ros::Publisher octomap_pub_;

        void getPointCloud();

        void initMoveitClient();

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr preProcessPointCloud(const sensor_msgs::PointCloud2ConstPtr& point_cloud_msg);

        // moveit
        bool moveArmToHome();

        bool visitPose(const geometry_msgs::Pose& pose);

        bool visitJoints(const std::vector<double>& joints);

        bool visitWaypoints(const std::vector<geometry_msgs::Pose>& waypoints, float jump_threshold=5.0f);

        void generateWaypoints(const std::vector<std::vector<float>>& waypoints, std::vector<geometry_msgs::Pose>& waypoints_msg);

        geometry_msgs::PoseStamped getCurrentPose();

        std::vector<double> getJointValues();
};

#endif // ROS_INTERFACE_H