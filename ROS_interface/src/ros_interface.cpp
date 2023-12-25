#include <ma_scvp_real/ros_interface.h>

#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <eigen_conversions/eigen_msg.h>

RosInterface::RosInterface(ros::NodeHandle& nh):
    nh_(nh),
    target_frame_("base_link")
{
    // initialize tf listener
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>();
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    octomap_pub_ = nh.advertise<octomap_msgs::Octomap>("octomap", 1, true);

    // initialize pointcloud publisher
    pc_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("pointcloud", 1, true);

    pose_array_pub_ = nh_.advertise<geometry_msgs::PoseArray>("pose_array", 1, true);

    path_pub_ = nh_.advertise<nav_msgs::Path>("path", 1, true);
    global_path_pub_ = nh_.advertise<nav_msgs::Path>("global_path", 1, true);
}

RosInterface::~RosInterface()
{
    
}

void RosInterface::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& point_cloud_msg)
{
    ROS_INFO("Pointcloud received");
    // unsubscribe to stop receiving messages
    point_cloud_sub_.shutdown();

    // pre-process point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc = preProcessPointCloud(point_cloud_msg);
    
    *share_data->cloud_now = *pc;

    // set flag
    is_point_cloud_received_ = true;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr RosInterface::preProcessPointCloud(const sensor_msgs::PointCloud2ConstPtr &point_cloud_msg)
{
    // transform pointcloud msg from camera link to target_frame
    sensor_msgs::PointCloud2 transformed_msg;
    transformed_msg.header.frame_id = target_frame_;

    if (tf_buffer_)
    {
        geometry_msgs::TransformStamped transform_stamped;
        try
        {
            transform_stamped = tf_buffer_->lookupTransform(target_frame_, point_cloud_msg->header.frame_id, ros::Time(0));

            tf2::doTransform(*point_cloud_msg, transformed_msg, transform_stamped);
        }
        catch (tf2::TransformException &ex)
        {
            ROS_ERROR("PCL transform failed: %s", ex.what());
            ROS_ERROR("Unable to transform pointcloud from %s to %s", point_cloud_msg->header.frame_id.c_str(), target_frame_.c_str());
            ROS_ERROR("Make sure that the transform between these two frames is published");
            
            return pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
        }
    } else {
        ROS_ERROR_THROTTLE(2.0, "tf_buffer_ is not initialized");
        return pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    }

    // Convert pointcloud to pcl::PointCloud
    pcl::PCLPointCloud2Ptr pc2(new pcl::PCLPointCloud2);
    pcl_conversions::toPCL(transformed_msg, *pc2);
    pc2->header.frame_id = transformed_msg.header.frame_id;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2(*pc2, *pc);

    ROS_INFO("Pointcloud received and transformed to %s", target_frame_.c_str());

    // Publish to rviz
    // sensor_msgs::PointCloud2 pc2_msg;
    // pcl::toROSMsg(*pc, pc2_msg);
    // pc2_msg.header.frame_id = target_frame_;
    // pc2_msg.header.stamp = ros::Time::now();
    // pc_pub_.publish(pc2_msg);

    // ROS_INFO("Pointcloud published to rviz");
    // ros::Duration(2.0).sleep();
    
    return pc;
}

void RosInterface::run()
{
    initMoveitClient();

    // wait for 2 seconds
    ros::Duration(1.0).sleep();

    // create 2 waypoints from the current pose
    geometry_msgs::PoseStamped current_pose = getCurrentPose();

    // initialize planner
    share_data = new Share_Data("/home/user/pan/ma-scvp-real/src/NBV_Simulation_MA-SCVP/DefaultConfiguration.yaml");
    string status="";

    // initialize planner to do ground truth planning
    if(share_data->gt_mode){
        //注意一般来说gt_mode需要运行二/三次，第一次是为了初始化获取场景点云，第二次才是获取视点坐标，三次的话会更准确
        cout<<"Ground truth mode"<<endl;

        bool is_first_time = true;

        //move to a choosed start point
        //start matrix 0.026033, 0.543750, 0.262568, -0.605269, 0.547053, 0.563584, 0.129445

        double start_x = 0.026033;
        double start_y = 0.543750;
        double start_z = 0.262568;
        double start_qx = -0.605269;
        double start_qy = 0.547053;
        double start_qz = 0.563584;
        double start_qw = 0.129445;

        vector<vector<float>> now_waypoints1;
		vector<float> temp_waypoint1 = {float(start_x), float(start_y), float(start_z), float(start_qx), float(start_qy), float(start_qz), float(start_qw)};
        now_waypoints1.push_back(temp_waypoint1);
        std::vector<geometry_msgs::Pose> waypoints_msg1;
        generateWaypoints(now_waypoints1, waypoints_msg1);
        if (visitWaypoints(waypoints_msg1)){
            ROS_INFO("MoveitClient: Arm moved to waypoints");
        }
        else{
            ROS_ERROR("MoveitClient: Failed to move arm to waypoints");
        }
        geometry_msgs::PoseStamped current_pose1 = getCurrentPose();

        //show start point
        Eigen::Quaterniond start_q(start_qw, start_qx, start_qy, start_qz);
        Eigen::Matrix3d start_rotation = start_q.toRotationMatrix();
        Eigen::Matrix4d start_pose_world = Eigen::Matrix4d::Identity();
        start_pose_world.block<3,3>(0, 0) = start_rotation;
        start_pose_world(0, 3) = start_x;
        start_pose_world(1, 3) = start_y;
        start_pose_world(2, 3) = start_z;

        share_data->now_camera_pose_world = start_pose_world * share_data->camera_depth_to_rgb.inverse();

        auto viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("3D Viewer"));
        viewer->setBackgroundColor(255, 255, 255);
        viewer->addCoordinateSystem(0.1);
        viewer->initCameraParameters();

        Eigen::Vector4d X(0.05, 0, 0, 1);
        Eigen::Vector4d Y(0, 0.05, 0, 1);
        Eigen::Vector4d Z(0, 0, 0.05, 1);
        Eigen::Vector4d O(0, 0, 0, 1);
        X = share_data->now_camera_pose_world * X;
        Y = share_data->now_camera_pose_world * Y;
        Z = share_data->now_camera_pose_world * Z;
        O = share_data->now_camera_pose_world * O;
        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(-1));
        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(-1));
        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(-1));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(-1));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(-1));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(-1));

        //get pointcloud and init ground truth
        getPointCloud();
        while(is_point_cloud_received_ == false){
            ros::Duration(0.2).sleep();
        }
        is_point_cloud_received_ = false;

        //尝试读取ground truth，如果存在就使用，如果不存在就计算
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(share_data->pcd_file_path + share_data->name_of_pcd + "_scene.pcd", *share_data->cloud_now) == -1) {
            cout << "can not find ground truth scene" << endl;
        }
        else{
            cout << "find ground truth scene" << endl;
            is_first_time = false;
        }
        *share_data->cloud_ground_truth = *share_data->cloud_now;
        std::cout << "point cloud ground truth size: " << share_data->cloud_ground_truth->size() << std::endl;

        nbv_plan = new NBV_Planner(share_data);

        viewer->setCameraPosition(0.8, 0.8, 0.8, share_data->object_center_world(0), share_data->object_center_world(1), share_data->object_center_world(2), 0, 0, 1);

        int num_of_movable_views = 0;
        vector<vector<double>> view_space_pose;
        vector<vector<double>> view_space_xyzq;

        // move to first view
        Eigen::Vector3d now_view_xyz(Eigen::Vector3d(start_pose_world(0,3), start_pose_world(1,3), start_pose_world(2,3)));
        Eigen::Vector3d next_view_xyz(nbv_plan->now_view_space->views[0].init_pos(0), nbv_plan->now_view_space->views[0].init_pos(1), nbv_plan->now_view_space->views[0].init_pos(2));
        vector<Eigen::Vector3d> points;
        int num_of_path = get_trajectory_xyz(points, now_view_xyz, next_view_xyz, nbv_plan->now_view_space->object_center_world, nbv_plan->now_view_space->predicted_size, share_data->move_dis_pre_point, share_data->safe_distance);
        if (num_of_path == -1) {
            cout << "no path. throw" << endl;
            share_data->over = true;
            return;
        }
        if (num_of_path == -2) cout << "Line" << endl;
        if (num_of_path > 0)  cout << "Obstcale" << endl;
        cout << "num_of_path:" << points.size() << endl;

        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(now_view_xyz(0), now_view_xyz(1), now_view_xyz(2)), pcl::PointXYZ(points[0](0), points[0](1), points[0](2)), 0, 128, 128, "trajectory" + to_string(-1));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "trajectory"  + to_string(-1));
        for (int k = 0; k < points.size() - 1; k++) {
            viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(points[k](0), points[k](1), points[k](2)), pcl::PointXYZ(points[k + 1](0), points[k + 1](1), points[k + 1](2)), 0, 128, 128, "trajectory" + to_string(k));
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "trajectory" + to_string(k));
        }

        // get waypoints
        vector<vector<float>> now_waypoints;
        Eigen::Matrix4d now_camera_pose_world = share_data->now_camera_pose_world;
        for(int j=0;j<points.size();j++){
            View temp_view(points[j]);
            temp_view.get_next_camera_pos(now_camera_pose_world,  nbv_plan->now_view_space->object_center_world);
            Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 

            cout<<"temp_camera_pose_world:"<<temp_camera_pose_world<<endl;
            if(j==points.size()-1){
                Eigen::Vector4d X(0.05, 0, 0, 1);
                Eigen::Vector4d Y(0, 0.05, 0, 1);
                Eigen::Vector4d Z(0, 0, 0.05, 1);
                Eigen::Vector4d O(0, 0, 0, 1);
                X = now_camera_pose_world * temp_view.pose.inverse() * X;
                Y = now_camera_pose_world * temp_view.pose.inverse() * Y;
                Z = now_camera_pose_world * temp_view.pose.inverse() * Z;
                O = now_camera_pose_world * temp_view.pose.inverse() * O;
                viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(j));
                viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(j));
                viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(j));
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(j));
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(j));
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(j));
            }

            Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
            Eigen::Quaterniond temp_q(temp_rotation);
            vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };
            now_waypoints.push_back(temp_waypoint);

            now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
        }

        //viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_ground_truth, "cloud_ground_truth");
        //viewer->spinOnce(100);

        std::vector<geometry_msgs::Pose> waypoints_msg;
        generateWaypoints(now_waypoints, waypoints_msg);
        if (visitWaypoints(waypoints_msg)){
            num_of_movable_views++;
            vector<double> current_joint = getJointValues();
            view_space_pose.push_back(current_joint);
            ROS_INFO("MoveitClient: Arm moved to waypoints");
        }
        else{
            ROS_ERROR("MoveitClient: Failed to move arm to waypoints");
        }
        geometry_msgs::PoseStamped current_pose = getCurrentPose();

        view_space_xyzq.push_back({current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z, current_pose.pose.orientation.x, current_pose.pose.orientation.y, current_pose.pose.orientation.z, current_pose.pose.orientation.w});
        share_data->now_camera_pose_world = now_camera_pose_world;

        ros::Duration(2.0).sleep();
        getPointCloud();
        while(is_point_cloud_received_ == false){
            ros::Duration(0.2).sleep();
        }
        is_point_cloud_received_ = false;
        nbv_plan->percept->precept(share_data->cloud_now);

        viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_scene, "cloud_scene");
        viewer->spinOnce(100);

        //return;

        //预先计算好的全局路径，32个视点
        vector<int> init_view_ids =  {0, 5, 27, 31, 30, 1, 7, 10, 18, 14, 17, 16, 6, 25, 2, 22, 20, 15, 13, 11, 8, 24, 3, 23, 29, 4, 12, 21, 19, 26, 9, 28};

        // move to next view space
        for (int i = 0; i < init_view_ids.size() - 1; i++) {
            cout<< "view_space_pose.size():" << view_space_pose.size()<<endl;
            // otherwise move robot
            Eigen::Vector3d now_view_xyz(nbv_plan->now_view_space->views[init_view_ids[i]].init_pos(0), nbv_plan->now_view_space->views[init_view_ids[i]].init_pos(1), nbv_plan->now_view_space->views[init_view_ids[i]].init_pos(2));
            Eigen::Vector3d next_view_xyz(nbv_plan->now_view_space->views[init_view_ids[i+1]].init_pos(0), nbv_plan->now_view_space->views[init_view_ids[i+1]].init_pos(1), nbv_plan->now_view_space->views[init_view_ids[i+1]].init_pos(2));
            vector<Eigen::Vector3d> points;
            int num_of_path = get_trajectory_xyz(points, now_view_xyz, next_view_xyz, share_data->object_center_world, share_data->predicted_size, share_data->move_dis_pre_point, 0.0);
            if (num_of_path == -1) {
                cout << "no path. throw" << endl;
                return;
            }
            if (num_of_path == -2) cout << "Line" << endl;
            if (num_of_path > 0)  cout << "Obstcale" << endl;
            cout << "num_of_path:" << points.size() << endl;

            viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(now_view_xyz(0), now_view_xyz(1), now_view_xyz(2)), pcl::PointXYZ(points[0](0), points[0](1), points[0](2)), 0, 128, 128, "trajectory" + to_string(i) + to_string(i + 1) + to_string(-1));
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "trajectory" + to_string(i) + to_string(i + 1) + to_string(-1));
            for (int k = 0; k < points.size() - 1; k++) {
                viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(points[k](0), points[k](1), points[k](2)), pcl::PointXYZ(points[k + 1](0), points[k + 1](1), points[k + 1](2)), 0, 128, 128, "trajectory" + to_string(i) + to_string(i + 1) + to_string(k));
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "trajectory" + to_string(i) + to_string(i + 1) + to_string(k));
            }

            // get waypoints
            vector<vector<float>> now_waypoints;
            Eigen::Matrix4d now_camera_pose_world = share_data->now_camera_pose_world;
            for(int j=0;j<points.size();j++){
                View temp_view(points[j]);
                temp_view.get_next_camera_pos(now_camera_pose_world,  share_data->object_center_world);
                Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 

                cout<<"temp_camera_pose_world:"<<temp_camera_pose_world<<endl;
                if(j==points.size()-1){
                    Eigen::Vector4d X(0.05, 0, 0, 1);
                    Eigen::Vector4d Y(0, 0.05, 0, 1);
                    Eigen::Vector4d Z(0, 0, 0.05, 1);
                    Eigen::Vector4d O(0, 0, 0, 1);
                    X = now_camera_pose_world * temp_view.pose.inverse() * X;
                    Y = now_camera_pose_world * temp_view.pose.inverse() * Y;
                    Z = now_camera_pose_world * temp_view.pose.inverse() * Z;
                    O = now_camera_pose_world * temp_view.pose.inverse() * O;
                    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i)+  to_string(j));
                    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i) + to_string(j));
                    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i) + to_string(j));
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(i) + to_string(j));
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(i) + to_string(j));
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(i) + to_string(j));
                }

                Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
                Eigen::Quaterniond temp_q(temp_rotation);
                vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };
                now_waypoints.push_back(temp_waypoint);

                now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
            }
            
            viewer->spinOnce(100);

            // try orginal path movement
            std::vector<geometry_msgs::Pose> waypoints_msg;
            generateWaypoints(now_waypoints, waypoints_msg);
            if (visitWaypoints(waypoints_msg)){
                num_of_movable_views++;
                vector<double> current_joint = getJointValues();
                view_space_pose.push_back(current_joint);
                ROS_INFO("MoveitClient: Arm moved to original waypoints 1");
            }
            else{
                // try z table height path movement
                ROS_ERROR("MoveitClient: Failed to move arm to original waypoints 1");
                
                // use the outside one to later update share_data->now_camera_pose_world
                now_camera_pose_world = share_data->now_camera_pose_world;

                vector<vector<float>> now_waypoints;
                for(int j=0;j<points.size();j++){
                    View temp_view(points[j]);
                    Eigen::Vector3d now_object_center_world = share_data->object_center_world;
                    now_object_center_world(2) = share_data->min_z_table;
                    temp_view.get_next_camera_pos(now_camera_pose_world,  now_object_center_world);
                    Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 

                    cout<<"temp_camera_pose_world:"<<temp_camera_pose_world<<endl;
                    if(j==points.size()-1){
                        Eigen::Vector4d X(0.05, 0, 0, 1);
                        Eigen::Vector4d Y(0, 0.05, 0, 1);
                        Eigen::Vector4d Z(0, 0, 0.05, 1);
                        Eigen::Vector4d O(0, 0, 0, 1);
                        X = now_camera_pose_world * temp_view.pose.inverse() * X;
                        Y = now_camera_pose_world * temp_view.pose.inverse() * Y;
                        Z = now_camera_pose_world * temp_view.pose.inverse() * Z;
                        O = now_camera_pose_world * temp_view.pose.inverse() * O;

                        viewer->removeCorrespondences("X" + to_string(i) + to_string(j));
                        viewer->removeCorrespondences("Y" + to_string(i) + to_string(j));
                        viewer->removeCorrespondences("Z" + to_string(i) + to_string(j));

                        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i)+  to_string(j));
                        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i) + to_string(j));
                        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i) + to_string(j));
                        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(i) + to_string(j));
                        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(i) + to_string(j));
                        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(i) + to_string(j));
                    }

                    Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
                    Eigen::Quaterniond temp_q(temp_rotation);
                    vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };
                    now_waypoints.push_back(temp_waypoint);

                    now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
                }

                std::vector<geometry_msgs::Pose> waypoints_msg;
                generateWaypoints(now_waypoints, waypoints_msg);
                if (visitWaypoints(waypoints_msg)){
                    num_of_movable_views++;
                    vector<double> current_joint = getJointValues();
                    view_space_pose.push_back(current_joint);
                    ROS_INFO("MoveitClient: Arm moved to look at table height waypoints 2");
                }
                else{
                    ROS_ERROR("MoveitClient: Failed to move arm to look at table height waypoints 2");

                    // use the outside one to later update share_data->now_camera_pose_world
                    now_camera_pose_world = share_data->now_camera_pose_world;

                    //original point movement
                    View temp_view(nbv_plan->now_view_space->views[init_view_ids[i+1]]);
                    Eigen::Vector3d now_object_center_world = share_data->object_center_world;
                    //now_object_center_world(2) = share_data->min_z_table;
                    temp_view.get_next_camera_pos(now_camera_pose_world,  now_object_center_world);
                    Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 
                    Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
                    Eigen::Quaterniond temp_q(temp_rotation);
                    vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };

                    // geometry_msgs::Pose target_orginal_pose;
                    // target_orginal_pose.position.x = temp_waypoint[0];
                    // target_orginal_pose.position.y = temp_waypoint[1];
                    // target_orginal_pose.position.z = temp_waypoint[2];
                    // target_orginal_pose.orientation.x = temp_waypoint[3];
                    // target_orginal_pose.orientation.y = temp_waypoint[4];
                    // target_orginal_pose.orientation.z = temp_waypoint[5];
                    // target_orginal_pose.orientation.w = temp_waypoint[6];
                    // cout << "target_orginal_pose:" << target_orginal_pose << endl;

                    // cout << "nbv_plan->now_view_space->views[init_view_ids[i+1]]:" << nbv_plan->now_view_space->views[init_view_ids[i+1]].init_pos << endl;
                    // cout << "temp_camera_pose_world:" << temp_camera_pose_world << endl;

                    vector<vector<float>> now_waypoints;
                    now_waypoints.push_back(temp_waypoint);
                    std::vector<geometry_msgs::Pose> waypoints_msg;
                    generateWaypoints(now_waypoints, waypoints_msg);

                    if (visitWaypoints(waypoints_msg)){
                        num_of_movable_views++;
                        vector<double> current_joint = getJointValues();
                        view_space_pose.push_back(current_joint);
                        ROS_INFO("MoveitClient: Arm moved to target original pose 3");
                        now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
                    }
                    else{
                        ROS_ERROR("MoveitClient: Failed to move arm to original pose 3");

                        // use the outside one to later update share_data->now_camera_pose_world
                        now_camera_pose_world = share_data->now_camera_pose_world;

                        //table height point movement
                        View temp_view(nbv_plan->now_view_space->views[init_view_ids[i+1]]);
                        Eigen::Vector3d now_object_center_world = share_data->object_center_world;
                        now_object_center_world(2) = share_data->min_z_table;
                        temp_view.get_next_camera_pos(now_camera_pose_world,  now_object_center_world);
                        Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 
                        Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
                        Eigen::Quaterniond temp_q(temp_rotation);
                        vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };

                        // geometry_msgs::Pose target_table_height_pose;
                        // target_table_height_pose.position.x = temp_waypoint[0];
                        // target_table_height_pose.position.y = temp_waypoint[1];
                        // target_table_height_pose.position.z = temp_waypoint[2];
                        // target_table_height_pose.orientation.x = temp_waypoint[3];
                        // target_table_height_pose.orientation.y = temp_waypoint[4];
                        // target_table_height_pose.orientation.z = temp_waypoint[5];
                        // target_table_height_pose.orientation.w = temp_waypoint[6];

                        vector<vector<float>> now_waypoints;
                        now_waypoints.push_back(temp_waypoint);
                        std::vector<geometry_msgs::Pose> waypoints_msg;
                        generateWaypoints(now_waypoints, waypoints_msg);

                        if (visitWaypoints(waypoints_msg)){
                            num_of_movable_views++;
                            vector<double> current_joint = getJointValues();
                            view_space_pose.push_back(current_joint);
                            ROS_INFO("MoveitClient: Arm moved to target look at table height pose 4");
                            now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
                        }
                        else{
                            ROS_ERROR("MoveitClient: Failed to move arm to look at table height pose 4");
                            //go to home
                            if (moveArmToHome()){
                                ROS_INFO("MoveitClient: Arm moved to home position");
                            }
                            else{
                                ROS_ERROR("MoveitClient: Failed to move arm to home position");
                            }
                            //change now camera pose world to home
                            // 0.011035, 0.390509, 0.338996, -0.542172, 0.472681, 0.521534, 0.458939
                            double start_x = 0.011035;
                            double start_y = 0.390509;
                            double start_z = 0.338996;
                            double start_qx = -0.542172;
                            double start_qy = 0.472681;
                            double start_qz = 0.521534;
                            double start_qw = 0.458939;
                            
                            Eigen::Quaterniond start_q(start_qw, start_qx, start_qy, start_qz);
                            Eigen::Matrix3d start_rotation = start_q.toRotationMatrix();
                            Eigen::Matrix4d start_pose_world = Eigen::Matrix4d::Identity();
                            start_pose_world.block<3,3>(0, 0) = start_rotation;
                            start_pose_world(0, 3) = start_x;
                            start_pose_world(1, 3) = start_y;
                            start_pose_world(2, 3) = start_z;

                            //update share_data->now_camera_pose_world
                            share_data->now_camera_pose_world = start_pose_world * share_data->camera_depth_to_rgb.inverse();

                            //use the outside one to later update share_data->now_camera_pose_world
                            now_camera_pose_world = share_data->now_camera_pose_world;
                            
                            Eigen::Vector3d now_view_xyz(now_camera_pose_world(0,3), now_camera_pose_world(1,3), now_camera_pose_world(2,3));
                            Eigen::Vector3d next_view_xyz(nbv_plan->now_view_space->views[init_view_ids[i+1]].init_pos(0), nbv_plan->now_view_space->views[init_view_ids[i+1]].init_pos(1), nbv_plan->now_view_space->views[init_view_ids[i+1]].init_pos(2));
                            vector<Eigen::Vector3d> points;
                            int num_of_path = get_trajectory_xyz(points, now_view_xyz, next_view_xyz, share_data->object_center_world, share_data->predicted_size, share_data->move_dis_pre_point, 0.0);
                            if (num_of_path == -1) {
                                cout << "no path. throw" << endl;
                                return;
                            }
                            if (num_of_path == -2) cout << "Line" << endl;
                            if (num_of_path > 0)  cout << "Obstcale" << endl;
                            cout << "num_of_path:" << points.size() << endl;

                            // try two path again
                            vector<vector<float>> now_waypoints;
                            for(int j=0;j<points.size();j++){
                                View temp_view(points[j]);
                                temp_view.get_next_camera_pos(now_camera_pose_world,  share_data->object_center_world);
                                Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 

                                cout<<"temp_camera_pose_world:"<<temp_camera_pose_world<<endl;
                                if(j==points.size()-1){
                                    Eigen::Vector4d X(0.05, 0, 0, 1);
                                    Eigen::Vector4d Y(0, 0.05, 0, 1);
                                    Eigen::Vector4d Z(0, 0, 0.05, 1);
                                    Eigen::Vector4d O(0, 0, 0, 1);
                                    X = now_camera_pose_world * temp_view.pose.inverse() * X;
                                    Y = now_camera_pose_world * temp_view.pose.inverse() * Y;
                                    Z = now_camera_pose_world * temp_view.pose.inverse() * Z;
                                    O = now_camera_pose_world * temp_view.pose.inverse() * O;
                                    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i)+  to_string(j));
                                    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i) + to_string(j));
                                    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i) + to_string(j));
                                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(i) + to_string(j));
                                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(i) + to_string(j));
                                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(i) + to_string(j));
                                }

                                Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
                                Eigen::Quaterniond temp_q(temp_rotation);
                                vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };
                                now_waypoints.push_back(temp_waypoint);

                                now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
                            }

                            // try orginal path movement from home
                            std::vector<geometry_msgs::Pose> waypoints_msg;
                            generateWaypoints(now_waypoints, waypoints_msg);
                            if (visitWaypoints(waypoints_msg)){
                                num_of_movable_views++;
                                vector<double> current_joint = getJointValues();
                                view_space_pose.push_back(current_joint);
                                ROS_INFO("MoveitClient: Arm moved to original waypoints from home 5");
                            }
                            else{
                                // try z table height path movement from home
                                ROS_ERROR("MoveitClient: Failed to move arm to original waypoints from home 5");
                                
                                // use the outside one to later update share_data->now_camera_pose_world
                                now_camera_pose_world = share_data->now_camera_pose_world;

                                vector<vector<float>> now_waypoints;
                                for(int j=0;j<points.size();j++){
                                    View temp_view(points[j]);
                                    Eigen::Vector3d now_object_center_world = share_data->object_center_world;
                                    now_object_center_world(2) = share_data->min_z_table;
                                    temp_view.get_next_camera_pos(now_camera_pose_world,  now_object_center_world);
                                    Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 

                                    cout<<"temp_camera_pose_world:"<<temp_camera_pose_world<<endl;
                                    if(j==points.size()-1){
                                        Eigen::Vector4d X(0.05, 0, 0, 1);
                                        Eigen::Vector4d Y(0, 0.05, 0, 1);
                                        Eigen::Vector4d Z(0, 0, 0.05, 1);
                                        Eigen::Vector4d O(0, 0, 0, 1);
                                        X = now_camera_pose_world * temp_view.pose.inverse() * X;
                                        Y = now_camera_pose_world * temp_view.pose.inverse() * Y;
                                        Z = now_camera_pose_world * temp_view.pose.inverse() * Z;
                                        O = now_camera_pose_world * temp_view.pose.inverse() * O;

                                        viewer->removeCorrespondences("X" + to_string(i) + to_string(j));
                                        viewer->removeCorrespondences("Y" + to_string(i) + to_string(j));
                                        viewer->removeCorrespondences("Z" + to_string(i) + to_string(j));

                                        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i)+  to_string(j));
                                        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i) + to_string(j));
                                        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i) + to_string(j));
                                        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(i) + to_string(j));
                                        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(i) + to_string(j));
                                        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(i) + to_string(j));
                                    }

                                    Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
                                    Eigen::Quaterniond temp_q(temp_rotation);
                                    vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };
                                    now_waypoints.push_back(temp_waypoint);

                                    now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
                                }

                                std::vector<geometry_msgs::Pose> waypoints_msg;
                                generateWaypoints(now_waypoints, waypoints_msg);
                                if (visitWaypoints(waypoints_msg)){
                                    num_of_movable_views++;
                                    vector<double> current_joint = getJointValues();
                                    view_space_pose.push_back(current_joint);
                                    ROS_INFO("MoveitClient: Arm moved to look at table height waypoints from home 6");
                                }
                                else{
                                    ROS_ERROR("MoveitClient: Failed to move arm to look at table height waypoints from home 6");

                                    now_camera_pose_world = share_data->now_camera_pose_world;

                                    //original point movement
                                    View temp_view(nbv_plan->now_view_space->views[init_view_ids[i+1]]);
                                    Eigen::Vector3d now_object_center_world = share_data->object_center_world;
                                    //now_object_center_world(2) = share_data->min_z_table;
                                    temp_view.get_next_camera_pos(now_camera_pose_world,  now_object_center_world);
                                    Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 
                                    Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
                                    Eigen::Quaterniond temp_q(temp_rotation);
                                    vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };

                                    // geometry_msgs::Pose target_orginal_pose;
                                    // target_orginal_pose.position.x = temp_waypoint[0];
                                    // target_orginal_pose.position.y = temp_waypoint[1];
                                    // target_orginal_pose.position.z = temp_waypoint[2];
                                    // target_orginal_pose.orientation.x = temp_waypoint[3];
                                    // target_orginal_pose.orientation.y = temp_waypoint[4];
                                    // target_orginal_pose.orientation.z = temp_waypoint[5];
                                    // target_orginal_pose.orientation.w = temp_waypoint[6];

                                    vector<vector<float>> now_waypoints;
                                    now_waypoints.push_back(temp_waypoint);
                                    std::vector<geometry_msgs::Pose> waypoints_msg;
                                    generateWaypoints(now_waypoints, waypoints_msg);

                                    if (visitWaypoints(waypoints_msg)){
                                        num_of_movable_views++;
                                        vector<double> current_joint = getJointValues();
                                        view_space_pose.push_back(current_joint);
                                        ROS_INFO("MoveitClient: Arm moved to target original pose 7");
                                        now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
                                    }
                                    else{
                                        ROS_ERROR("MoveitClient: Failed to move arm to original pose 7");

                                        now_camera_pose_world = share_data->now_camera_pose_world;

                                        //table height point movement
                                        View temp_view(nbv_plan->now_view_space->views[init_view_ids[i+1]]);
                                        Eigen::Vector3d now_object_center_world = share_data->object_center_world;
                                        now_object_center_world(2) = share_data->min_z_table;
                                        temp_view.get_next_camera_pos(now_camera_pose_world,  now_object_center_world);
                                        Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 
                                        Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
                                        Eigen::Quaterniond temp_q(temp_rotation);
                                        vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };

                                        // geometry_msgs::Pose target_table_height_pose;
                                        // target_table_height_pose.position.x = temp_waypoint[0];
                                        // target_table_height_pose.position.y = temp_waypoint[1];
                                        // target_table_height_pose.position.z = temp_waypoint[2];
                                        // target_table_height_pose.orientation.x = temp_waypoint[3];
                                        // target_table_height_pose.orientation.y = temp_waypoint[4];
                                        // target_table_height_pose.orientation.z = temp_waypoint[5];
                                        // target_table_height_pose.orientation.w = temp_waypoint[6];

                                        vector<vector<float>> now_waypoints;
                                        now_waypoints.push_back(temp_waypoint);
                                        std::vector<geometry_msgs::Pose> waypoints_msg;
                                        generateWaypoints(now_waypoints, waypoints_msg);

                                        if (visitWaypoints(waypoints_msg)){
                                            num_of_movable_views++;
                                            vector<double> current_joint = getJointValues();
                                            view_space_pose.push_back(current_joint);
                                            ROS_INFO("MoveitClient: Arm moved to target look at table height pose 8");
                                            now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
                                        }
                                        else{
                                            ROS_ERROR("MoveitClient: Failed to move arm to look at table height pose 8");

                                            //go to home
                                            if (moveArmToHome()){
                                                ROS_INFO("MoveitClient: Arm moved to home position");
                                            }
                                            else{
                                                ROS_ERROR("MoveitClient: Failed to move arm to home position");
                                            }
                                            //change now camera pose world to home
                                            // 0.011035, 0.390509, 0.338996, -0.542172, 0.472681, 0.521534, 0.458939
                                            double start_x = 0.011035;
                                            double start_y = 0.390509;
                                            double start_z = 0.338996;
                                            double start_qx = -0.542172;
                                            double start_qy = 0.472681;
                                            double start_qz = 0.521534;
                                            double start_qw = 0.458939;

                                            Eigen::Quaterniond start_q(start_qw, start_qx, start_qy, start_qz);
                                            Eigen::Matrix3d start_rotation = start_q.toRotationMatrix();
                                            Eigen::Matrix4d start_pose_world = Eigen::Matrix4d::Identity();
                                            start_pose_world.block<3,3>(0, 0) = start_rotation;
                                            start_pose_world(0, 3) = start_x;
                                            start_pose_world(1, 3) = start_y;
                                            start_pose_world(2, 3) = start_z;

                                            now_camera_pose_world = start_pose_world * share_data->camera_depth_to_rgb.inverse();

                                            return;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            geometry_msgs::PoseStamped current_pose = getCurrentPose();

            //更新share_data->now_camera_pose_world
            double now_x = current_pose.pose.position.x;
            double now_y = current_pose.pose.position.y;
            double now_z = current_pose.pose.position.z;
            double now_qx = current_pose.pose.orientation.x;
            double now_qy = current_pose.pose.orientation.y;
            double now_qz = current_pose.pose.orientation.z;
            double now_qw = current_pose.pose.orientation.w;

            Eigen::Quaterniond now_q(now_qw, now_qx, now_qy, now_qz);
            Eigen::Matrix3d now_rotation = now_q.toRotationMatrix();
            Eigen::Matrix4d now_current_camera_pose_world = Eigen::Matrix4d::Identity();
            now_current_camera_pose_world.block<3,3>(0, 0) = now_rotation;
            now_current_camera_pose_world(0, 3) = now_x;
            now_current_camera_pose_world(1, 3) = now_y;
            now_current_camera_pose_world(2, 3) = now_z;

            share_data->now_camera_pose_world = now_current_camera_pose_world * share_data->camera_depth_to_rgb.inverse();

            ROS_INFO("moved views: %d", num_of_movable_views);

            //如果不是home的话, 按精度0.01对比
            if(fabs(current_pose.pose.position.x - (-0.073738)) > 0.01 || fabs(current_pose.pose.position.y - (0.457149)) > 0.01 || fabs(current_pose.pose.position.z - (0.431199)) > 0.01 || fabs(current_pose.pose.orientation.x - (-0.002172)) > 0.01 || fabs(current_pose.pose.orientation.y - (0.684334)) > 0.01 || fabs(current_pose.pose.orientation.z - (-0.011654)) > 0.01 || fabs(current_pose.pose.orientation.w - (0.729073)) > 0.01){
                view_space_xyzq.push_back({current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z, current_pose.pose.orientation.x, current_pose.pose.orientation.y, current_pose.pose.orientation.z, current_pose.pose.orientation.w});
                //获取点云
                ros::Duration(2.0).sleep();
                getPointCloud();
                while(is_point_cloud_received_ == false){
                    ros::Duration(0.2).sleep();
                }
                is_point_cloud_received_ = false;
                nbv_plan->percept->precept(share_data->cloud_now);

                viewer->removePointCloud("cloud_scene");
                viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_scene, "cloud_scene");
                viewer->spinOnce(100);
            }

            // //如果末端轴很危险，就回到home
            // bool dangerous_check = false;
            // if (view_space_pose[view_space_pose.size()-1][3] < -4.71238898038469 || view_space_pose[view_space_pose.size()-1][3] > 4.71238898038469){
            //     dangerous_check = true;
            // }
            // if (view_space_pose[view_space_pose.size()-1][4] < -4.71238898038469 || view_space_pose[view_space_pose.size()-1][4] > 4.71238898038469){
            //     dangerous_check = true;
            // }
            // if(view_space_pose[view_space_pose.size()-1][5] < -4.71238898038469 || view_space_pose[view_space_pose.size()-1][5] > 4.71238898038469){
            //     dangerous_check = true;
            // }
            // if(dangerous_check){
            //     ROS_ERROR("MoveitClient: Arm moved to dangerous position, move to home position");
            //     if (moveArmToHome()){
            //         ROS_INFO("MoveitClient: Arm moved to home position");
            //     }
            //     else{
            //         ROS_ERROR("MoveitClient: Failed to move arm to home position");
            //     }

            //     //0.011035, 0.390509, 0.338996, -0.542172, 0.472681, 0.521534, 0.458939
            //     double start_x = 0.011035;
            //     double start_y = 0.390509;
            //     double start_z = 0.338996;
            //     double start_qx = -0.542172;
            //     double start_qy = 0.472681;
            //     double start_qz = 0.521534;
            //     double start_qw = 0.458939;

            //     Eigen::Quaterniond start_q(start_qw, start_qx, start_qy, start_qz);
            //     Eigen::Matrix3d start_rotation = start_q.toRotationMatrix();
            //     Eigen::Matrix4d start_pose_world = Eigen::Matrix4d::Identity();
            //     start_pose_world.block<3,3>(0, 0) = start_rotation;
            //     start_pose_world(0, 3) = start_x;
            //     start_pose_world(1, 3) = start_y;
            //     start_pose_world(2, 3) = start_z;

            //     share_data->now_camera_pose_world = start_pose_world * share_data->camera_depth_to_rgb.inverse();
            // }
        }

        cout<<"num_of_movable_views: "<<num_of_movable_views<<endl;

        //把nbv_plan->now_view_space->views按映射排列回来
        vector<vector<double>> temp_view_space_pose;    temp_view_space_pose.resize(init_view_ids.size());
        vector<vector<double>> temp_view_space_xyzq;    temp_view_space_xyzq.resize(init_view_ids.size());
        for(int i=0;i<init_view_ids.size();i++){
            temp_view_space_pose[init_view_ids[i]] = view_space_pose[i];
            temp_view_space_xyzq[init_view_ids[i]] = view_space_xyzq[i];
        }
        view_space_pose = temp_view_space_pose;
        view_space_xyzq = temp_view_space_xyzq;

        //保存结果
        if(is_first_time){
            share_data->access_directory(share_data->pcd_file_path);
            //初始仅保存场景点云
            pcl::io::savePCDFileBinary<pcl::PointXYZRGB>(share_data->pcd_file_path + share_data->name_of_pcd + "_scene.pcd", *share_data->cloud_scene);
        }
        else{
            share_data->access_directory(share_data->pcd_file_path);
            //保存场景点云，物体点云，视角，视角位姿
            ofstream fout_viewspace_pose(share_data->pcd_file_path + share_data->name_of_pcd + "_vs_pose.txt");
            for(int i = 0; i < view_space_pose.size(); i++){
                fout_viewspace_pose<<view_space_pose[i][0]<<" "<<view_space_pose[i][1]<<" "<<view_space_pose[i][2]<<" "<<view_space_pose[i][3]<<" "<<view_space_pose[i][4]<<" "<<view_space_pose[i][5]<<endl;
            }

            ofstream fout_viewspace_xyzq(share_data->pcd_file_path + share_data->name_of_pcd + "_vs_xyzq.txt");
            for(int i = 0; i < view_space_xyzq.size(); i++){
                fout_viewspace_xyzq<<view_space_xyzq[i][0]<<" "<<view_space_xyzq[i][1]<<" "<<view_space_xyzq[i][2]<<" "<<view_space_xyzq[i][3]<<" "<<view_space_xyzq[i][4]<<" "<<view_space_xyzq[i][5]<<" "<<view_space_xyzq[i][6]<<endl;
            }

            //filter the obejct model
            pcl::PassThrough<pcl::PointXYZRGB> pass;
            pass.setFilterLimitsNegative(false); 
            pass.setInputCloud(share_data->cloud_final);          
            pass.setFilterFieldName("x");         
            pass.setFilterLimits(share_data->object_center_world(0) - share_data->predicted_size, share_data->object_center_world(0) + share_data->predicted_size);       
            pass.filter(*share_data->cloud_final);
            pass.setInputCloud(share_data->cloud_final);         
            pass.setFilterFieldName("y");         
            pass.setFilterLimits(share_data->object_center_world(1) - share_data->predicted_size, share_data->object_center_world(1) + share_data->predicted_size);          
            pass.filter(*share_data->cloud_final);
            pass.setInputCloud(share_data->cloud_final);     
            pass.setFilterFieldName("z");      
            pass.setFilterLimits(max(share_data->height_of_ground, share_data->object_center_world(2) - share_data->predicted_size), min(share_data->height_to_filter_arm,share_data->object_center_world(2) + share_data->predicted_size));     
            pass.filter(*share_data->cloud_final);

            pcl::io::savePCDFileBinary<pcl::PointXYZRGB>(share_data->pcd_file_path + share_data->name_of_pcd + ".pcd", *share_data->cloud_final);
            pcl::io::savePCDFileBinary<pcl::PointXYZRGB>(share_data->pcd_file_path + share_data->name_of_pcd + "_scene.pcd", *share_data->cloud_scene);

            int num = 0;
            unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
            for (int j = 0; j < share_data->cloud_final->points.size(); j++) {
                octomap::OcTreeKey key = share_data->ground_truth_model->coordToKey(share_data->cloud_final->points[j].x, share_data->cloud_final->points[j].y, share_data->cloud_final->points[j].z);
                if (voxel->find(key) == voxel->end()) {
                    (*voxel)[key] = num++;
                }
            }

            ofstream fout_GT_points_number(share_data->pcd_file_path + share_data->name_of_pcd + ".txt");
            fout_GT_points_number<<num;

            ofstream fout_view_space(share_data->pcd_file_path + share_data->name_of_pcd + "_vs.txt");
            fout_view_space << nbv_plan->now_view_space->num_of_views << '\n';
            fout_view_space << nbv_plan->now_view_space->object_center_world(0) << ' ' << nbv_plan->now_view_space->object_center_world(1) << ' ' << nbv_plan->now_view_space->object_center_world(2) << '\n';
            fout_view_space << nbv_plan->now_view_space->predicted_size << '\n';
            for (int i = 0; i < nbv_plan->now_view_space->num_of_views; i++)
                fout_view_space << nbv_plan->now_view_space->views[i].init_pos(0) << ' ' << nbv_plan->now_view_space->views[i].init_pos(1) << ' ' << nbv_plan->now_view_space->views[i].init_pos(2) << '\n';
        }
        
        cout<<"GT mode finish"<<endl;

        while (!viewer->wasStopped())
        {
            viewer->spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }
        viewer->close();

        pcl::visualization::PCLVisualizer::Ptr viewer1 = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("cloud_final"));
        viewer1->setBackgroundColor(255, 255, 255);
        viewer1->addCoordinateSystem(0.1);
        viewer1->initCameraParameters();
        viewer1->setCameraPosition(0.8, 0.8, 0.8, share_data->object_center_world(0), share_data->object_center_world(1), share_data->object_center_world(2), 0, 0, 1);
        viewer1->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_final, "cloud_final)");
        viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "cloud_final");
        while (!viewer1->wasStopped())
        {
            viewer1->spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }
        viewer1->close();

        return;
    }

    // go to the initial position and plesae set from first view point
    cout<<"reconstruction mode"<<endl;

    if (moveArmToHome()){
        ROS_INFO("MoveitClient: Arm moved to home position");
    }
    else{
        ROS_ERROR("MoveitClient: Failed to move arm to home position");
    }
    ros::Duration(2.0).sleep();

    vector<double> current_joints = getJointValues();

    double initial_time_stamp = ros::Time::now().toSec();

    ifstream fin_viewspace_pose(share_data->pcd_file_path + share_data->name_of_pcd + "_vs_pose.txt");
    vector<vector<double>> view_space_pose;
    view_space_pose.resize(share_data->num_of_views);
    for(int i = 0; i < share_data->num_of_views; i++){
        view_space_pose[i].resize(6);
        fin_viewspace_pose>>view_space_pose[i][0]>>view_space_pose[i][1]>>view_space_pose[i][2]>>view_space_pose[i][3]>>view_space_pose[i][4]>>view_space_pose[i][5];
    }

    // go to the first view point
    if (visitJoints(view_space_pose[share_data->first_view_id])){
        ROS_INFO("MoveitClient: Arm moved to first view pose");
    }
    else{
        ROS_ERROR("MoveitClient: Failed to move arm to first view pose");
    }
    ros::Duration(1.0).sleep();

    // initialize subscribers to receive init pointcloud
    getPointCloud();
    while(is_point_cloud_received_ == false){
        ros::Duration(0.2).sleep();
    }
    is_point_cloud_received_ = false;
    // initialize planner to do normal planning
    nbv_plan = new NBV_Planner(share_data);
    getPointCloud();
    while(is_point_cloud_received_ == false){
        ros::Duration(0.2).sleep();
    }
    is_point_cloud_received_ = false;
    nbv_plan->percept->precept(share_data->cloud_now);

    share_data->cloud_scene->header.frame_id = "base_link";
    pcl_conversions::toPCL(ros::Time::now(), share_data->cloud_scene->header.stamp);
    pc_pub_.publish(*(share_data->cloud_scene));

    //set up the viewer and first view
    auto viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("Scene and Path"));
    viewer->setBackgroundColor(255, 255, 255);
    viewer->addCoordinateSystem(0.1);
    viewer->initCameraParameters();
    pcl::visualization::Camera cam;
    viewer->getCameraParameters(cam);
    cam.window_size[0] = 1920;
    cam.window_size[1] = 1080;
    viewer->setCameraParameters(cam);
    viewer->setCameraPosition(0.8, 0.8, 0.8, share_data->object_center_world(0), share_data->object_center_world(1), share_data->object_center_world(2), 0, 0, 1);

    Eigen::Vector4d X(0.05, 0, 0, 1);
    Eigen::Vector4d Y(0, 0.05, 0, 1);
    Eigen::Vector4d Z(0, 0, 0.05, 1);
    Eigen::Vector4d O(0, 0, 0, 1);
    X = nbv_plan->now_best_view->pose.inverse() * X;
    Y = nbv_plan->now_best_view->pose.inverse() * Y;
    Z = nbv_plan->now_best_view->pose.inverse() * Z;
    O = nbv_plan->now_best_view->pose.inverse() * O;
    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(-1));
    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(-1));
    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(-1));
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(-1));
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(-1));
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(-1));

    viewer->addPointCloud(share_data->cloud_scene, "cloud_scene");

    if(share_data->show_bbx) nbv_plan->now_view_space->add_bbx_to_cloud(viewer);

    viewer->spinOnce(100);

    Eigen::Affine3d pose_eigen;
    pose_eigen.matrix() = nbv_plan->now_best_view->pose.inverse().matrix();
    geometry_msgs::PoseStamped pose_msg;
    tf::poseEigenToMsg(pose_eigen, pose_msg.pose);
    pose_array_.poses.push_back(pose_msg.pose);
    pose_array_.header.frame_id = "base_link";
    pose_array_.header.stamp = ros::Time::now();

    pose_msg.header.frame_id = "base_link";
    pose_msg.header.stamp = ros::Time::now();
    path_.poses.push_back(pose_msg);

    pose_array_pub_.publish(pose_array_);

    int num_of_movable_views = 1;

    // VPP loop
	while (status != "Over") {
        // plan until the status is "WaitMoving"
        while (status != "WaitMoving") {
            nbv_plan->plan();
            if (status != nbv_plan->out_status()) {
                status = nbv_plan->out_status();
                cout << "NBV_Planner's status is " << status << endl;
            }
        }

        if(share_data->show_bbx){
            //更新viewer的BBX
            nbv_plan->now_view_space->remove_bbx(viewer);
            nbv_plan->now_view_space->add_bbx_to_cloud(viewer);
            viewer->spinOnce(100);
        }

        octomap::ColorOcTree copied_ot(share_data->octo_model->getResolution());
        for (auto it = share_data->octo_model->begin_leafs(), end = share_data->octo_model->end_leafs(); it != end; ++it) {
            if (share_data->octo_model->isNodeOccupied(*it)) {
                copied_ot.updateNode(it.getKey(), true);
                copied_ot.setNodeColor(it.getKey(), it->getColor().r, it->getColor().g, it->getColor().b);
            }
        }
        copied_ot.updateInnerOccupancy();
        octomap_msgs::Octomap map_msg;
        map_msg.header.frame_id = "base_link";
        map_msg.header.stamp = ros::Time::now();
        bool msg_generated = octomap_msgs::fullMapToMsg(copied_ot, map_msg);
        if (msg_generated)
        {
            octomap_pub_.publish(map_msg);
        }

        // plan once to check the status
        nbv_plan->plan();
        if (status != nbv_plan->out_status()) {
            status = nbv_plan->out_status();
            cout << "NBV_Planner's status is " << status << endl;
        }

        //record iteration time
        double number_of_views_time = ros::Time::now().toSec() - initial_time_stamp;
        share_data->access_directory(share_data->save_path + "/number_of_views_time");
        ofstream fout_number_of_views_time(share_data->save_path + "/number_of_views_time/"+to_string(nbv_plan->iterations + 1)+".txt", ios::app);
        fout_number_of_views_time<<number_of_views_time<<endl;

        //check if the status is "Over"
        if (status == "Over") {
            break;
        }

        // Motion Planning, waypoints to move to next best view
        Eigen::Vector3d now_view_xyz(share_data->now_camera_pose_world(0, 3), share_data->now_camera_pose_world(1, 3), share_data->now_camera_pose_world(2, 3));
        Eigen::Vector3d next_view_xyz = nbv_plan->now_best_view->init_pos;
        vector<Eigen::Vector3d> points;
        int num_of_path = get_trajectory_xyz(points, now_view_xyz, next_view_xyz, nbv_plan->now_view_space->object_center_world, nbv_plan->now_view_space->predicted_size, share_data->move_dis_pre_point, share_data->safe_distance);
        if (num_of_path == -1) {
            cout << "no path. throw" << endl;
            return;
        }
        if (num_of_path == -2) cout << "Line" << endl;
        if (num_of_path > 0)  cout << "Obstcale" << endl;
        cout << "num_of_path:" << points.size() << endl;

        bool is_global_path = share_data->method_of_IG == SCVP && nbv_plan->iterations >= share_data->num_of_nbvs_combined;

        if(is_global_path) viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(now_view_xyz(0), now_view_xyz(1), now_view_xyz(2)), pcl::PointXYZ(points[0](0), points[0](1), points[0](2)), 128, 0, 128, "trajectory" + to_string(nbv_plan->iterations)  + to_string(-1));
        else viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(now_view_xyz(0), now_view_xyz(1), now_view_xyz(2)), pcl::PointXYZ(points[0](0), points[0](1), points[0](2)), 0, 128, 128, "trajectory" + to_string(nbv_plan->iterations) + to_string(-1));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "trajectory"  + to_string(nbv_plan->iterations) + to_string(-1));
        for (int k = 0; k < points.size() - 1; k++) {
            if(is_global_path) viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(points[k](0), points[k](1), points[k](2)), pcl::PointXYZ(points[k + 1](0), points[k + 1](1), points[k + 1](2)), 128, 0, 128, "trajectory" + to_string(nbv_plan->iterations) + to_string(k)); 
            else viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(points[k](0), points[k](1), points[k](2)), pcl::PointXYZ(points[k + 1](0), points[k + 1](1), points[k + 1](2)), 0, 128, 128, "trajectory" + to_string(nbv_plan->iterations) + to_string(k)); 
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "trajectory" + to_string(nbv_plan->iterations) + to_string(k));
        }
        viewer->spinOnce(100);

        if(is_global_path)
        {
            Eigen::Affine3d pose_eigen;
            pose_eigen.matrix() = share_data->now_camera_pose_world;
            geometry_msgs::PoseStamped pose_msg;
            tf::poseEigenToMsg(pose_eigen, pose_msg.pose);
            pose_msg.header.frame_id = "base_link";
            pose_msg.header.stamp = ros::Time::now();
            global_path_.poses.push_back(pose_msg);
        }
        
        // get waypoints
        vector<vector<float>> now_waypoints;
        Eigen::Matrix4d now_camera_pose_world = share_data->now_camera_pose_world;
        for(int j=0;j<points.size();j++){
            View temp_view(points[j]);
            temp_view.get_next_camera_pos(now_camera_pose_world,  nbv_plan->now_view_space->object_center_world);
            Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 

            cout<<"temp_camera_pose_world:"<<temp_camera_pose_world<<endl;
            if(j==points.size()-1){
                Eigen::Vector4d X(0.05, 0, 0, 1);
                Eigen::Vector4d Y(0, 0.05, 0, 1);
                Eigen::Vector4d Z(0, 0, 0.05, 1);
                Eigen::Vector4d O(0, 0, 0, 1);
                X = now_camera_pose_world * temp_view.pose.inverse() * X;
                Y = now_camera_pose_world * temp_view.pose.inverse() * Y;
                Z = now_camera_pose_world * temp_view.pose.inverse() * Z;
                O = now_camera_pose_world * temp_view.pose.inverse() * O;
                viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(nbv_plan->iterations) + to_string(j));
                viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(nbv_plan->iterations) + to_string(j));
                viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(nbv_plan->iterations) + to_string(j));
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(nbv_plan->iterations) + to_string(j));
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(nbv_plan->iterations) + to_string(j));
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(nbv_plan->iterations) + to_string(j));
                viewer->spinOnce(100);
            }

            Eigen::Affine3d pose_eigen;
            pose_eigen.matrix() = now_camera_pose_world * temp_view.pose.inverse().matrix();
            geometry_msgs::PoseStamped pose_msg;
            tf::poseEigenToMsg(pose_eigen, pose_msg.pose);
            pose_msg.header.frame_id = "base_link";
            pose_msg.header.stamp = ros::Time::now();
            if(is_global_path)
                global_path_.poses.push_back(pose_msg);
            else
                path_.poses.push_back(pose_msg);
                
            if (j==points.size()-1)
            {
                pose_array_.poses.push_back(pose_msg.pose);
                pose_array_.header.frame_id = "base_link";
                pose_array_.header.stamp = ros::Time::now();
                pose_array_pub_.publish(pose_array_);
                if(is_global_path)
                {
                    global_path_.header.frame_id = "base_link";
                    global_path_.header.stamp = ros::Time::now();
                    global_path_pub_.publish(global_path_);
                }
                else
                {
                    path_.header.frame_id = "base_link";
                    path_.header.stamp = ros::Time::now();
                    path_pub_.publish(path_);
                }
                
            }

            Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
            Eigen::Quaterniond temp_q(temp_rotation);
            vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };
            now_waypoints.push_back(temp_waypoint);

            now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
        }
        // move arm to waypoints
        std::vector<geometry_msgs::Pose> waypoints_msg;
        generateWaypoints(now_waypoints, waypoints_msg);
        if (visitWaypoints(waypoints_msg)){
            num_of_movable_views++;
            ROS_INFO("MoveitClient: Arm moved to original waypoints 1st");
        }
        else{
            // try z table height path movement
            ROS_ERROR("MoveitClient: Failed to move arm to original waypoints 1st");
            
            now_camera_pose_world = share_data->now_camera_pose_world;

            vector<vector<float>> now_waypoints;
            for(int j=0;j<points.size();j++){
                View temp_view(points[j]);
                Eigen::Vector3d now_object_center_world = nbv_plan->now_view_space->object_center_world;
                now_object_center_world(2) = share_data->min_z_table;
                temp_view.get_next_camera_pos(now_camera_pose_world,  now_object_center_world);
                Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 

                cout<<"temp_camera_pose_world:"<<temp_camera_pose_world<<endl;
                if(j==points.size()-1){
                    Eigen::Vector4d X(0.05, 0, 0, 1);
                    Eigen::Vector4d Y(0, 0.05, 0, 1);
                    Eigen::Vector4d Z(0, 0, 0.05, 1);
                    Eigen::Vector4d O(0, 0, 0, 1);
                    X = now_camera_pose_world * temp_view.pose.inverse() * X;
                    Y = now_camera_pose_world * temp_view.pose.inverse() * Y;
                    Z = now_camera_pose_world * temp_view.pose.inverse() * Z;
                    O = now_camera_pose_world * temp_view.pose.inverse() * O;

                    viewer->removeCorrespondences("X" + to_string(nbv_plan->iterations) + to_string(j));
                    viewer->removeCorrespondences("Y" + to_string(nbv_plan->iterations) + to_string(j));
                    viewer->removeCorrespondences("Z" + to_string(nbv_plan->iterations) + to_string(j));

                    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(nbv_plan->iterations)+  to_string(j));
                    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(nbv_plan->iterations) + to_string(j));
                    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(nbv_plan->iterations) + to_string(j));
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(nbv_plan->iterations) + to_string(j));
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(nbv_plan->iterations) + to_string(j));
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(nbv_plan->iterations) + to_string(j));
                    viewer->spinOnce(100);
                }

                Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
                Eigen::Quaterniond temp_q(temp_rotation);
                vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };
                now_waypoints.push_back(temp_waypoint);

                now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
            }

            std::vector<geometry_msgs::Pose> waypoints_msg;
            generateWaypoints(now_waypoints, waypoints_msg);
            if (visitWaypoints(waypoints_msg)){
                num_of_movable_views++;
                ROS_INFO("MoveitClient: Arm moved to look at table height waypoints 2nd");
            }
            else{
                ROS_ERROR("MoveitClient: Failed to move arm to look at table height waypoints 2nd");

                //如果有障碍物，则必须回到home位置
                if(num_of_path > 0){
                    //回到home位置
                    if (moveArmToHome()){
                        ROS_INFO("MoveitClient: Arm moved to home position");
                    }
                    else{
                        ROS_ERROR("MoveitClient: Failed to move arm to home position");
                    }
                }

                //然后访问GT位置
                if (visitJoints(view_space_pose[nbv_plan->now_best_view->id])){
                    num_of_movable_views++;
                    ROS_INFO("MoveitClient: Arm moved to target gt pose 3rd");
                }
                else{
                    ROS_ERROR("MoveitClient: Failed to move arm to target gt pose 3rd");
                    //do nothing
                }  
            }
        }
        geometry_msgs::PoseStamped current_pose = getCurrentPose();
        ros::Duration(1.0).sleep();
        // update flag outside
        share_data->move_on = true;
        
        // plan until the status is "WaitData"
        while (status != "WaitData") {
            nbv_plan->plan();
            if (status != nbv_plan->out_status()) {
                status = nbv_plan->out_status();
                cout << "NBV_Planner's status is " << status << endl;
            }
        }
        getPointCloud();
        while(is_point_cloud_received_ == false){
            ros::Duration(0.2).sleep();
        }
        is_point_cloud_received_ = false;
        nbv_plan->percept->precept(share_data->cloud_now);

        viewer->removePointCloud("cloud_scene");
        viewer->addPointCloud(share_data->cloud_scene, "cloud_scene");

        viewer->spinOnce(100);

        share_data->cloud_scene->header.frame_id = "base_link";
        pcl_conversions::toPCL(ros::Time::now(), share_data->cloud_scene->header.stamp);
        pc_pub_.publish(*(share_data->cloud_scene));
    }

    cout<<"num_of_planned_views: "<<nbv_plan->iterations + 1<<endl;
    cout<<"num_of_movable_views: "<<num_of_movable_views<<endl;

    //filter the obejct model
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setFilterLimitsNegative(false); 
    pass.setInputCloud(share_data->cloud_final);          
    pass.setFilterFieldName("x");         
    pass.setFilterLimits(share_data->object_center_world(0) - share_data->predicted_size, share_data->object_center_world(0) + share_data->predicted_size);       
    pass.filter(*share_data->cloud_final);
    pass.setInputCloud(share_data->cloud_final);         
    pass.setFilterFieldName("y");         
    pass.setFilterLimits(share_data->object_center_world(1) - share_data->predicted_size, share_data->object_center_world(1) + share_data->predicted_size);          
    pass.filter(*share_data->cloud_final);
    pass.setInputCloud(share_data->cloud_final);     
    pass.setFilterFieldName("z");      
    pass.setFilterLimits(max(share_data->height_of_ground, share_data->object_center_world(2) - share_data->predicted_size), min(share_data->height_to_filter_arm,share_data->object_center_world(2) + share_data->predicted_size));     
    pass.filter(*share_data->cloud_final);  

    pcl::io::savePCDFileBinary<pcl::PointXYZRGB>(share_data->save_path + "/" + share_data->name_of_pcd + "_final.pcd", *share_data->cloud_final);
    pcl::io::savePCDFileBinary<pcl::PointXYZRGB>(share_data->save_path + "/" + share_data->name_of_pcd + "_scene.pcd", *share_data->cloud_scene);

    double final_time_stamp = ros::Time::now().toSec();

    cout<<"reconstruction mode finish"<<endl;

    ofstream fout_time(share_data->save_path + "/runtime.txt");
    fout_time<<final_time_stamp - initial_time_stamp<<endl;

    if(share_data->final_viewer){
        //最终显示场景和重建的点云
        viewer->setCameraPosition(-0.2, 0.8, 1.0, share_data->object_center_world(0), share_data->object_center_world(1), share_data->object_center_world(2), 0, 0, 1);
        while (!viewer->wasStopped())
        {
            viewer->spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }
        viewer->close();
        pcl::visualization::PCLVisualizer::Ptr viewer1 = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("cloud_final"));
        viewer1->setBackgroundColor(255, 255, 255);
        viewer1->addCoordinateSystem(0.1);
        viewer1->initCameraParameters();
        viewer1->setCameraPosition(0.8, 0.8, 0.8, share_data->object_center_world(0), share_data->object_center_world(1), share_data->object_center_world(2), 0, 0, 1);
        viewer1->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_final, "cloud_final)");
        viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "cloud_final");
        while (!viewer1->wasStopped())
        {
            viewer1->spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }
        viewer1->close();
    }

    return;
}

void RosInterface::getPointCloud()
{
    // point cloud subscriber
    sensor_msgs::PointCloud2ConstPtr point_cloud_msg = ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/camera/depth/color/points", ros::Duration(10.0));

    if (point_cloud_msg)
    {
        // pre-process point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc = preProcessPointCloud(point_cloud_msg);

        *share_data->cloud_now = *pc;

        // set flag
        is_point_cloud_received_ = true;
    }
}

void RosInterface::initMoveitClient()
{
    move_group_arm_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(PLANNING_GROUP_ARM);

    // move to home position
    //if (moveArmToHome())
    //{
    //    ROS_INFO("MoveitClient: Arm moved to home position");
    //}
    //else
    //{
    //    ROS_ERROR("MoveitClient: Failed to move arm to home position");
    //}

    // print current pose
    // auto _ = getCurrentPose();
}

bool RosInterface::moveArmToHome()
{
    // move to home position
    // 0.011035, 0.390509, 0.338996, -0.542172, 0.472681, 0.521534, 0.458939
     geometry_msgs::Pose home_pose;

    // now home pose
    //1.219672, -1.815668, 1.872075, 4.685153, -1.596196, 4.232074
    vector<double> home_joints = { 1.219672, -1.815668, 1.872075, 4.685153, -1.596196, 4.232074 };

    // move arm to waypoints
    if (visitJoints(home_joints)){
        ROS_INFO("MoveitClient: Arm moved to home");
        return true;
    }
    else{
        ROS_ERROR("MoveitClient: Failed to move arm to home");
        return false;
    }
}

bool RosInterface::visitPose(const geometry_msgs::Pose& pose)
{
    move_group_arm_->setJointValueTarget(pose);

    bool success = (move_group_arm_->plan(arm_plan_) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success)
    {
        move_group_arm_->execute(arm_plan_);
        ROS_INFO("MoveitClient: Arm moved to the position");
        return true;
    }
    else
    {
        ROS_ERROR("MoveitClient: Failed to move arm to the position");
        return false;
    }
}

bool RosInterface::visitJoints(const std::vector<double>& joints)
{
    move_group_arm_->setJointValueTarget(joints);

    bool success = (move_group_arm_->plan(arm_plan_) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success)
    {
        move_group_arm_->execute(arm_plan_);
        ROS_INFO("MoveitClient: Arm moved to the position");
        return true;
    }
    else
    {
        ROS_ERROR("MoveitClient: Failed to move arm to the position");
        return false;
    }
}

bool RosInterface::visitWaypoints(const std::vector<geometry_msgs::Pose>& waypoints, float jump_threshold)
{
    // move arm in a cartesian path
    moveit_msgs::RobotTrajectory trajectory;
    double fraction = move_group_arm_->computeCartesianPath(waypoints, 0.01, jump_threshold, trajectory);

    int maxtries = 100;
    int attempts = 0;
    while (fraction < 1.0 && attempts < maxtries)
    {
        fraction = move_group_arm_->computeCartesianPath(waypoints, 0.01, jump_threshold, trajectory);
        attempts++;
        
        if(attempts % 30 == 0){
            ROS_INFO("MoveitClient: Cartesian path computed with fraction: %f ", fraction);
            ROS_INFO("MoveitClient: Retrying to compute cartesian path");
        }
    }

    ROS_INFO("MoveitClient: Cartesian path computed with fraction: %f ", fraction);

    if (fraction < 1.0)
    {
        ROS_ERROR("MoveitClient: Failed to compute cartesian path");
        return false;
    }

    // execute trajectory
    moveit::core::MoveItErrorCode result = move_group_arm_->execute(trajectory);
    if (result != moveit::core::MoveItErrorCode::SUCCESS)
    {
        ROS_ERROR("MoveitClient: Failed to execute trajectory");
        return false;
    }

    return true;
}

void RosInterface::generateWaypoints(const std::vector<std::vector<float>>& waypoints, std::vector<geometry_msgs::Pose>& waypoints_msg)
{
    for (const auto& waypoint : waypoints)
    {
        geometry_msgs::Pose waypoint_msg;
        waypoint_msg.position.x = waypoint[0];
        waypoint_msg.position.y = waypoint[1];
        waypoint_msg.position.z = waypoint[2];

        waypoint_msg.orientation.x = waypoint[3];
        waypoint_msg.orientation.y = waypoint[4];
        waypoint_msg.orientation.z = waypoint[5];
        waypoint_msg.orientation.w = waypoint[6];

        waypoints_msg.push_back(waypoint_msg);
    }
}

geometry_msgs::PoseStamped RosInterface::getCurrentPose()
{
    geometry_msgs::PoseStamped current_pose = move_group_arm_->getCurrentPose();
    ROS_INFO("MoveitClient: Current pose (xyz): %f, %f, %f", current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z);

    // convert quaternion to euler angles
    tf2::Quaternion q(current_pose.pose.orientation.x,
                     current_pose.pose.orientation.y,
                     current_pose.pose.orientation.z,
                     current_pose.pose.orientation.w);

    ROS_INFO("MoveitClient: Current pose (q): %f, %f, %f, %f", q.x(), q.y(), q.z(), q.w());
    
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    // convert to degrees
    roll = roll * 180 / M_PI;
    pitch = pitch * 180 / M_PI;
    yaw = yaw * 180 / M_PI;

    ROS_INFO("MoveitClient: Current pose (rpy): %f, %f, %f", roll, pitch, yaw);

    return current_pose;
}

std::vector<double> RosInterface::getJointValues()
{
    std::vector<double> joint_values = move_group_arm_->getCurrentJointValues();

    ROS_INFO("MoveitClient: Current joints: %f, %f, %f, %f, %f, %f", joint_values[0], joint_values[1], joint_values[2], joint_values[3], joint_values[4], joint_values[5]);

    return joint_values;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "ma_scvp_real");
    ros::NodeHandle nh;
    ros::AsyncSpinner spinner(4);
    spinner.start();
    RosInterface ros_interface(nh);
    ros_interface.run();
    ros::waitForShutdown();
    return 0;
}