#pragma once
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <fstream>  
#include <string>  
#include <vector> 
#include <thread>
#include <chrono>
#include <atomic>
#include <ctime> 
#include <cmath>
#include <mutex>
#include <map>
#include <set>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_features.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>


#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


using namespace std;

/** \brief Distortion model: defines how pixel coordinates should be mapped to sensor coordinates. */
typedef enum rs2_distortion
{
	RS2_DISTORTION_NONE, /**< Rectilinear images. No distortion compensation required. */
	RS2_DISTORTION_MODIFIED_BROWN_CONRADY, /**< Equivalent to Brown-Conrady distortion, except that tangential distortion is applied to radially distorted points */
	RS2_DISTORTION_INVERSE_BROWN_CONRADY, /**< Equivalent to Brown-Conrady distortion, except undistorts image instead of distorting it */
	RS2_DISTORTION_FTHETA, /**< F-Theta fish-eye distortion model */
	RS2_DISTORTION_BROWN_CONRADY, /**< Unmodified Brown-Conrady distortion model */
	RS2_DISTORTION_KANNALA_BRANDT4, /**< Four parameter Kannala Brandt distortion model */
	RS2_DISTORTION_COUNT                   /**< Number of enumeration values. Not a valid input: intended to be used in for-loops. */
} rs2_distortion;

/** \brief Video stream intrinsics. */
typedef struct rs2_intrinsics
{
	int           width;     /**< Width of the image in pixels */
	int           height;    /**< Height of the image in pixels */
	float         ppx;       /**< Horizontal coordinate of the principal point of the image, as a pixel offset from the left edge */
	float         ppy;       /**< Vertical coordinate of the principal point of the image, as a pixel offset from the top edge */
	float         fx;        /**< Focal length of the image plane, as a multiple of pixel width */
	float         fy;        /**< Focal length of the image plane, as a multiple of pixel height */
	rs2_distortion model;    /**< Distortion model of the image */
	float         coeffs[5]; /**< Distortion coefficients */
} rs2_intrinsics;

/* Given a point in 3D space, compute the corresponding pixel coordinates in an image with no distortion or forward distortion coefficients produced by the same camera */
static void rs2_project_point_to_pixel(float pixel[2], const struct rs2_intrinsics* intrin, const float point[3])
{
	float x = point[0] / point[2], y = point[1] / point[2];

	if ((intrin->model == RS2_DISTORTION_MODIFIED_BROWN_CONRADY) ||
		(intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY))
	{

		float r2 = x * x + y * y;
		float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2 * r2 + intrin->coeffs[4] * r2 * r2 * r2;
		x *= f;
		y *= f;
		float dx = x + 2 * intrin->coeffs[2] * x * y + intrin->coeffs[3] * (r2 + 2 * x * x);
		float dy = y + 2 * intrin->coeffs[3] * x * y + intrin->coeffs[2] * (r2 + 2 * y * y);
		x = dx;
		y = dy;
	}
	if (intrin->model == RS2_DISTORTION_FTHETA)
	{
		float r = sqrtf(x * x + y * y);
		if (r < FLT_EPSILON)
		{
			r = FLT_EPSILON;
		}
		float rd = (float)(1.0f / intrin->coeffs[0] * atan(2 * r * tan(intrin->coeffs[0] / 2.0f)));
		x *= rd / r;
		y *= rd / r;
	}
	if (intrin->model == RS2_DISTORTION_KANNALA_BRANDT4)
	{
		float r = sqrtf(x * x + y * y);
		if (r < FLT_EPSILON)
		{
			r = FLT_EPSILON;
		}
		float theta = atan(r);
		float theta2 = theta * theta;
		float series = 1 + theta2 * (intrin->coeffs[0] + theta2 * (intrin->coeffs[1] + theta2 * (intrin->coeffs[2] + theta2 * intrin->coeffs[3])));
		float rd = theta * series;
		x *= rd / r;
		y *= rd / r;
	}

	pixel[0] = x * intrin->fx + intrin->ppx;
	pixel[1] = y * intrin->fy + intrin->ppy;
}

/* Given pixel coordinates and depth in an image with no distortion or inverse distortion coefficients, compute the corresponding point in 3D space relative to the same camera */
static void rs2_deproject_pixel_to_point(float point[3], const struct rs2_intrinsics* intrin, const float pixel[2], float depth)
{
	assert(intrin->model != RS2_DISTORTION_MODIFIED_BROWN_CONRADY); // Cannot deproject from a forward-distorted image
	//assert(intrin->model != RS2_DISTORTION_BROWN_CONRADY); // Cannot deproject to an brown conrady model

	float x = (pixel[0] - intrin->ppx) / intrin->fx;
	float y = (pixel[1] - intrin->ppy) / intrin->fy;
	if (intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY)
	{
		float r2 = x * x + y * y;
		float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2 * r2 + intrin->coeffs[4] * r2 * r2 * r2;
		float ux = x * f + 2 * intrin->coeffs[2] * x * y + intrin->coeffs[3] * (r2 + 2 * x * x);
		float uy = y * f + 2 * intrin->coeffs[3] * x * y + intrin->coeffs[2] * (r2 + 2 * y * y);
		x = ux;
		y = uy;
	}
	if (intrin->model == RS2_DISTORTION_KANNALA_BRANDT4)
	{
		float rd = sqrtf(x * x + y * y);
		if (rd < FLT_EPSILON)
		{
			rd = FLT_EPSILON;
		}

		float theta = rd;
		float theta2 = rd * rd;
		for (int i = 0; i < 4; i++)
		{
			float f = theta * (1 + theta2 * (intrin->coeffs[0] + theta2 * (intrin->coeffs[1] + theta2 * (intrin->coeffs[2] + theta2 * intrin->coeffs[3])))) - rd;
			if (abs(f) < FLT_EPSILON)
			{
				break;
			}
			float df = 1 + theta2 * (3 * intrin->coeffs[0] + theta2 * (5 * intrin->coeffs[1] + theta2 * (7 * intrin->coeffs[2] + 9 * theta2 * intrin->coeffs[3])));
			theta -= f / df;
			theta2 = theta * theta;
		}
		float r = tan(theta);
		x *= r / rd;
		y *= r / rd;
	}
	if (intrin->model == RS2_DISTORTION_FTHETA)
	{
		float rd = sqrtf(x * x + y * y);
		if (rd < FLT_EPSILON)
		{
			rd = FLT_EPSILON;
		}
		float r = (float)(tan(intrin->coeffs[0] * rd) / atan(2 * tan(intrin->coeffs[0] / 2.0f)));
		x *= r / rd;
		y *= r / rd;
	}

	point[0] = depth * x;
	point[1] = depth * y;
	point[2] = depth;
}

#define MCMF 0
#define OA 1
#define UV 2
#define RSE 3
#define APORA 4
#define Kr 5
#define NBVNET 6
#define SCVP 7
#define Random 8
#define PCNBV 9
#define GMC 10

class Share_Data
{
public:
	//�ɱ��������
	string pcd_file_path;
	string yaml_file_path;
	string name_of_pcd;
	string nbv_net_path;
	string pcnbv_path;
	string sc_net_path;

	int num_of_views;					//һ�β����ӵ����
	double cost_weight;
	rs2_intrinsics color_intrinsics;
	double depth_scale;

	//���в���
	int process_cnt;					//���̱��
	atomic<double> pre_clock;			//ϵͳʱ��
	atomic<bool> over;					//�����Ƿ����
	bool show;
	bool is_save;
	int num_of_max_iteration;

	//��������
	atomic<int> vaild_clouds;
	vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr > clouds;							//������
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pcd;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ground_truth;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_final;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_scene;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_now;
	bool move_wait;
	map<string, double> mp_scale;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ground_truth_downsampled;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_final_downsampled;

	//�˲��ͼ
	octomap::ColorOcTree* octo_model;
	octomap::ColorOcTree* cloud_model;
	octomap::ColorOcTree* ground_truth_model;
	octomap::ColorOcTree* GT_sample;
	double octomap_resolution;
	double ground_truth_resolution;
	double p_unknown_upper_bound; //! Upper bound for voxels to still be considered uncertain. Default: 0.97.
	double p_unknown_lower_bound; //! Lower bound for voxels to still be considered uncertain. Default: 0.12.
	
	//�����ռ����ӵ�ռ�
	atomic<bool> now_view_space_processed;
	atomic<bool> now_views_infromation_processed;
	atomic<bool> move_on;

	Eigen::Matrix4d now_camera_pose_world;
	Eigen::Vector3d object_center_world;		//��������
	double predicted_size;						//����BBX�뾶

	int method_of_IG;
	bool MA_SCVP_on;
	bool Combined_on;
	int num_of_nbvs_combined;
	pcl::visualization::PCLVisualizer::Ptr viewer;

	double stop_thresh_map;
	double stop_thresh_view;

	double skip_coefficient;

	double sum_local_information;
	double sum_global_information;

	double sum_robot_cost;
	double move_weight;
	bool robot_cost_negtive;
	bool move_cost_on;

	int num_of_max_flow_node;
	double interesting_threshold;

	double skip_threshold;
	double visble_rate;
	double move_rate;

	unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel_gt;
	unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel_gt_sample;
	unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel_final;

	int init_voxels;     //����voxel����
	int voxels_in_BBX;   //��ͼvoxel����
	double init_entropy; //��ͼ��Ϣ��
	double movement_cost = 0; //���˶�����

	string pre_path;
	string save_path;

	int rotate_state = 0; //��ת�ı���
	int first_view_id = 0; //��ʼ�ӵ���
	vector<int> view_label_id;
	double min_z_table;

	int GT_points_number = -1; //Ԥ�Ȼ�ȡ�����пɼ����صĸ���
	int cloud_points_number; //����GT�еĵ���

	bool final_viewer = false; //是否显示最终的点云

	bool show_bbx = false; //是否显示bbx

	int num_of_max_thread;
	double height_of_ground;
	double camera_to_object_dis;
	double move_dis_pre_point;
	vector<vector<float>> waypoints;
	double up_shift;
	bool gt_mode;
	Eigen::Matrix4d camera_to_endeffector;
	Eigen::Matrix4d camera_depth_to_rgb;
	double icp_distance;
	double height_to_filter_arm;
	double size_scale;
	double safe_distance;

	double table_x_min;
	double table_x_max;
	double table_y_min;
	double table_y_max;

	vector<int> f_voxels; //frontier voxels的数量

	int f_stop_iter = -1; //根据f_voxels停止的迭代轮次
	double f_stop_threshold = -1; //根据f_voxels停止的阈值

	int f_stop_iter_lenient = -1; //根据f_voxels停止的迭代轮次
	double f_stop_threshold_lenient = -1; //根据f_voxels停止的阈值

	int mascvp_nbv_needed_views = -1; //MA_SCVP+1NBV需要的视点数

	int object_center_mode = 0; // 0:Online_Dynamic, 1:Prior_GT

	vector<Eigen::Matrix4d> nbvs_pose_world; //所有视点的位姿

	double map_scale = 1.2; // 扩大的地图，用于避免重复更新

	Eigen::Vector3d map_center; // 地图的中心
	double map_size; // 地图的大小

	int vsc_nabourhood_num = 1; // 用于计算vsc的邻域数量

	int test_index = -1; // 用于测试的index

	Share_Data(string _config_file_path, string test_name = "", int test_rotate = -1, int test_view = -1, int test_method = -1, int move_test_on = -1, int combined_test_on = -1)
	{
		process_cnt = -1;
		yaml_file_path = _config_file_path;
		//��ȡyaml�ļ�
		cv::FileStorage fs;
		fs.open(yaml_file_path, cv::FileStorage::READ);
		fs["pre_path"] >> pre_path;
		fs["model_path"] >> pcd_file_path;
		fs["name_of_pcd"] >> name_of_pcd;
		fs["method_of_IG"] >> method_of_IG;
		fs["MA_SCVP_on"] >> MA_SCVP_on;
		fs["Combined_on"] >> Combined_on;
		fs["num_of_nbvs_combined"] >> num_of_nbvs_combined;
		fs["octomap_resolution"] >> octomap_resolution;
		fs["ground_truth_resolution"] >> ground_truth_resolution;
		fs["num_of_max_iteration"] >> num_of_max_iteration;
		fs["test_index"] >> test_index;
		fs["first_view_id"] >> first_view_id;
		fs["num_of_max_thread"] >> num_of_max_thread;
		fs["height_of_ground"] >> height_of_ground;
		fs["min_z_table"] >> min_z_table;
		fs["camera_to_object_dis"] >> camera_to_object_dis;
		fs["move_dis_pre_point"] >> move_dis_pre_point;
		fs["up_shift"] >> up_shift;
		fs["gt_mode"] >> gt_mode;
		fs["show"] >> show;
		fs["is_save"] >> is_save;
		fs["icp_distance"] >> icp_distance;
		fs["height_to_filter_arm"] >> height_to_filter_arm;
		fs["size_scale"] >> size_scale;
		fs["safe_distance"] >> safe_distance;
		fs["f_stop_threshold"] >> f_stop_threshold;
		fs["mascvp_nbv_needed_views"] >> mascvp_nbv_needed_views;
		fs["object_center_mode"] >> object_center_mode;
		fs["map_scale"] >> map_scale;
		fs["vsc_nabourhood_num"] >> vsc_nabourhood_num;
		fs["move_wait"] >> move_wait;
		fs["final_viewer"] >> final_viewer;
		fs["show_bbx"] >> show_bbx;
		fs["nbv_net_path"] >> nbv_net_path;
		fs["pcnbv_path"] >> pcnbv_path;
		fs["sc_net_path"] >> sc_net_path;
		fs["p_unknown_upper_bound"] >> p_unknown_upper_bound;
		fs["p_unknown_lower_bound"] >> p_unknown_lower_bound;
		fs["num_of_views"] >> num_of_views;
		fs["move_cost_on"] >> move_cost_on;
		fs["move_weight"] >> move_weight;
		fs["cost_weight"] >> cost_weight;
		fs["robot_cost_negtive"] >> robot_cost_negtive;
		fs["skip_coefficient"] >> skip_coefficient;
		fs["num_of_max_flow_node"] >> num_of_max_flow_node;
		fs["interesting_threshold"] >> interesting_threshold;
		fs["skip_threshold"] >> skip_threshold;
		fs["visble_rate"] >> visble_rate;
		fs["move_rate"] >> move_rate;
		fs["color_width"] >> color_intrinsics.width;
		fs["color_height"] >> color_intrinsics.height;
		fs["color_fx"] >> color_intrinsics.fx;
		fs["color_fy"] >> color_intrinsics.fy;
		fs["color_ppx"] >> color_intrinsics.ppx;
		fs["color_ppy"] >> color_intrinsics.ppy;
		fs["color_model"] >> color_intrinsics.model;
		fs["color_k1"] >> color_intrinsics.coeffs[0];
		fs["color_k2"] >> color_intrinsics.coeffs[1];
		fs["color_k3"] >> color_intrinsics.coeffs[2];
		fs["color_p1"] >> color_intrinsics.coeffs[3];
		fs["color_p2"] >> color_intrinsics.coeffs[4];
		fs["depth_scale"] >> depth_scale;
		fs["table_x_min"] >> table_x_min;
		fs["table_x_max"] >> table_x_max;
		fs["table_y_min"] >> table_y_min;
		fs["table_y_max"] >> table_y_max;
		fs.release();
		//�Զ�������
		if (test_name != "") name_of_pcd = test_name;
		if (test_rotate != -1) rotate_state = test_rotate;
		if (test_view != -1) first_view_id = test_view;
		if (test_method != -1) method_of_IG = test_method;
		if (move_test_on != -1) move_cost_on = move_test_on;
		if (combined_test_on != -1) Combined_on = combined_test_on;
		if (method_of_IG == 7 || Combined_on == true) num_of_max_iteration = 32; //SCVPϵ�в��������ֵ
		if (Combined_on == false) num_of_nbvs_combined = 0; //���ۺ�ϵͳ�����ó�ʼNBV����ƫ��(����ָʾMASCVP)
		/* //����ؽ�����ΪĬ��20
		if (method_of_IG != 7) {
			string path_scvp = pre_path + name_of_pcd + "_r" + to_string(rotate_state) + "_v" + to_string(first_view_id) + "_m7";
			ifstream fin_all_needed_views;
			fin_all_needed_views.open(path_scvp + "/all_needed_views.txt");
			if (fin_all_needed_views.is_open()) {
				int all_needed_views;
				fin_all_needed_views >> all_needed_views;
				num_of_max_iteration = all_needed_views;
				cout << "max iteration set to " << num_of_max_iteration << endl;
			}
			else {
				cout << "max iteration is default " << num_of_max_iteration << endl;
			}
		}
		*/

		
		// camrea_to_endeffector
        double c_to_e_x = 0.0;
        double c_to_e_y = 0.0;
        double c_to_e_z = 0.0;
        double c_to_e_qx = -0.002;
        double c_to_e_qy = 0.001;
        double c_to_e_qz = -0.707;
        double c_to_e_qw = 0.707;
        Eigen::Quaterniond c_to_e_q(c_to_e_qw, c_to_e_qx, c_to_e_qy, c_to_e_qz);
        Eigen::Matrix3d c_to_e_rotation = c_to_e_q.toRotationMatrix();
        camera_to_endeffector = Eigen::Matrix4d::Identity();
        camera_to_endeffector.block<3,3>(0, 0) = c_to_e_rotation;
        camera_to_endeffector(0, 3) = c_to_e_x;
        camera_to_endeffector(1, 3) = c_to_e_y;
        camera_to_endeffector(2, 3) = c_to_e_z;

		camera_depth_to_rgb(0, 0) = 0; camera_depth_to_rgb(0, 1) = 1; camera_depth_to_rgb(0, 2) = 0; camera_depth_to_rgb(0, 3) = 0;
		camera_depth_to_rgb(1, 0) = 0; camera_depth_to_rgb(1, 1) = 0; camera_depth_to_rgb(1, 2) = 1; camera_depth_to_rgb(1, 3) = 0;
		camera_depth_to_rgb(2, 0) = 1; camera_depth_to_rgb(2, 1) = 0; camera_depth_to_rgb(2, 2) = 0; camera_depth_to_rgb(2, 3) = 0;
		camera_depth_to_rgb(3, 0) = 0;	camera_depth_to_rgb(3, 1) = 0;	camera_depth_to_rgb(3, 2) = 0;	camera_depth_to_rgb(3, 3) = 1;

		//cout << "camera_to_endeffector: " << camera_to_endeffector << endl;
		//cout<<"camera_depth_to_rgb: "<<camera_depth_to_rgb<<endl;

		//读取pcd文件
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_pcd(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud_ground_truth = temp_pcd;
		if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(pcd_file_path + name_of_pcd + ".pcd", *cloud_ground_truth) == -1) {
			cout << "Can not read 3d model file. Check." << endl;
		}
		cout<< "cloud_ground_truth->points.size() is "<<cloud_ground_truth->points.size()<<endl;

		//��GT
		ifstream fin_GT_points_number;
		fin_GT_points_number.open(pcd_file_path + name_of_pcd + ".txt");
		if (fin_GT_points_number.is_open()) {
			int points_number;
			fin_GT_points_number >> points_number;
			GT_points_number = points_number;
			cout << "GT_points_number is " << GT_points_number << endl;
		}
		else {
			cout << "no GT_points_number, run mode 3 first. This process will continue without GT_points_number." << endl;
		}

		//octo_model = new octomap::ColorOcTree(octomap_resolution);
		//octo_model->setProbHit(0.95);	//���ô�����������,��ʼ0.7
		//octo_model->setProbMiss(0.05);	//���ô�����ʧ���ʣ���ʼ0.4
		//octo_model->setClampingThresMax(1.0);	//���õ�ͼ�ڵ����ֵ����ʼ0.971
		//octo_model->setClampingThresMin(0.0);	//���õ�ͼ�ڵ���Сֵ����ʼ0.1192
		//octo_model->setOccupancyThres(0.65);	//���ýڵ�ռ����ֵ����ʼ0.5
		ground_truth_model = new octomap::ColorOcTree(ground_truth_resolution);
		//ground_truth_model->setProbHit(0.95);	//���ô�����������,��ʼ0.7
		//ground_truth_model->setProbMiss(0.05);	//���ô�����ʧ���ʣ���ʼ0.4
		//ground_truth_model->setClampingThresMax(1.0);	//���õ�ͼ�ڵ����ֵ����ʼ0.971
		//ground_truth_model->setClampingThresMin(0.0);	//���õ�ͼ�ڵ���Сֵ����ʼ0.1192
		GT_sample = new octomap::ColorOcTree(ground_truth_resolution * 2);
		//GT_sample->setProbHit(0.95);	//���ô�����������,��ʼ0.7
		//GT_sample->setProbMiss(0.05);	//���ô�����ʧ���ʣ���ʼ0.4
		//GT_sample->setClampingThresMax(1.0);	//���õ�ͼ�ڵ����ֵ����ʼ0.971
		//GT_sample->setClampingThresMin(0.0);	//���õ�ͼ�ڵ���Сֵ����ʼ0.1192
		cloud_model = new octomap::ColorOcTree(ground_truth_resolution);
		//cloud_model->setProbHit(0.95);	//���ô�����������,��ʼ0.7
		//cloud_model->setProbMiss(0.05);	//���ô�����ʧ���ʣ���ʼ0.4
		//cloud_model->setClampingThresMax(1.0);	//���õ�ͼ�ڵ����ֵ����ʼ0.971
		//cloud_model->setClampingThresMin(0.0);	//���õ�ͼ�ڵ���Сֵ����ʼ0.1192
		//cloud_model->setOccupancyThres(0.65);	//���ýڵ�ռ����ֵ����ʼ0.5*/
		if (num_of_max_flow_node == -1) num_of_max_flow_node = num_of_views;
		now_camera_pose_world = Eigen::Matrix4d::Identity(4, 4);
		over = false;
		pre_clock = clock();
		vaild_clouds = 0;
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud_final = temp;
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_scene(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud_scene = temp_scene;
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_now(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud_now = temp_now;
		voxel_gt = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
		voxel_gt_sample = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
		voxel_final = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
		access_directory(pre_path);
		save_path = pre_path + name_of_pcd + "_v" + to_string(first_view_id) + "_m" + to_string(method_of_IG);
		//save_path = pre_path + name_of_pcd + "_r" + to_string(rotate_state) + "_v" + to_string(first_view_id) + "_m" + to_string(method_of_IG);
		if (method_of_IG==7 && MA_SCVP_on == false) save_path += "_single";
		if (move_cost_on == true) save_path += '_' + to_string(move_weight);
		if (Combined_on == true) save_path += "_combined_" + to_string(num_of_nbvs_combined);
		//if (method_of_IG == 10) save_path += "_mov";
		if (object_center_mode == 1) save_path += "_gtcenter";

		if (test_index >=0) save_path += "_t" + to_string(test_index);

		cout << "pcd and yaml files readed." << endl;
		cout << "save_path is: " << save_path << endl;

		f_stop_threshold_lenient = f_stop_threshold * 5;

		if (!(Combined_on == true || method_of_IG == 7)) { // 随机方法需要对比的数量
			string test_path_fix = "";
			if (test_index >=0) test_path_fix = "_t" + to_string(test_index);

			ifstream fin_mascvp_nbv_needed_views(pre_path + name_of_pcd + "_v" + to_string(first_view_id) + "_m9_combined_1"+ test_path_fix +"/all_needed_views.txt");
			if (!fin_mascvp_nbv_needed_views) {
				cout << "no all_needed_views from mascvp+1nbv. run mascvp+1nbv first." << endl;
			}
			else {
				fin_mascvp_nbv_needed_views >> mascvp_nbv_needed_views;
				cout << "mascvp_nbv_needed_views num is " << mascvp_nbv_needed_views << endl;
			}
		}

		if (object_center_mode == 1 || Combined_on == true) { // gt模式或者combined模式（即我们的方法），无需放大地图
			map_scale = 1.0;
		}

		srand(clock());
	}

	~Share_Data() {
		delete octo_model;
		delete ground_truth_model;
		delete cloud_model;
		delete voxel_gt;
		delete voxel_gt_sample;
		delete voxel_final;
		cloud_pcd->~PointCloud();
		cloud_final->~PointCloud();
		cloud_ground_truth->~PointCloud();
		for (int i = 0; i < clouds.size(); i++)
			clouds[i]->~PointCloud();
		if (show) viewer->~PCLVisualizer();
	}

	Eigen::Matrix4d get_toward_pose(int toward_state)
	{
		Eigen::Matrix4d pose(Eigen::Matrix4d::Identity(4, 4));
		switch (toward_state) {
		case 0://z<->z
			return pose;
		case 1://z<->-z
			pose(0, 0) = 1; pose(0, 1) = 0; pose(0, 2) = 0; pose(0, 3) = 0;
			pose(1, 0) = 0; pose(1, 1) = 1; pose(1, 2) = 0; pose(1, 3) = 0;
			pose(2, 0) = 0; pose(2, 1) = 0; pose(2, 2) = -1; pose(2, 3) = 0;
			pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
			return pose;
		case 2://z<->x
			pose(0, 0) = 0; pose(0, 1) = 0; pose(0, 2) = 1; pose(0, 3) = 0;
			pose(1, 0) = 0; pose(1, 1) = 1; pose(1, 2) = 0; pose(1, 3) = 0;
			pose(2, 0) = 1; pose(2, 1) = 0; pose(2, 2) = 0; pose(2, 3) = 0;
			pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
			return pose;
		case 3://z<->-x
			pose(0, 0) = 0; pose(0, 1) = 0; pose(0, 2) = 1; pose(0, 3) = 0;
			pose(1, 0) = 0; pose(1, 1) = 1; pose(1, 2) = 0; pose(1, 3) = 0;
			pose(2, 0) = -1; pose(2, 1) = 0; pose(2, 2) = 0; pose(2, 3) = 0;
			pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
			return pose;
		case 4://z<->y
			pose(0, 0) = 1; pose(0, 1) = 0; pose(0, 2) = 0; pose(0, 3) = 0;
			pose(1, 0) = 0; pose(1, 1) = 0; pose(1, 2) = 1; pose(1, 3) = 0;
			pose(2, 0) = 0; pose(2, 1) = 1; pose(2, 2) = 0; pose(2, 3) = 0;
			pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
			return pose;
		case 5://z<->-y
			pose(0, 0) = 1; pose(0, 1) = 0; pose(0, 2) = 0; pose(0, 3) = 0;
			pose(1, 0) = 0; pose(1, 1) = 0; pose(1, 2) = 1; pose(1, 3) = 0;
			pose(2, 0) = 0; pose(2, 1) = -1; pose(2, 2) = 0; pose(2, 3) = 0;
			pose(3, 0) = 0;	pose(3, 1) = 0;	pose(3, 2) = 0;	pose(3, 3) = 1;
			return pose;
		}
		return pose;
	}

	double out_clock()
	{   //������ʱ��������ʱ��
		double now_clock = clock();
		double time_len = now_clock - pre_clock;
		pre_clock = now_clock;
		return time_len;
	}

	void access_directory(string cd)
	{   //���༶Ŀ¼���ļ����Ƿ���ڣ������ھʹ���
		cout << cd << endl;
		string temp;
		for (int i = 0; i < cd.length(); i++)
			if (cd[i] == '/') {
				if (access(temp.c_str(), F_OK) == -1) mkdir(temp.c_str(), 0777);
				temp += cd[i];
			}
			else temp += cd[i];
		if (access(temp.c_str(), F_OK) == -1) mkdir(temp.c_str(), 0777);
	}

	void save_posetrans_to_disk(Eigen::Matrix4d& T, string cd, string name, int frames_cnt)
	{   //�����ת��������������
		std::stringstream pose_stream, path_stream;
		std::string pose_file, path;
		path_stream << "../data" << "_" << process_cnt << cd;
		path_stream >> path;
		access_directory(path);
		pose_stream << "../data" << "_" << process_cnt << cd << "/" << name << "_" << frames_cnt << ".txt";
		pose_stream >> pose_file;
		ofstream fout(pose_file);
		fout << T;
	}

	void save_octomap_log_to_disk(int voxels, double entropy, string cd, string name,int iterations)
	{
		std::stringstream log_stream, path_stream;
		std::string log_file, path;
		path_stream << "../data" << "_" << process_cnt << cd;
		path_stream >> path;
		access_directory(path);
		log_stream << "../data" << "_" << process_cnt << cd << "/" << name << "_" << iterations << ".txt";
		log_stream >> log_file;
		ofstream fout(log_file);
		fout << voxels << " " << entropy << endl;
	}

	void save_cloud_to_disk(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, string cd, string name)
	{   //��ŵ������������̣��ٶȺ���������
		std::stringstream cloud_stream, path_stream;
		std::string cloud_file, path;
		path_stream << save_path << cd;
		path_stream >> path;
		access_directory(path);
		cloud_stream << save_path << cd << "/" << name << ".pcd";
		cloud_stream >> cloud_file;
		//pcl::io::savePCDFile<pcl::PointXYZRGB>(cloud_file, *cloud);
		pcl::io::savePCDFileBinary<pcl::PointXYZRGB>(cloud_file, *cloud);
	}

	void save_cloud_to_disk(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, string cd, string name, int frames_cnt)
	{   //��ŵ������������̣��ٶȺ���������
		std::stringstream cloud_stream, path_stream;
		std::string cloud_file, path;
		path_stream << "../data" << "_" << process_cnt << cd;
		path_stream >> path;
		access_directory(path);
		cloud_stream << "../data" << "_" << process_cnt << cd << "/" << name << "_" << frames_cnt << ".pcd";
		cloud_stream >> cloud_file;
		cout<<cloud_file<<endl;
		//pcl::io::savePCDFile<pcl::PointXYZRGB>(cloud_file, *cloud);
		pcl::io::savePCDFileBinary<pcl::PointXYZRGB>(cloud_file, *cloud);
	}

	void save_octomap_to_disk(octomap::ColorOcTree* octo_model, string cd, string name)
	{   //��ŵ������������̣��ٶȺ���������
		std::stringstream octomap_stream, path_stream;
		std::string octomap_file, path;
		path_stream << "../data" << "_" << process_cnt << cd;
		path_stream >> path;
		access_directory(path);
		octomap_stream << "../data" << "_" << process_cnt << cd << "/" << name << ".ot";
		octomap_stream >> octomap_file;
		octo_model->write(octomap_file);
	}

};

inline double pow2(double x) {
	return x * x;
}