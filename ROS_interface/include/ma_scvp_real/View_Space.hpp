#pragma once
#include <iostream> 
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <time.h>
#include <mutex>
#include <unordered_set>
#include <bitset>

#include <opencv2/opencv.hpp>

#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>

#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace std;

class Voxel_Information
{
public:
	double p_unknown_upper_bound;
	double p_unknown_lower_bound;
	double k_vis;
	double b_vis;
	mutex mutex_rays;
	vector<mutex*> mutex_voxels;
	vector<mutex*> mutex_views;
	vector<Eigen::Vector4d> convex;
	double skip_coefficient;
	double octomap_resolution;

	Voxel_Information(double _p_unknown_lower_bound, double _p_unknown_upper_bound) {
		p_unknown_upper_bound = _p_unknown_upper_bound;
		p_unknown_lower_bound = _p_unknown_lower_bound;
		k_vis = (0.0 - 1.0) / (p_unknown_upper_bound - p_unknown_lower_bound);
		b_vis = -k_vis * p_unknown_upper_bound;
	}

	~Voxel_Information() {
		for (int i = 0; i < mutex_voxels.size(); i++)
			delete mutex_voxels[i];
		for (int i = 0; i < mutex_views.size(); i++)
			delete mutex_views[i];
	}

	void clear_mutex_voxels(){
		for (int i = 0; i < mutex_voxels.size(); i++)
			delete mutex_voxels[i];
		mutex_voxels.clear();
	}

	void init_mutex_voxels(int init_voxels) {
		mutex_voxels.resize(init_voxels);
		for (int i = 0; i < mutex_voxels.size(); i++)
			mutex_voxels[i] = new mutex;
	}

	void init_mutex_views(int init_views) {
		mutex_views.resize(init_views);
		for (int i = 0; i < mutex_views.size(); i++)
			mutex_views[i] = new mutex;
	}

	double entropy(double& occupancy) {
		double p_free = 1 - occupancy;
		if (occupancy == 0 || p_free == 0)	return 0;
		double vox_ig = -occupancy * log(occupancy) - p_free * log(p_free);
		return vox_ig;
	}

	bool is_known(double& occupancy) {
		return occupancy >= p_unknown_upper_bound || occupancy <= p_unknown_lower_bound;
	}

	bool is_unknown(double& occupancy) {
		return occupancy < p_unknown_upper_bound && occupancy > p_unknown_lower_bound;
	}

	bool is_free(double& occupancy)
	{
		return occupancy < p_unknown_lower_bound;
	}

	bool is_occupied(double& occupancy)
	{
		return occupancy > p_unknown_upper_bound;
	}

	bool voxel_unknown(octomap::ColorOcTreeNode* traversed_voxel) {
		double occupancy = traversed_voxel->getOccupancy();
		return is_unknown(occupancy);
	}

	bool voxel_free(octomap::ColorOcTreeNode* traversed_voxel) {
		double occupancy = traversed_voxel->getOccupancy();
		return is_free(occupancy);
	}

	bool voxel_occupied(octomap::ColorOcTreeNode* traversed_voxel) {
		double occupancy = traversed_voxel->getOccupancy();
		return is_occupied(occupancy);
	}
	//-5x+3.25
	double get_voxel_visible(double occupancy) {
		if (occupancy > p_unknown_upper_bound) return 0.0;
		if (occupancy < p_unknown_lower_bound) return 1.0;
		return k_vis * occupancy + b_vis;
	}

	double get_voxel_visible(octomap::ColorOcTreeNode* traversed_voxel) {
		double occupancy = traversed_voxel->getOccupancy();
		if (occupancy > p_unknown_upper_bound) return 1.0;
		if (occupancy < p_unknown_lower_bound) return 0.0;
		return k_vis * occupancy + b_vis;
	}

	double get_voxel_information(octomap::ColorOcTreeNode* traversed_voxel){
		double occupancy = traversed_voxel->getOccupancy();
		double information = entropy(occupancy);
		return information;
	}

	double voxel_object(octomap::OcTreeKey& voxel_key, unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash>* object_weight) {
		auto key = object_weight->find(voxel_key);
		if (key == object_weight->end()) return 0;
		return key->second;
	}

	double get_voxel_object_visible(octomap::OcTreeKey& voxel_key, unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash>* object_weight) {
		double object = voxel_object(voxel_key, object_weight);
		double p_vis = 1 - object;
		return p_vis;
	}

};

inline double get_random_coordinate(double from, double to) {
	//���ɱȽ������0-1�������ӳ�䵽����[from,to]
	double len = to - from;
	long long x = (long long)rand() * ((long long)RAND_MAX + 1) + (long long)rand();
	long long field = (long long)RAND_MAX * (long long)RAND_MAX + 2 * (long long)RAND_MAX;
	return (double)x / (double)field * len + from;
}

void add_trajectory_to_cloud(Eigen::Matrix4d now_camera_pose_world, vector<Eigen::Vector3d>& points, pcl::visualization::PCLVisualizer::Ptr viewer) {
	viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(now_camera_pose_world(0, 3), now_camera_pose_world(1, 3), now_camera_pose_world(2, 3)), pcl::PointXYZ(points[0](0), points[0](1), points[0](2)), 255, 255, 0, "trajectory" + to_string(-1));
	for (int i = 0; i < points.size() - 1; i++) {
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(points[i](0), points[i](1), points[i](2)), pcl::PointXYZ(points[i + 1](0), points[i + 1](1), points[i + 1](2)), 255, 255, 0, "trajectory" + to_string(i));
	}
}

void delete_trajectory_in_cloud(int num, pcl::visualization::PCLVisualizer::Ptr viewer) {
	viewer->removeCorrespondences("trajectory" + to_string(-1));
	for (int i = 0; i < num - 1; i++) {
		viewer->removeCorrespondences("trajectory" + to_string(i));
	}
}

class View
{
public:
	int space_id;
	int id;
	Eigen::Vector3d init_pos;	//��ʼλ��
	Eigen::Matrix4d pose;		//view_i��view_i+1��ת����
	double information_gain;
	int voxel_num;
	double robot_cost;
	double dis_to_obejct;
	double final_utility;
	atomic<bool> robot_moved;
	int path_num;
	int vis;
	bool can_move;
	bitset<64> in_coverage;
	bool in_cover;

	View(Eigen::Vector3d _init_pos) {
		init_pos = _init_pos;
		pose = Eigen::Matrix4d::Identity(4, 4);
		information_gain = 0;
		voxel_num = 0;
		robot_cost = 0;
		dis_to_obejct = 0;
		final_utility = 0;
		robot_moved = false;
		path_num = 0;
		vis = 0;
		can_move = true;
	}

	View(const View &other) {
		space_id = other.space_id;
		id = other.id;
		init_pos = other.init_pos;
		pose = other.pose;
		information_gain = (double)other.information_gain;
		voxel_num = (int)other.voxel_num;
		robot_cost = other.robot_cost;
		dis_to_obejct = other.dis_to_obejct;
		final_utility = other.final_utility;
		robot_moved = (bool)other.robot_moved;
		path_num = other.path_num;
		vis = other.vis;
		can_move = other.can_move;
		in_coverage = other.in_coverage;
	}

	View& operator=(const View& other) {
		init_pos = other.init_pos;
		space_id = other.space_id;
		id = other.id;
		pose = other.pose;
		information_gain = (double)other.information_gain;
		voxel_num = (int)other.voxel_num;
		robot_cost = other.robot_cost;
		dis_to_obejct = other.dis_to_obejct;
		final_utility = other.final_utility;
		robot_moved = (bool)other.robot_moved;
		path_num = other.path_num;
		vis = other.vis;
		can_move = other.can_move;
		in_coverage = other.in_coverage;
		return *this;
	}

	~View() {
		;
	}

	double global_function(int x) {
		return exp(-1.0*x);
	}

	double get_global_information() {
		double information = 0;
		for (int i = 0; i <= space_id && i < 64; i++) //space_id
			information += in_coverage[i] * global_function(space_id - i);
		return information;
	}

	// smallest difference to pervious view
    void get_next_camera_pos(Eigen::Matrix4d now_camera_pose_world, Eigen::Vector3d object_center_world) {
        //归一化乘法
        Eigen::Vector4d object_center_now_camera;
        object_center_now_camera = now_camera_pose_world.inverse() * Eigen::Vector4d(object_center_world(0), object_center_world(1), object_center_world(2), 1);
        Eigen::Vector4d view_now_camera;
        view_now_camera = now_camera_pose_world.inverse() * Eigen::Vector4d(init_pos(0), init_pos(1), init_pos(2), 1);
        //定义指向物体为Z+，从上一个相机位置发出射线至当前为X+，计算两个相机坐标系之间的变换矩阵，object与view为上一个相机坐标系下的坐标
        Eigen::Vector3d object(object_center_now_camera(0), object_center_now_camera(1), object_center_now_camera(2));
        Eigen::Vector3d view(view_now_camera(0), view_now_camera(1), view_now_camera(2));
        Eigen::Vector3d Z;	 Z = object - view;	 Z = Z.normalized();
        //注意左右手系，不要弄反了
        Eigen::Vector3d X;	 X = Z.cross(view);	 X = X.normalized();
        Eigen::Vector3d Y;	 Y = Z.cross(X);	 Y = Y.normalized();
        Eigen::Matrix4d T(4, 4);
        T(0, 0) = 1; T(0, 1) = 0; T(0, 2) = 0; T(0, 3) = -view(0);
        T(1, 0) = 0; T(1, 1) = 1; T(1, 2) = 0; T(1, 3) = -view(1);
        T(2, 0) = 0; T(2, 1) = 0; T(2, 2) = 1; T(2, 3) = -view(2);
        T(3, 0) = 0; T(3, 1) = 0; T(3, 2) = 0; T(3, 3) = 1;
        Eigen::Matrix4d R(4, 4);
        R(0, 0) = X(0); R(0, 1) = Y(0); R(0, 2) = Z(0); R(0, 3) = 0;
        R(1, 0) = X(1); R(1, 1) = Y(1); R(1, 2) = Z(1); R(1, 3) = 0;
        R(2, 0) = X(2); R(2, 1) = Y(2); R(2, 2) = Z(2); R(2, 3) = 0;
        R(3, 0) = 0;	R(3, 1) = 0;	R(3, 2) = 0;	R(3, 3) = 1;
        //绕Z轴旋转，使得与上一次旋转计算x轴与y轴夹角最小
        Eigen::Matrix3d Rz_min(Eigen::Matrix3d::Identity(3, 3));
        Eigen::Vector4d x(1, 0, 0, 1);
        Eigen::Vector4d y(0, 1, 0, 1);
        Eigen::Vector4d x_ray(1, 0, 0, 1);
        Eigen::Vector4d y_ray(0, 1, 0, 1);
        x_ray = R.inverse() * T * x_ray;
        y_ray = R.inverse() * T * y_ray;
        double min_y = acos(y(1) * y_ray(1));
        double min_x = acos(x(0) * x_ray(0));
        for (double i = 5; i < 360; i += 5) {
            Eigen::Matrix3d rotation;
            rotation = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()) *
                Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(i * acos(-1.0) / 180.0, Eigen::Vector3d::UnitZ());
            Eigen::Matrix4d Rz(Eigen::Matrix4d::Identity(4, 4));
            Rz(0, 0) = rotation(0, 0); Rz(0, 1) = rotation(0, 1); Rz(0, 2) = rotation(0, 2); Rz(0, 3) = 0;
            Rz(1, 0) = rotation(1, 0); Rz(1, 1) = rotation(1, 1); Rz(1, 2) = rotation(1, 2); Rz(1, 3) = 0;
            Rz(2, 0) = rotation(2, 0); Rz(2, 1) = rotation(2, 1); Rz(2, 2) = rotation(2, 2); Rz(2, 3) = 0;
            Rz(3, 0) = 0;			   Rz(3, 1) = 0;			  Rz(3, 2) = 0;			     Rz(3, 3) = 1;
            Eigen::Vector4d x_ray(1, 0, 0, 1);
            Eigen::Vector4d y_ray(0, 1, 0, 1);
            x_ray = (R * Rz).inverse() * T * x_ray;
            y_ray = (R * Rz).inverse() * T * y_ray;
            double cos_y = acos(y(1) * y_ray(1));
            double cos_x = acos(x(0) * x_ray(0));
            if (cos_y < min_y) {
                Rz_min = rotation.eval();
                min_y = cos_y;
                min_x = cos_x;
            }
            else if (fabs(cos_y - min_y) < 1e-6 && cos_x < min_x) {
                Rz_min = rotation.eval();
                min_y = cos_y;
                min_x = cos_x;
            }
        }
        Eigen::Vector3d eulerAngle = Rz_min.eulerAngles(0, 1, 2);
        //cout << "Rotate getted with angel " << eulerAngle(0)<<","<< eulerAngle(1) << "," << eulerAngle(2)<<" and l2 "<< min_l2 << endl;
        Eigen::Matrix4d Rz(Eigen::Matrix4d::Identity(4, 4));
        Rz(0, 0) = Rz_min(0, 0); Rz(0, 1) = Rz_min(0, 1); Rz(0, 2) = Rz_min(0, 2); Rz(0, 3) = 0;
        Rz(1, 0) = Rz_min(1, 0); Rz(1, 1) = Rz_min(1, 1); Rz(1, 2) = Rz_min(1, 2); Rz(1, 3) = 0;
        Rz(2, 0) = Rz_min(2, 0); Rz(2, 1) = Rz_min(2, 1); Rz(2, 2) = Rz_min(2, 2); Rz(2, 3) = 0;
        Rz(3, 0) = 0;			 Rz(3, 1) = 0;			  Rz(3, 2) = 0;			   Rz(3, 3) = 1;
        pose = ((R * Rz).inverse() * T).eval();
        //pose = (R.inverse() * T).eval();
    }

    /*
    //y top
    void get_next_camera_pos(Eigen::Matrix4d now_camera_pose_world, Eigen::Vector3d object_center_world) {
        //归一化乘法
        Eigen::Vector4d object_center_now_camera;
        object_center_now_camera = now_camera_pose_world.inverse() * Eigen::Vector4d(object_center_world(0), object_center_world(1), object_center_world(2), 1);
        Eigen::Vector4d view_now_camera;
        view_now_camera = now_camera_pose_world.inverse() * Eigen::Vector4d(init_pos(0), init_pos(1), init_pos(2), 1);
        //定义指向物体为Z+，从上一个相机位置发出射线至当前为X+，计算两个相机坐标系之间的变换矩阵，object与view为上一个相机坐标系下的坐标
        Eigen::Vector3d object(object_center_now_camera(0), object_center_now_camera(1), object_center_now_camera(2));
        Eigen::Vector3d view(view_now_camera(0), view_now_camera(1), view_now_camera(2));
        Eigen::Vector3d Z;	 Z = object - view;	 Z = Z.normalized();
        //注意左右手系，不要弄反了
        Eigen::Vector3d X;	 X = Z.cross(view);	 X = X.normalized();
        Eigen::Vector3d Y;	 Y = Z.cross(X);	 Y = Y.normalized();
        Eigen::Matrix4d T(4, 4);
        T(0, 0) = 1; T(0, 1) = 0; T(0, 2) = 0; T(0, 3) = -view(0);
        T(1, 0) = 0; T(1, 1) = 1; T(1, 2) = 0; T(1, 3) = -view(1);
        T(2, 0) = 0; T(2, 1) = 0; T(2, 2) = 1; T(2, 3) = -view(2);
        T(3, 0) = 0; T(3, 1) = 0; T(3, 2) = 0; T(3, 3) = 1;
        Eigen::Matrix4d R(4, 4);
        R(0, 0) = X(0); R(0, 1) = Y(0); R(0, 2) = Z(0); R(0, 3) = 0;
        R(1, 0) = X(1); R(1, 1) = Y(1); R(1, 2) = Z(1); R(1, 3) = 0;
        R(2, 0) = X(2); R(2, 1) = Y(2); R(2, 2) = Z(2); R(2, 3) = 0;
        R(3, 0) = 0;	R(3, 1) = 0;	R(3, 2) = 0;	R(3, 3) = 1;
        //绕Z轴旋转，使得y轴最高
        Eigen::Matrix3d Rz_min(Eigen::Matrix3d::Identity(3, 3));
        Eigen::Vector4d y_highest = (now_camera_pose_world * R * T * Eigen::Vector4d(0, 1, 0, 1)).eval();
        for (double i = 5; i < 360; i += 5) {
            Eigen::Matrix3d rotation;
            rotation = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()) *
                Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(i * acos(-1.0) / 180.0, Eigen::Vector3d::UnitZ());
            Eigen::Matrix4d Rz(Eigen::Matrix4d::Identity(4, 4));
            Rz(0, 0) = rotation(0, 0); Rz(0, 1) = rotation(0, 1); Rz(0, 2) = rotation(0, 2); Rz(0, 3) = 0;
            Rz(1, 0) = rotation(1, 0); Rz(1, 1) = rotation(1, 1); Rz(1, 2) = rotation(1, 2); Rz(1, 3) = 0;
            Rz(2, 0) = rotation(2, 0); Rz(2, 1) = rotation(2, 1); Rz(2, 2) = rotation(2, 2); Rz(2, 3) = 0;
            Rz(3, 0) = 0;			   Rz(3, 1) = 0;			  Rz(3, 2) = 0;			     Rz(3, 3) = 1;
            Eigen::Vector4d y_now = (now_camera_pose_world * R * Rz * T * Eigen::Vector4d(0, 1, 0, 1)).eval();
            if (y_now(2) > y_highest(2)) {
                Rz_min = rotation.eval();
                y_highest = y_now;
            }
        }
        Eigen::Vector3d eulerAngle = Rz_min.eulerAngles(0, 1, 2);
        //cout << "Rotate getted with angel " << eulerAngle(0)<<","<< eulerAngle(1) << "," << eulerAngle(2)<<" and l2 "<< min_l2 << endl;
        Eigen::Matrix4d Rz(Eigen::Matrix4d::Identity(4, 4));
        Rz(0, 0) = Rz_min(0, 0); Rz(0, 1) = Rz_min(0, 1); Rz(0, 2) = Rz_min(0, 2); Rz(0, 3) = 0;
        Rz(1, 0) = Rz_min(1, 0); Rz(1, 1) = Rz_min(1, 1); Rz(1, 2) = Rz_min(1, 2); Rz(1, 3) = 0;
        Rz(2, 0) = Rz_min(2, 0); Rz(2, 1) = Rz_min(2, 1); Rz(2, 2) = Rz_min(2, 2); Rz(2, 3) = 0;
        Rz(3, 0) = 0;			 Rz(3, 1) = 0;			  Rz(3, 2) = 0;			   Rz(3, 3) = 1;
        pose = ((R * Rz).inverse() * T).eval();
        //pose = (R.inverse() * T).eval();
    }
	*/

	void add_view_coordinates_to_cloud(Eigen::Matrix4d now_camera_pose_world, pcl::visualization::PCLVisualizer::Ptr viewer,int space_id) {
		//view.get_next_camera_pos(view_space->now_camera_pose_world, view_space->object_center_world);
		Eigen::Vector4d X(0.05, 0, 0, 1);
		Eigen::Vector4d Y(0, 0.05, 0, 1);
		Eigen::Vector4d Z(0, 0, 0.05, 1);
		Eigen::Vector4d weight(final_utility,final_utility, final_utility, 1);
		X = now_camera_pose_world * X;
		Y = now_camera_pose_world * Y;
		Z = now_camera_pose_world * Z;
		weight = now_camera_pose_world * weight;
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(init_pos(0), init_pos(1), init_pos(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(space_id) + "-" + to_string(id));
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(init_pos(0), init_pos(1), init_pos(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(space_id) + "-" + to_string(id));
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(init_pos(0), init_pos(1), init_pos(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(space_id) + "-" + to_string(id));
		//viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(init_pos(0), init_pos(1), init_pos(2)), pcl::PointXYZ(weight(0), weight(1), weight(2)), 0, 255, 255, "weight" + to_string(space_id) + "-" + to_string(id));
	}

};

bool view_id_compare(View& a, View& b) {
	return a.id < b.id;
}

bool view_utility_compare(View& a, View& b) {
	if(a.final_utility == b.final_utility) return a.robot_cost < b.robot_cost;
	return a.final_utility > b.final_utility;
}

#define ErrorPath -2
#define WrongPath -1
#define LinePath 0
#define CirclePath 1
//return path mode and length from M to N under an circle obstacle with radius r
pair<int, double> get_local_path(Eigen::Vector3d M, Eigen::Vector3d N, Eigen::Vector3d O, double r) {
	double x0, y0, z0, x1, y1, z1, x2, y2, z2, a, b, c, delta, t3, t4, x3, y3, z3, x4, y4, z4;
	x1 = M(0), y1 = M(1), z1 = M(2);
	x2 = N(0), y2 = N(1), z2 = N(2);
	x0 = O(0), y0 = O(1), z0 = O(2);
	//����ֱ��MN����O-r�Ľ���PQ
	a = pow2(x2 - x1) + pow2(y2 - y1) + pow2(z2 - z1);
	b = 2.0 * ((x2 - x1) * (x1 - x0) + (y2 - y1) * (y1 - y0) + (z2 - z1) * (z1 - z0));
	c = pow2(x1 - x0) + pow2(y1 - y0) + pow2(z1 - z0) - pow2(r);
	delta = pow2(b) - 4.0 * a * c;
	//cout << delta << endl;
	if (delta <= 0) {//delta <= 0
		//���û�н������һ�����㣬�Ϳ��Ի�ֱ�߹�ȥ
		double d = (N - M).norm();
		//cout << "d: " << d << endl;
		return make_pair(LinePath, d);
	}
	else {
		//�����Ҫ�������壬������������ж�
		t3 = (-b - sqrt(pow2(b) - 4.0 * a * c)) / (2.0 * a);
		t4 = (-b + sqrt(pow2(b) - 4.0 * a * c)) / (2.0 * a);
		if ((t3 < 0 || t3 > 1) && (t4 < 0 || t4 > 1)) {
			//�������㣬ֱ�ӹ�ȥ
			double d = (N - M).norm();
			//cout << "d: " << d << endl;
			return make_pair(LinePath, d);
		}
		else if ((t3 < 0 || t3 > 1) || (t4 < 0 || t4 > 1)) {
			//�����յ����ϰ�������
			return make_pair(WrongPath, 1e10);
		}
		if (t3 > t4) {
			double temp = t3;
			t3 = t4;
			t4 = temp;
		}
		x3 = (x2 - x1) * t3 + x1;
		y3 = (y2 - y1) * t3 + y1;
		z3 = (z2 - z1) * t3 + z1;
		Eigen::Vector3d P(x3, y3, z3);
		//cout << "P: " << x3 << "," << y3 << "," << z3 << endl;
		x4 = (x2 - x1) * t4 + x1;
		y4 = (y2 - y1) * t4 + y1;
		z4 = (z2 - z1) * t4 + z1;
		Eigen::Vector3d Q(x4, y4, z4);
		//cout << "Q: " << x4 << "," << y4 << "," << z4 << endl;
		//MONƽ�淽��
		double A, B, C, D, X1, X2, Y1, Y2, Z1, Z2;
		X1 = x3 - x0; X2 = x4 - x0;
		Y1 = y3 - y0; Y2 = y4 - y0;
		Z1 = z3 - z0; Z2 = z4 - z0;
		A = Y1 * Z2 - Y2 * Z1;
		B = Z1 * X2 - Z2 * X1;
		C = X1 * Y2 - X2 * Y1;
		D = -A * x0 - B * y0 - C * z0;
		//D = -(x0 * Y1 * Z2 + X1 * Y2 * z0 + X2 * y0 * Z1 - X2 * Y1 * z0 - X1 * y0 * Z2 - x0 * Y2 * Z1);
		//�������������P,Q�Ĳ���ֵ
		double theta3, theta4, flag, MP, QN, L, d;
		double sin_theta3, sin_theta4;
		sin_theta3 = -(z3 - z0) / r * sqrt(pow2(A) + pow2(B) + pow2(C)) / sqrt(pow2(A) + pow2(B));
		theta3 = asin(sin_theta3);
		if (theta3 < 0) theta3 += 2.0 * acos(-1.0);
		if (theta3 >= 2.0 * acos(-1.0)) theta3 -= 2.0 * acos(-1.0);
		double x3_theta3, y3_theta3;
		x3_theta3 = x0 + r * B / sqrt(pow2(A) + pow2(B)) * cos(theta3) + r * A * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta3);
		y3_theta3 = y0 - r * A / sqrt(pow2(A) + pow2(B)) * cos(theta3) + r * B * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta3);
		//cout << x3_theta3 << " " << y3_theta3 << " " << theta3 << endl;
		if (fabs(x3 - x3_theta3) > 1e-6 || fabs(y3 - y3_theta3) > 1e-6) {
			theta3 = acos(-1.0) - theta3;
			if (theta3 < 0) theta3 += 2.0 * acos(-1.0);
			if (theta3 >= 2.0 * acos(-1.0)) theta3 -= 2.0 * acos(-1.0);
		}
		sin_theta4 = -(z4 - z0) / r * sqrt(pow2(A) + pow2(B) + pow2(C)) / sqrt(pow2(A) + pow2(B));
		theta4 = asin(sin_theta4);
		if (theta4 < 0) theta4 += 2.0 * acos(-1.0);
		if (theta4 >= 2.0 * acos(-1.0)) theta4 -= 2.0 * acos(-1.0);
		double x4_theta4, y4_theta4;
		x4_theta4 = x0 + r * B / sqrt(pow2(A) + pow2(B)) * cos(theta4) + r * A * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta4);
		y4_theta4 = y0 - r * A / sqrt(pow2(A) + pow2(B)) * cos(theta4) + r * B * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta4);
		//cout << x4_theta4 << " " << y4_theta4 << " " << theta4 << endl;
		if (fabs(x4 - x4_theta4) > 1e-6 || fabs(y4 - y4_theta4) > 1e-6) {
			theta4 = acos(-1.0) - theta4;
			if (theta4 < 0) theta4 += 2.0 * acos(-1.0);
			if (theta4 >= 2.0 * acos(-1.0)) theta4 -= 2.0 * acos(-1.0);
		}
		//cout << "theta3: " << theta3 << endl;
		//cout << "theta4: " << theta4 << endl;
		if (theta3 < theta4) flag = 1;
		else flag = -1;
		MP = (M - P).norm();
		QN = (Q - N).norm();
		L = fabs(theta3 - theta4) * r;
		//cout << "L: " << L << endl;
		d = MP + L + QN;
		//cout << "d: " << d << endl;
		return make_pair(CirclePath, d);
	}
	//δ������Ϊ
	return make_pair(ErrorPath, 1e10);
}


int get_trajectory_xyz(vector<Eigen::Vector3d>& points, Eigen::Vector3d M, Eigen::Vector3d N, Eigen::Vector3d O, double predicted_size, double distanse_of_pre_move, double camera_to_object_dis) {
    int num_of_path = -1;
    double x0, y0, z0, x1, y1, z1, x2, y2, z2, a, b, c, delta, t3, t4, x3, y3, z3, x4, y4, z4, r;
    x1 = M(0), y1 = M(1), z1 = M(2);
    x2 = N(0), y2 = N(1), z2 = N(2);
    x0 = O(0), y0 = O(1), z0 = O(2);
	//不考虑相机深度距离
    //r = predicted_size * 1.1;
	//考虑相机深度距离
	r = predicted_size + camera_to_object_dis;
    //计算直线MN与球O-r的交点PQ
    a = pow2(x2 - x1) + pow2(y2 - y1) + pow2(z2 - z1);
    b = 2.0 * ((x2 - x1) * (x1 - x0) + (y2 - y1) * (y1 - y0) + (z2 - z1) * (z1 - z0));
    c = pow2(x1 - x0) + pow2(y1 - y0) + pow2(z1 - z0) - pow2(r);
    delta = pow2(b) - 4.0 * a * c;
    //cout << delta << endl;
    if (delta <= 0) {//delta <= 0
        //如果没有交点或者一个交点，就可以画直线过去
        double d = (N - M).norm();
        //cout << "d: " << d << endl;
        num_of_path = (int)(d / distanse_of_pre_move) + 1;
        //cout << "num_of_path: " << num_of_path << endl;
        double t_pre_point = d / num_of_path;
        for (int i = 1; i <= num_of_path; i++) {
            double di = t_pre_point * i;
            //cout << "di: " << di << endl;
            double xi, yi, zi;
            xi = (x2 - x1) * (di / d) + x1;
            yi = (y2 - y1) * (di / d) + y1;
            zi = (z2 - z1) * (di / d) + z1;
            points.push_back(Eigen::Vector3d(xi, yi, zi));
        }
        return -2;
    }
    else {
        //如果需要穿过球体，则沿着球表面行动
        t3 = (-b - sqrt(pow2(b) - 4.0 * a * c)) / (2.0 * a);
        t4 = (-b + sqrt(pow2(b) - 4.0 * a * c)) / (2.0 * a);
        if ((t3 < 0 || t3 > 1) && (t4 < 0 || t4 > 1)) {
            //球外两点，直接过去
            double d = (N - M).norm();
            //cout << "d: " << d << endl;
            num_of_path = (int)(d / distanse_of_pre_move) + 1;
            //cout << "num_of_path: " << num_of_path << endl;
            double t_pre_point = d / num_of_path;
            for (int i = 1; i <= num_of_path; i++) {
                double di = t_pre_point * i;
                //cout << "di: " << di << endl;
                double xi, yi, zi;
                xi = (x2 - x1) * (di / d) + x1;
                yi = (y2 - y1) * (di / d) + y1;
                zi = (z2 - z1) * (di / d) + z1;
                points.push_back(Eigen::Vector3d(xi, yi, zi));
            }
            return num_of_path;
        }
        else if ((t3 < 0 || t3 > 1) || (t4 < 0 || t4 > 1)) {
            cout << "has one viewport in circle. check?" << endl;
            return -1;
        }
        if (t3 > t4) {
            double temp = t3;
            t3 = t4;
            t4 = temp;
        }
        x3 = (x2 - x1) * t3 + x1;
        y3 = (y2 - y1) * t3 + y1;
        z3 = (z2 - z1) * t3 + z1;
        Eigen::Vector3d P(x3, y3, z3);
        //cout << "P: " << x3 << "," << y3 << "," << z3 << endl;
        x4 = (x2 - x1) * t4 + x1;
        y4 = (y2 - y1) * t4 + y1;
        z4 = (z2 - z1) * t4 + z1;
        Eigen::Vector3d Q(x4, y4, z4);
        //cout << "Q: " << x4 << "," << y4 << "," << z4 << endl;
        //MON平面方程
        double A, B, C, D, X1, X2, Y1, Y2, Z1, Z2;
        X1 = x3 - x0; X2 = x4 - x0;
        Y1 = y3 - y0; Y2 = y4 - y0;
        Z1 = z3 - z0; Z2 = z4 - z0;
        A = Y1 * Z2 - Y2 * Z1;
        B = Z1 * X2 - Z2 * X1;
        C = X1 * Y2 - X2 * Y1;
        D = -A * x0 - B * y0 - C * z0;
        //D = -(x0 * Y1 * Z2 + X1 * Y2 * z0 + X2 * y0 * Z1 - X2 * Y1 * z0 - X1 * y0 * Z2 - x0 * Y2 * Z1);
        //计算参数方程中P,Q的参数值
        double theta3, theta4, flag, MP, QN, L, d;
        double sin_theta3, sin_theta4;
        sin_theta3 = -(z3 - z0) / r * sqrt(pow2(A) + pow2(B) + pow2(C)) / sqrt(pow2(A) + pow2(B));
        theta3 = asin(sin_theta3);
        if (theta3 < 0) theta3 += 2.0 * acos(-1.0);
        if (theta3 >= 2.0 * acos(-1.0)) theta3 -= 2.0 * acos(-1.0);
        double x3_theta3, y3_theta3;
        x3_theta3 = x0 + r * B / sqrt(pow2(A) + pow2(B)) * cos(theta3) + r * A * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta3);
        y3_theta3 = y0 - r * A / sqrt(pow2(A) + pow2(B)) * cos(theta3) + r * B * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta3);
        //cout << x3_theta3 << " " << y3_theta3 << " " << theta3 << endl;
        if (fabs(x3 - x3_theta3) > 1e-6 || fabs(y3 - y3_theta3) > 1e-6) {
            theta3 = acos(-1.0) - theta3;
            if (theta3 < 0) theta3 += 2.0 * acos(-1.0);
            if (theta3 >= 2.0 * acos(-1.0)) theta3 -= 2.0 * acos(-1.0);
        }
        sin_theta4 = -(z4 - z0) / r * sqrt(pow2(A) + pow2(B) + pow2(C)) / sqrt(pow2(A) + pow2(B));
        theta4 = asin(sin_theta4);
        if (theta4 < 0) theta4 += 2.0 * acos(-1.0);
        if (theta4 >= 2.0 * acos(-1.0)) theta4 -= 2.0 * acos(-1.0);
        double x4_theta4, y4_theta4;
        x4_theta4 = x0 + r * B / sqrt(pow2(A) + pow2(B)) * cos(theta4) + r * A * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta4);
        y4_theta4 = y0 - r * A / sqrt(pow2(A) + pow2(B)) * cos(theta4) + r * B * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta4);
        //cout << x4_theta4 << " " << y4_theta4 << " " << theta4 << endl;
        if (fabs(x4 - x4_theta4) > 1e-6 || fabs(y4 - y4_theta4) > 1e-6) {
            theta4 = acos(-1.0) - theta4;
            if (theta4 < 0) theta4 += 2.0 * acos(-1.0);
            if (theta4 >= 2.0 * acos(-1.0)) theta4 -= 2.0 * acos(-1.0);
        }
        //cout << "theta3: " << theta3 << endl;
        //cout << "theta4: " << theta4 << endl;
        if (theta3 < theta4) flag = 1;
        else flag = -1;
        MP = (M - P).norm();
        QN = (Q - N).norm();
        L = fabs(theta3 - theta4) * r;
        //cout << "L: " << L << endl;
        d = MP + L + QN;
        //cout << "d: " << d << endl;
        num_of_path = (int)(d / distanse_of_pre_move) + 1;
        //cout << "num_of_path: " << num_of_path << endl;
        double t_pre_point = d / num_of_path;
        bool on_ground = true;
        for (int i = 1; i <= num_of_path; i++) {
            double di = t_pre_point * i;
            //cout << "di: " << di << endl;
            double xi, yi, zi;
            if (di <= MP || di >= MP + L) {
                xi = (x2 - x1) * (di / d) + x1;
                yi = (y2 - y1) * (di / d) + y1;
                zi = (z2 - z1) * (di / d) + z1;
            }
            else {
                double di_theta = di - MP;
                double theta_i = flag * di_theta / r + theta3;
                //cout << "theta_i: " << theta_i << endl;
                xi = x0 + r * B / sqrt(pow2(A) + pow2(B)) * cos(theta_i) + r * A * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta_i);
                yi = y0 - r * A / sqrt(pow2(A) + pow2(B)) * cos(theta_i) + r * B * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta_i);
                zi = z0 - r * sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta_i);
                if (zi < 0.05) {
                    //如果路径点高度为负值，表示求解的路径是下半部分球体，翻转方向
                    on_ground = false;
                    break;
                }
            }
            points.push_back(Eigen::Vector3d(xi, yi, zi));
        }
        //return d;
        if (!on_ground) {
            cout << "Another way." << endl;
            L = 2.0 * acos(-1.0) * r - fabs(theta3 - theta4) * r;
            //cout << "L: " << L << endl;
            d = MP + L + QN;
            //cout << "d: " << d << endl;
            num_of_path = (int)(d / distanse_of_pre_move) + 1;
            //cout << "num_of_path: " << num_of_path << endl;
            t_pre_point = d / num_of_path;
            flag = -flag;
            points.clear();
            for (int i = 1; i <= num_of_path; i++) {
                double di = t_pre_point * i;
                //cout << "di: " << di << endl;
                double xi, yi, zi;
                if (di <= MP || di >= MP + L) {
                    xi = (x2 - x1) * (di / d) + x1;
                    yi = (y2 - y1) * (di / d) + y1;
                    zi = (z2 - z1) * (di / d) + z1;
                }
                else {
                    double di_theta = di - MP;
                    double theta_i = flag * di_theta / r + theta3;
                    //cout << "theta_i: " << theta_i << endl;
                    xi = x0 + r * B / sqrt(pow2(A) + pow2(B)) * cos(theta_i) + r * A * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta_i);
                    yi = y0 - r * A / sqrt(pow2(A) + pow2(B)) * cos(theta_i) + r * B * C / sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta_i);
                    zi = z0 - r * sqrt(pow2(A) + pow2(B)) / sqrt(pow2(A) + pow2(B) + pow2(C)) * sin(theta_i);
                }
                points.push_back(Eigen::Vector3d(xi, yi, zi));
            }
        }
    }
    return num_of_path;
}

class View_Space
{
public:
	int num_of_views;						//�ӵ����
	vector<View> views;							//�ռ�Ĳ����ӵ�
	Eigen::Vector3d object_center_world;		//��������
	double predicted_size;						//����BBX�뾶
	Eigen::Vector3d map_center;
	double map_size;
	int id;										//�ڼ���nbv����
	Eigen::Matrix4d now_camera_pose_world;		//���nbv���������λ��
	int first_view_id;
	int occupied_voxels;						
	double map_entropy;	
	bool object_map_changed;
	double octomap_resolution;
	pcl::visualization::PCLVisualizer::Ptr viewer;
	double height_of_ground;
	double cut_size;
	unordered_set<octomap::OcTreeKey,octomap::OcTreeKey::KeyHash>* views_key_set;
	octomap::ColorOcTree* octo_model;
	Voxel_Information* voxel_information;
	double camera_to_object_dis;
	Share_Data* share_data;

	bool vaild_view(View& view) {
		double x = view.init_pos(0);
		double y = view.init_pos(1);
		double z = view.init_pos(2);
		bool vaild = true;
		//����bbx����2���ڲ������ӵ�
		if (x > object_center_world(0) - 2 * predicted_size && x < object_center_world(0) + 2 * predicted_size
		&&  y > object_center_world(1) - 2 * predicted_size && y < object_center_world(1) + 2 * predicted_size
		&&  z > object_center_world(2) - 2 * predicted_size && z < object_center_world(2) + 2 * predicted_size) vaild = false;
		//�ڰ뾶Ϊ4��BBX��С������
		if (pow2(x - object_center_world(0)) + pow2(y - object_center_world(1)) + pow2(z- object_center_world(2)) - pow2(4* predicted_size) > 0 ) vaild = false;
		//�˲��������д�����hash����û��
		//octomap::OcTreeKey key;	bool key_have = octo_model->coordToKeyChecked(x,y,z, key); 
		//if (!key_have) vaild = false;
		//if (key_have && views_key_set->find(key) != views_key_set->end())vaild = false;
		return vaild;
	}

	double check_size(double predicted_size, vector<Eigen::Vector3d>& points) {
		int vaild_points = 0;
		for (auto& ptr : points) {
			if (ptr(0) < object_center_world(0) - predicted_size || ptr(0) > object_center_world(0) + predicted_size) continue;
			if (ptr(1) < object_center_world(1) - predicted_size || ptr(1) > object_center_world(1) + predicted_size) continue;
			if (ptr(2) < object_center_world(2) - predicted_size || ptr(2) > object_center_world(2) + predicted_size) continue;
			vaild_points++;
		}
		return (double)vaild_points / (double)points.size();
	}

	void get_view_space(vector<Eigen::Vector3d>& points) {
		double now_time = clock();
		object_center_world = Eigen::Vector3d(0, 0, 0);
		//获取点云中心
		for (auto& ptr : points) {
			object_center_world(0) += ptr(0);
			object_center_world(1) += ptr(1);
			object_center_world(2) += ptr(2);
		}
		object_center_world(0) /= points.size();
		object_center_world(1) /= points.size();
		object_center_world(2) /= points.size();

		// //物体尺寸预测老版本，获取点云中心到最远点的距离
		// predicted_size = 0.0;
		// for (auto& ptr : points) {
		// 	predicted_size = max(predicted_size, (object_center_world - ptr).norm());
		// }
		// predicted_size *= share_data->size_scale;
		// //cout << "object's bbx solved within precentage "<< precent<< " with executed time " << clock() - now_time << " ms." << endl;
		// cout << "object's pos is ("<< object_center_world(0) << "," << object_center_world(1) << "," << object_center_world(2) << ") and size is " << predicted_size << endl;

		//物体尺寸预测新版本，获取点云中心到95%点的距离
		predicted_size = 0.0;
		double precent = 0.95; //考虑95%的点因为有些点是离群点和噪声点
		vector<double> dis;
		for (auto& ptr : points) {
			dis.push_back((object_center_world - ptr).norm());
		}
		sort(dis.begin(), dis.end());
		predicted_size = dis[(int)(dis.size() * precent)];
		predicted_size *= share_data->size_scale; //预测尺寸放大,因为只考虑了95%的点，所以放大一点
		cout << "object's pos is (" << object_center_world(0) << "," << object_center_world(1) << "," << object_center_world(2) << ") and size is " << predicted_size << endl;

		// //随机采样视点
		// int sample_num = 0;
		// int viewnum = 0;
		// //��һ���ӵ�̶�Ϊģ������
		// View view(Eigen::Vector3d(-0.065348 + object_center_world(0), 0.292504 + object_center_world(1), 0.0130882 + object_center_world(2)));
		// if (!vaild_view(view)) cout << "check init view." << endl;
		// views.push_back(view);
		// views_key_set->insert(octo_model->coordToKey(view.init_pos(0), view.init_pos(1), view.init_pos(2)));
		// viewnum++;
		// while (viewnum != num_of_views) {
		// 	//3��BBX��һ����������
		// 	double x = get_random_coordinate(object_center_world(0) - predicted_size * 4, object_center_world(0) + predicted_size * 4);
		// 	double y = get_random_coordinate(object_center_world(1), object_center_world(1) + predicted_size * 4);
		// 	double z = get_random_coordinate(object_center_world(2) - predicted_size * 4, object_center_world(2) + predicted_size * 4);
		// 	View view(Eigen::Vector3d(x, y, z));
		// 	view.id = viewnum;
		// 	//cout << x<<" " << y << " " << z << endl;
		// 	//�����������ӵ㱣��
		// 	if (vaild_view(view)) {
		// 		view.space_id = id;
		// 		view.dis_to_obejct = (object_center_world - view.init_pos).norm();
		// 		pair<int, double> local_path = get_local_path(Eigen::Vector3d(now_camera_pose_world(0, 3), now_camera_pose_world(1, 3), now_camera_pose_world(2, 3)).eval(), view.init_pos.eval(), object_center_world.eval(), predicted_size * sqrt(2)); //��Χ�а뾶�ǰ�߳��ĸ���2��
		// 		if (local_path.first < 0) cout << "local path wrong." << endl;
		// 		view.robot_cost = local_path.second;
		// 		views.push_back(view);
		// 		views_key_set->insert(octo_model->coordToKey(x,y,z));
		// 		viewnum++;
		// 	}
		// 	sample_num++;
		// 	if (sample_num >= 10 * num_of_views) {
		// 		cout << "lack of space to get view. error." << endl;
		// 		break;
		// 	}
		// }
		// cout << "view set is " << views_key_set->size() << endl;
		// cout<< views.size() << " views getted with sample_times " << sample_num << endl;
		// cout << "view_space getted form octomap with executed time " << clock() - now_time << " ms." << endl;
	}

	~View_Space() {
		//delete views_key_set;
	}

	View_Space(int _id, Share_Data* _share_data, Voxel_Information* _voxel_information, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,int _first_view_id) {
		share_data = _share_data;
		object_map_changed = false;
		first_view_id = _first_view_id;
		id = _id;
		num_of_views = share_data->num_of_views;
		now_camera_pose_world = share_data->now_camera_pose_world;
		voxel_information = _voxel_information;
		viewer = share_data->viewer;
		//views_key_set = new unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash>();
		//gt或者在线模式
		if (share_data->gt_mode || share_data->object_center_mode == 0) {
			ifstream fin("/home/user/pan/ma-scvp-real/src/NBV_Simulation_MA-SCVP/view_space.txt");
			if (fin.is_open()) { //�����ļ��Ͷ��ӵ㼯��
				int num;
				fin >> num;
				share_data->num_of_views = num;
				num_of_views = num;
				//对点云进行滤波，除去重复点，否则质心会有偏移
				pcl::VoxelGrid<pcl::PointXYZRGB> sor;
				sor.setLeafSize(share_data->ground_truth_resolution, share_data->ground_truth_resolution, share_data->ground_truth_resolution);
				sor.setInputCloud(cloud);
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
				sor.filter(*cloud_filtered);
				//计算点云的质心
				vector<Eigen::Vector3d> points;
				for (auto& ptr : cloud_filtered->points) {
					Eigen::Vector3d pt(ptr.x, ptr.y, ptr.z);
					//计算尺寸的时候需要保守一点，高度上多加1cm，因为点云的高度可能会有一点点误差
					if (pt(2) < share_data->height_of_ground + 0.01) continue;
					points.push_back(pt);
				}
				get_view_space(points);
				//中心点可以往下移动一点，因为点云的高度可能会有一点点误差
				object_center_world(2) -= 0.01;
				views.clear();
				//views_key_set->clear();
				for (int i = 0; i < num_of_views; i++) {
					double init_pos[3];
					fin >> init_pos[0] >> init_pos[1] >> init_pos[2];
					init_pos[0] = init_pos[0] / 0.4 * (share_data->camera_to_object_dis + predicted_size);
					init_pos[1] = init_pos[1] / 0.4 * (share_data->camera_to_object_dis + predicted_size);
					init_pos[2] = init_pos[2] / 0.4 * (share_data->camera_to_object_dis + predicted_size);
					View view(Eigen::Vector3d(init_pos[0] + object_center_world(0), init_pos[1] + object_center_world(1), init_pos[2] + object_center_world(2) + share_data->up_shift));
					view.id = i;
					view.space_id = id;
					view.dis_to_obejct = (object_center_world - view.init_pos).norm();
					views.push_back(view);
					//views_key_set->insert(octo_model->coordToKey(init_pos[0], init_pos[1], init_pos[2]));
				}
				for (int i = 0; i < num_of_views; i++) {
					pair<int, double> local_path = get_local_path(views[first_view_id].init_pos.eval(), views[i].init_pos.eval(), object_center_world.eval(), predicted_size + share_data->safe_distance); //��Χ�а뾶�ǰ�߳��ĸ���2��
					if (local_path.first < 0) cout << "local path wrong." << endl;
					views[i].robot_cost = local_path.second;
				}
				cout << "viewspace readed." << endl;
			}
			else {
				cout << "no view space. check!" << endl;
			}
		}
		else { //online mode with gt
			ifstream fin(share_data->pcd_file_path + share_data->name_of_pcd + "_vs.txt");
			if (fin.is_open()) { //�����ļ��Ͷ��ӵ㼯��
				int num;
				fin >> num;
				if (num != num_of_views) cout << "viewspace read error. check input viewspace size." << endl;
				double object_center[3];
				fin >> object_center[0] >> object_center[1] >> object_center[2];
				object_center_world(0) = object_center[0];
				object_center_world(1) = object_center[1];
				object_center_world(2) = object_center[2];
				fin >> predicted_size;
				for (int i = 0; i < num_of_views; i++) {
					double init_pos[3];
					fin >> init_pos[0] >> init_pos[1] >> init_pos[2];
					View view(Eigen::Vector3d(init_pos[0], init_pos[1], init_pos[2]));
					view.id = i;
					view.space_id = id;
					view.dis_to_obejct = (object_center_world - view.init_pos).norm();
					view.robot_cost = (Eigen::Vector3d(now_camera_pose_world(0, 3), now_camera_pose_world(1, 3), now_camera_pose_world(2, 3)).eval() - view.init_pos).norm();
					views.push_back(view);
					//views_key_set->insert(octo_model->coordToKey(init_pos[0], init_pos[1], init_pos[2]));
				}
				cout << "object's pos is ("<< object_center_world(0) << "," << object_center_world(1) << "," << object_center_world(2) << ") and size is " << predicted_size << endl;
				for (int i = 0; i < num_of_views; i++) {
					pair<int, double> local_path = get_local_path(views[first_view_id].init_pos.eval(), views[i].init_pos.eval(), object_center_world.eval(), predicted_size + share_data->safe_distance); //��Χ�а뾶�ǰ�߳��ĸ���2��
					if (local_path.first < 0) cout << "local path wrong." << endl;
					views[i].robot_cost = local_path.second;
				}
				cout << "viewspace readed." << endl;
			}
			else {
				cout << "no viewspace readed. Check!" << endl;
			}
		}
		//GT模式，直接用GT的物体中心和尺寸
		share_data->object_center_world = object_center_world;
		share_data->predicted_size = predicted_size;

		//地图尺寸大一些，用于适应物体中心和动态尺寸
		map_size = share_data->map_scale * predicted_size;
		map_center = object_center_world;
		share_data->map_size = map_size;
		share_data->map_center = map_center;

		//动态分辨率
		double predicted_octomap_resolution = map_size * 2.0 / 32.0; //使用地图尺寸来预测分辨率
		cout << "choose octomap_resolution: " << predicted_octomap_resolution << " m." << endl;
		share_data->octomap_resolution = predicted_octomap_resolution;
		share_data->octo_model = new octomap::ColorOcTree(share_data->octomap_resolution);
		share_data->octo_model->setOccupancyThres(0.65);
		//share_data->GT_sample = new octomap::ColorOcTree(share_data->octomap_resolution);

		octo_model = share_data->octo_model;
		octomap_resolution = share_data->octomap_resolution;

		//初始化地图
		double now_time = clock();
		for (double x = share_data->map_center(0) - share_data->map_size; x <= share_data->map_center(0) + share_data->map_size; x += share_data->octomap_resolution)
			for (double y = share_data->map_center(1) - share_data->map_size; y <= share_data->map_center(1) + share_data->map_size; y += share_data->octomap_resolution)
				for (double z = share_data->map_center(2) - share_data->map_size; z <= share_data->map_center(2) + share_data->map_size; z += share_data->octomap_resolution)
					octo_model->setNodeValue(x, y, z, (float)0, true); //occ0.5 = logodds0	
		octo_model->updateInnerOccupancy();
		share_data->init_entropy = 0;
		share_data->voxels_in_BBX = 0;
		for (double x = share_data->map_center(0) - share_data->map_size; x <= share_data->map_center(0) + share_data->map_size; x += share_data->octomap_resolution)
			for (double y = share_data->map_center(1) - share_data->map_size; y <= share_data->map_center(1) + share_data->map_size; y += share_data->octomap_resolution)
				for (double z = share_data->map_center(2) - share_data->map_size; z <= share_data->map_center(2) + share_data->map_size; z += share_data->octomap_resolution)
				{
					double occupancy = octo_model->search(x, y, z)->getOccupancy();
					share_data->init_entropy += voxel_information->entropy(occupancy);
					share_data->voxels_in_BBX++;
				}
		voxel_information->init_mutex_voxels(share_data->voxels_in_BBX);
		cout << "Map_init has voxels(in BBX) " << share_data->voxels_in_BBX << " and entropy " << share_data->init_entropy << endl;
	}

	void update(int _id, Share_Data* _share_data, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,pcl::PointCloud<pcl::PointXYZRGB>::Ptr update_cloud) {
		share_data = _share_data;
		object_map_changed = false;
		id = _id;
		now_camera_pose_world = share_data->now_camera_pose_world;

		//第一次更新，没有变化
		if(id != 0){
			//如果是gt模式或者在线模式，更新物体中心和尺寸
			if (share_data->gt_mode || share_data->object_center_mode == 0) {
				ifstream fin("/home/user/pan/ma-scvp-real/src/NBV_Simulation_MA-SCVP/view_space.txt");
				if (fin.is_open()) { //�����ļ��Ͷ��ӵ㼯��
					//保存中间view
					vector<View> temp_views;
					//注意视点需要按照id排序来建立映射
					sort(views.begin(), views.end(), view_id_compare);
					for (auto& view : views) {
						temp_views.push_back(view);
					}
					//重新打开viewspace
					int num;
					fin >> num;
					share_data->num_of_views = num;
					num_of_views = num;
					//对点云进行滤波，除去重复点，否则质心会有偏移
					pcl::VoxelGrid<pcl::PointXYZRGB> sor;
					sor.setLeafSize(share_data->ground_truth_resolution, share_data->ground_truth_resolution, share_data->ground_truth_resolution);
					sor.setInputCloud(cloud);
					pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
					sor.filter(*cloud_filtered);
					//计算点云的质心
					vector<Eigen::Vector3d> points;
					for (auto& ptr : cloud_filtered->points) {
						Eigen::Vector3d pt(ptr.x, ptr.y, ptr.z);
						//计算尺寸的时候需要保守一点，高度上多加1cm，因为点云的高度可能会有一点点误差
						if (pt(2) < share_data->height_of_ground + 0.01) continue;
						points.push_back(pt);
					}
					get_view_space(points);
					//中心点可以往下移动一点，因为点云的高度可能会有一点点误差
					object_center_world(2) -= 0.01;
					views.clear();
					for (int i = 0; i < num_of_views; i++) {
						double init_pos[3];
						fin >> init_pos[0] >> init_pos[1] >> init_pos[2];
						init_pos[0] = init_pos[0] / 0.4 * (share_data->camera_to_object_dis + predicted_size);
						init_pos[1] = init_pos[1] / 0.4 * (share_data->camera_to_object_dis + predicted_size);
						init_pos[2] = init_pos[2] / 0.4 * (share_data->camera_to_object_dis + predicted_size);
						View view(Eigen::Vector3d(init_pos[0] + object_center_world(0), init_pos[1] + object_center_world(1), init_pos[2] + object_center_world(2) + share_data->up_shift));
						view.id = i;
						view.space_id = id;
						view.dis_to_obejct = (object_center_world - view.init_pos).norm();
						view.vis = temp_views[i].vis;
						views.push_back(view);
					}
					cout << "viewspace readed again." << endl;
				}
				else {
					cout << "no view space. check!" << endl;
				}
			}
			//更新物体尺寸和中心
			share_data->object_center_world = object_center_world;
			share_data->predicted_size = predicted_size;
			//更新完毕后，如果物体中心或者尺寸发生较大变化，超出地图描述范围，需要重新初始化地图
			bool large_change = false;
			//检查新的物体BBX是否在map范围内
			if(object_center_world(0) - predicted_size < share_data->map_center(0) - share_data->map_size) large_change = true;
			if(object_center_world(0) + predicted_size > share_data->map_center(0) + share_data->map_size) large_change = true;
			if(object_center_world(1) - predicted_size < share_data->map_center(1) - share_data->map_size) large_change = true;
			if(object_center_world(1) + predicted_size > share_data->map_center(1) + share_data->map_size) large_change = true;
			if(object_center_world(2) - predicted_size < share_data->map_center(2) - share_data->map_size) large_change = true;
			if(object_center_world(2) + predicted_size > share_data->map_center(2) + share_data->map_size) large_change = true;

			// 我们的方法中，如果不需要使用地图了，那么就不需要重新初始化地图
			if (share_data->method_of_IG == 7 && id > 0 + share_data->num_of_nbvs_combined) large_change = false;

			//如果发生较大变化，需要重新初始化地图
			if(!large_change){
				cout << "not large change. keep map. update center and size." << endl;
			}
			else
			{
				cout<< "large change. reinit map. take some time ..." << endl;
				cout<< "To aviod reinit, you can use a larger map by change map_scale! Note this will affect the resolution a bit." << endl;

				//地图尺寸大一些，用于适应物体中心和动态尺寸
				map_size = share_data->map_scale * predicted_size;
				map_center = object_center_world;
				share_data->map_size = map_size;
				share_data->map_center = map_center;

				//动态分辨率
				double predicted_octomap_resolution = map_size * 2.0 / 32.0; //使用地图尺寸来预测分辨率
				cout << "rechoose octomap_resolution: " << predicted_octomap_resolution << " m." << endl;
				share_data->octomap_resolution = predicted_octomap_resolution;
				delete share_data->octo_model;
				share_data->octo_model = new octomap::ColorOcTree(share_data->octomap_resolution);
				share_data->octo_model->setOccupancyThres(0.65);

				octo_model = share_data->octo_model;
				octomap_resolution = share_data->octomap_resolution;

				//初始化地图
				double now_time = clock();
				for (double x = share_data->map_center(0) - share_data->map_size; x <= share_data->map_center(0) + share_data->map_size; x += share_data->octomap_resolution)
					for (double y = share_data->map_center(1) - share_data->map_size; y <= share_data->map_center(1) + share_data->map_size; y += share_data->octomap_resolution)
						for (double z = share_data->map_center(2) - share_data->map_size; z <= share_data->map_center(2) + share_data->map_size; z += share_data->octomap_resolution)
							octo_model->setNodeValue(x, y, z, (float)0, true); //occ0.5 = logodds0	
				octo_model->updateInnerOccupancy();
				share_data->init_entropy = 0;
				share_data->voxels_in_BBX = 0;
				for (double x = share_data->map_center(0) - share_data->map_size; x <= share_data->map_center(0) + share_data->map_size; x += share_data->octomap_resolution)
					for (double y = share_data->map_center(1) - share_data->map_size; y <= share_data->map_center(1) + share_data->map_size; y += share_data->octomap_resolution)
						for (double z = share_data->map_center(2) - share_data->map_size; z <= share_data->map_center(2) + share_data->map_size; z += share_data->octomap_resolution)
						{
							double occupancy = octo_model->search(x, y, z)->getOccupancy();
							share_data->init_entropy += voxel_information->entropy(occupancy);
							share_data->voxels_in_BBX++;
						}
				voxel_information->clear_mutex_voxels();
				voxel_information->init_mutex_voxels(share_data->voxels_in_BBX);
				cout << "Map_reinit has voxels(in BBX) " << share_data->voxels_in_BBX << " and entropy " << share_data->init_entropy << endl;

				//插入之前的点云，并更新f_voxels
				share_data->f_voxels.clear();
				for (int nbv_idx = 0; nbv_idx < id; nbv_idx++){
					octomap::Pointcloud cloud_octo;
					for (auto p : share_data->clouds[nbv_idx]->points) {
						cloud_octo.push_back(p.x, p.y, p.z);
					}
					octo_model->insertPointCloud(cloud_octo, octomap::point3d(share_data->nbvs_pose_world[nbv_idx](0, 3), share_data->nbvs_pose_world[nbv_idx](1, 3), share_data->nbvs_pose_world[nbv_idx](2, 3)), -1, true, false);
					for (auto p : share_data->clouds[nbv_idx]->points) {
						octo_model->integrateNodeColor(p.x, p.y, p.z, p.r, p.g, p.b);
					}
					octo_model->updateInnerOccupancy();
					cout << "reinsert cloud into new octomap with NBV " << nbv_idx << endl;
					//if the method is not (combined) one-shot and random, then use f_voxel to decide whether to stop
					if(!(share_data->Combined_on == true || share_data->method_of_IG == 7 || share_data->method_of_IG == 8)){
						//compute f_voxels
						int f_voxels_num = 0;
						for (octomap::ColorOcTree::leaf_iterator it = octo_model->begin_leafs(), end = octo_model->end_leafs(); it != end; ++it) {
							double occupancy = (*it).getOccupancy();
							if (fabs(occupancy - 0.5) < 1e-3) { // unknown
								auto coordinate = it.getCoordinate();
								if (coordinate.x() >= share_data->map_center(0) - share_data->map_size && coordinate.x() <= share_data->map_center(0) + share_data->map_size
									&& coordinate.y() >= share_data->map_center(1) - share_data->map_size && coordinate.y() <= share_data->map_center(1) + share_data->map_size
									&& coordinate.z() >= share_data->map_center(2) - share_data->map_size && coordinate.z() <= share_data->map_center(2) + share_data->map_size)
								{
									// compute the frontier voxels that is unknown and has at least one free and one occupied neighbor
									int free_cnt = 0;
									int occupied_cnt = 0;
									for (int i = -1; i <= 1; i++)
										for (int j = -1; j <= 1; j++)
											for (int k = -1; k <= 1; k++)
											{
												if (i == 0 && j == 0 && k == 0) continue;
												double x = coordinate.x() + i * share_data->octomap_resolution;
												double y = coordinate.y() + j * share_data->octomap_resolution;
												double z = coordinate.z() + k * share_data->octomap_resolution;
												octomap::point3d neighbour(x, y, z);
												octomap::OcTreeKey neighbour_key;  bool neighbour_key_have = share_data->octo_model->coordToKeyChecked(neighbour, neighbour_key);
												if (neighbour_key_have) {
													octomap::ColorOcTreeNode* neighbour_voxel = share_data->octo_model->search(neighbour_key);
													if (neighbour_voxel != NULL) {
														double neighbour_occupancy = neighbour_voxel->getOccupancy();
														free_cnt += neighbour_occupancy < 0.5 ? 1 : 0;
														occupied_cnt += neighbour_occupancy > 0.5 ? 1 : 0;
													}
												}
											}
									//edge
									if (free_cnt >= 1 && occupied_cnt >= 1) {
										f_voxels_num++;
										//cout << "f voxel: " << coordinate.x() << " " << coordinate.y() << " " << coordinate.z() << endl;
									}
								}
							}
						}
						share_data->f_voxels.push_back(f_voxels_num);
						cout << "update map " << nbv_idx << " with f_voxels " << f_voxels_num << endl;
					}
				}

				//注意一定要更新标志
				object_map_changed = true;
			}
		}

		//更新试点路径
		for (int i = 0; i < views.size(); i++) {
			views[i].space_id = id;
			pair<int, double> local_path = get_local_path(Eigen::Vector3d(now_camera_pose_world(0, 3), now_camera_pose_world(1, 3), now_camera_pose_world(2, 3)).eval(), views[i].init_pos.eval(), object_center_world.eval(), predicted_size + share_data->safe_distance); //��Χ�а뾶�ǰ�߳��ĸ���2��
			if (local_path.first < 0) cout << "local path wrong." << endl;
			views[i].robot_cost = local_path.second;
		}
		//插入当前点云
		double now_time = clock();
		octomap::Pointcloud cloud_octo;
		for (auto p : update_cloud->points) {
			cloud_octo.push_back(p.x, p.y, p.z);
		}
		octo_model->insertPointCloud(cloud_octo, octomap::point3d(now_camera_pose_world(0, 3), now_camera_pose_world(1, 3), now_camera_pose_world(2, 3)), -1, true, false);
		for (auto p : update_cloud->points) {
			octo_model->integrateNodeColor(p.x, p.y, p.z, p.r, p.g, p.b);
		}
		octo_model->updateInnerOccupancy();
		cout << "Octomap updated via cloud with executed time " << clock() - now_time << " ms." << endl;
		map_entropy = 0;
		occupied_voxels = 0;

		//统计地图信息
		for (double x = share_data->map_center(0) - share_data->map_size; x <= share_data->map_center(0) + share_data->map_size; x += share_data->octomap_resolution)
			for (double y = share_data->map_center(1) - share_data->map_size; y <= share_data->map_center(1) + share_data->map_size; y += share_data->octomap_resolution)
				for (double z = share_data->map_center(2) - share_data->map_size; z <= share_data->map_center(2) + share_data->map_size; z += share_data->octomap_resolution)
				{
					auto node = octo_model->search(x, y, z);
					if (node == NULL) cout << "map out of range!" << endl;
					double occupancy = node->getOccupancy();
					map_entropy += voxel_information->entropy(occupancy);
					if (occupancy > 0.5 && z > share_data->height_of_ground) occupied_voxels++;
				}
		if(share_data->is_save)	{
			share_data->access_directory(share_data->save_path + "/octomaps");
			share_data->octo_model->write(share_data->save_path + "/octomaps/octomap"+to_string(id)+".ot");
		}
		
		if (id == 0) {
			share_data->access_directory(share_data->save_path + "/quantitative");
			ofstream fout_map(share_data->save_path+"/quantitative/Map" + to_string(-1) + ".txt");
			fout_map << 0 << '\t' << share_data->init_entropy << '\t' << 0 << '\t' << 1 << endl;
		}

		cout << "Map " << id << " has voxels " << occupied_voxels << ". Map " << id << " has entropy " << map_entropy << endl;
		cout << "Map " << id << " has voxels(rate) " << 1.0 * occupied_voxels / share_data->init_voxels << ". Map " << id << " has entropy(rate) " << map_entropy / share_data->init_entropy << endl;
		share_data->access_directory(share_data->save_path+"/quantitative");
		ofstream fout_map(share_data->save_path +"/quantitative/Map" + to_string(id) + ".txt");
		fout_map << occupied_voxels << '\t' << map_entropy << '\t' << 1.0 * occupied_voxels / share_data->init_voxels << '\t' << map_entropy / share_data->init_entropy << endl;

		// //每次更新voxel_final，效果不佳
		// for (auto p : update_cloud->points) {
		// 	octomap::OcTreeKey key = share_data->ground_truth_model->coordToKey(p.x, p.y, p.z);
		// 	if(share_data->voxel_final->find(key) == share_data->voxel_final->end()) {
		// 		(*share_data->voxel_final)[key] = 1;
		// 	}
		// }

		//做一次ICP
		share_data->cloud_final_downsampled = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::VoxelGrid<pcl::PointXYZRGB> sor;
		sor.setLeafSize(0.005f, 0.005f, 0.005f);
		sor.setInputCloud(share_data->cloud_final);
		sor.filter(*share_data->cloud_final_downsampled);
		pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp_rgb;
		icp_rgb.setTransformationEpsilon(1e-10);
		icp_rgb.setEuclideanFitnessEpsilon(1e-6);
		icp_rgb.setMaximumIterations(100000);
		icp_rgb.setMaxCorrespondenceDistance(share_data->icp_distance);
		// do icp
		icp_rgb.setInputSource(share_data->cloud_final_downsampled);
		icp_rgb.setInputTarget(share_data->cloud_ground_truth_downsampled);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_icp(new pcl::PointCloud<pcl::PointXYZRGB>);
		icp_rgb.align(*cloud_icp);
		Eigen::Matrix4d T_icp(4, 4);
		T_icp = icp_rgb.getFinalTransformation().cast<double>();
		cout << "ICP matraix: " << T_icp << endl;
		//把cloud_final按T_icp变换，存到到cloud_icp
		pcl::transformPointCloud(*share_data->cloud_final, *cloud_icp, T_icp);
		share_data->voxel_final->clear();
		for (auto p : cloud_icp->points) {
			octomap::OcTreeKey key = share_data->ground_truth_model->coordToKey(p.x, p.y, p.z);
			if(share_data->voxel_final->find(key) == share_data->voxel_final->end()) {
				(*share_data->voxel_final)[key] = 1;
			}
		}

		//统计覆盖率
		int num = 0;
		unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel_overlap = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
		for (auto it = share_data->voxel_gt->begin(); it != share_data->voxel_gt->end(); it++) {
			octomap::OcTreeKey key = it->first;
			octomap::point3d point = share_data->ground_truth_model->keyToCoord(key);
			bool gt_in_recontruct = false;
			//search nabourhoods
			for (double x = -share_data->ground_truth_resolution * share_data->vsc_nabourhood_num; x <= share_data->ground_truth_resolution * share_data->vsc_nabourhood_num; x += share_data->ground_truth_resolution)
				for (double y = -share_data->ground_truth_resolution * share_data->vsc_nabourhood_num; y <= share_data->ground_truth_resolution * share_data->vsc_nabourhood_num; y += share_data->ground_truth_resolution)
					for (double z = -share_data->ground_truth_resolution * share_data->vsc_nabourhood_num; z <= share_data->ground_truth_resolution * share_data->vsc_nabourhood_num; z += share_data->ground_truth_resolution)
					{
						octomap::OcTreeKey key_nabourhood = share_data->ground_truth_model->coordToKey(point.x() + x, point.y() + y, point.z() + z);
						if (share_data->voxel_final->find(key_nabourhood) != share_data->voxel_final->end()) {
							gt_in_recontruct = true;
							break;
						}
					}
			if (gt_in_recontruct && voxel_overlap->find(key) == voxel_overlap->end()) {
				(*voxel_overlap)[key] = num++;
			}
		}

		if(share_data->is_save){
			//保存中心信息，用于后续分析: object_center_world, predicted_size, map_center, map_size, resolution, voxel in bbx
			share_data->access_directory(share_data->save_path + "/centers");
			ofstream fout_center(share_data->save_path + "/centers/" + to_string(id) + ".txt");
			fout_center << object_center_world(0) << '\t' << object_center_world(1) << '\t' << object_center_world(2) << '\t' << predicted_size << endl;
			fout_center << map_center(0) << '\t' << map_center(1) << '\t' << map_center(2) << '\t' << map_size << endl;
			fout_center << share_data->octomap_resolution << '\t' << share_data->voxels_in_BBX << endl;
			//保存voxel_overlap的octomap
			share_data->access_directory(share_data->save_path + "/covered_voxels");
			octomap::ColorOcTree* covered_voxels_model = new octomap::ColorOcTree(share_data->ground_truth_resolution);
			covered_voxels_model->setOccupancyThres(0.65);
			for (auto it = share_data->voxel_gt->begin(); it != share_data->voxel_gt->end(); it++) {
				octomap::OcTreeKey key = it->first;
				//设置为已占用体素
				covered_voxels_model->setNodeValue(key, octomap::logodds(1.0), true);
				covered_voxels_model->setNodeColor(key, 128, 128, 128);
			}
			covered_voxels_model->updateInnerOccupancy();
			for (auto it = voxel_overlap->begin(); it != voxel_overlap->end(); it++) {
				octomap::OcTreeKey key = it->first;
				covered_voxels_model->setNodeColor(key, 255, 0, 0);
			}
			covered_voxels_model->updateInnerOccupancy();
			covered_voxels_model->write(share_data->save_path + "/covered_voxels/" + to_string(id) + ".ot");
			delete covered_voxels_model;
		}

		cout << "Cloud " << id << " has voxels " << num << endl;
		cout << "Cloud " << id << " has voxels(rate/GT) " << 1.0 * num / share_data->cloud_points_number << endl; 
		ofstream fout_cloud(share_data->save_path + "/quantitative/Cloud" + to_string(id) + ".txt");
		fout_cloud <<'\t'<< num << '\t' << 1.0 * num  / share_data->cloud_points_number << endl;
		delete voxel_overlap;
	}

	void remove_bbx(pcl::visualization::PCLVisualizer::Ptr viewer) {
		viewer->removeShape("cube1");
		viewer->removeShape("cube2");
		viewer->removeShape("cube3");
		viewer->removeShape("cube4");
		viewer->removeShape("cube5");
		viewer->removeShape("cube6");
		viewer->removeShape("cube7");
		viewer->removeShape("cube8");
		viewer->removeShape("cube9");
		viewer->removeShape("cube10");
		viewer->removeShape("cube11");
		viewer->removeShape("cube12");
	}

	void add_bbx_to_cloud(pcl::visualization::PCLVisualizer::Ptr viewer) {
		double x1 = object_center_world(0) - predicted_size;
		double x2 = object_center_world(0) + predicted_size;
		double y1 = object_center_world(1) - predicted_size;
		double y2 = object_center_world(1) + predicted_size;
		double z1 = object_center_world(2) - predicted_size;
		double z2 = object_center_world(2) + predicted_size;
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x1, y1, z1), pcl::PointXYZ(x1, y2, z1), 0, 255, 0, "cube1");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x1, y1, z1), pcl::PointXYZ(x2, y1, z1), 0, 255, 0, "cube2");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x1, y1, z1), pcl::PointXYZ(x1, y1, z2), 0, 255, 0, "cube3");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x2, y2, z2), pcl::PointXYZ(x1, y2, z2), 0, 255, 0, "cube4");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x2, y2, z2), pcl::PointXYZ(x2, y1, z2), 0, 255, 0, "cube5");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x2, y2, z2), pcl::PointXYZ(x2, y2, z1), 0, 255, 0, "cube6");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x2, y1, z2), pcl::PointXYZ(x1, y1, z2), 0, 255, 0, "cube8");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x2, y1, z2), pcl::PointXYZ(x2, y1, z1), 0, 255, 0, "cube9");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x1, y2, z2), pcl::PointXYZ(x1, y1, z2), 0, 255, 0, "cube10");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x1, y2, z2), pcl::PointXYZ(x1, y2, z1), 0, 255, 0, "cube11");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x2, y2, z1), pcl::PointXYZ(x1, y2, z1), 0, 255, 0, "cube12");
		viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(x2, y2, z1), pcl::PointXYZ(x2, y1, z1), 0, 255, 0, "cube7");
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "cube1");
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "cube2");
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "cube3");
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "cube4");
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "cube5");
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "cube6");
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "cube7");
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "cube8");
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "cube9");
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "cube10");
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "cube11");
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "cube12");
	}

};
