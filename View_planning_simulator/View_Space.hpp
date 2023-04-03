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
	//生成比较随机的0-1随机数并映射到区间[from,to]
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
	Eigen::Vector3d init_pos;	//初始位置
	Eigen::Matrix4d pose;		//view_i到view_i+1旋转矩阵
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
		return make_pair(LinePath, d);
	}
	else {
		//如果需要穿过球体，则沿着球表面行动
		t3 = (-b - sqrt(pow2(b) - 4.0 * a * c)) / (2.0 * a);
		t4 = (-b + sqrt(pow2(b) - 4.0 * a * c)) / (2.0 * a);
		if ((t3 < 0 || t3 > 1) && (t4 < 0 || t4 > 1)) {
			//球外两点，直接过去
			double d = (N - M).norm();
			//cout << "d: " << d << endl;
			return make_pair(LinePath, d);
		}
		else if ((t3 < 0 || t3 > 1) || (t4 < 0 || t4 > 1)) {
			//起点或终点在障碍物球内
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
		return make_pair(CirclePath, d);
	}
	//未定义行为
	return make_pair(ErrorPath, 1e10);
}

class View_Space
{
public:
	int num_of_views;						//视点个数
	vector<View> views;							//空间的采样视点
	Eigen::Vector3d object_center_world;		//物体中心
	double predicted_size;						//物体BBX半径
	int id;										//第几次nbv迭代
	Eigen::Matrix4d now_camera_pose_world;		//这次nbv迭代的相机位置
	int first_view_id;
	int occupied_voxels;						
	double map_entropy;	
	bool object_changed;
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
		//物体bbx扩大2倍内不生成视点
		if (x > object_center_world(0) - 2 * predicted_size && x < object_center_world(0) + 2 * predicted_size
		&&  y > object_center_world(1) - 2 * predicted_size && y < object_center_world(1) + 2 * predicted_size
		&&  z > object_center_world(2) - 2 * predicted_size && z < object_center_world(2) + 2 * predicted_size) vaild = false;
		//在半径为4倍BBX大小的球内
		if (pow2(x - object_center_world(0)) + pow2(y - object_center_world(1)) + pow2(z- object_center_world(2)) - pow2(4* predicted_size) > 0 ) vaild = false;
		//八叉树索引中存在且hash表中没有
		octomap::OcTreeKey key;	bool key_have = octo_model->coordToKeyChecked(x,y,z, key); 
		if (!key_have) vaild = false;
		if (key_have && views_key_set->find(key) != views_key_set->end())vaild = false;
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
		//计算点云质心
		for (auto& ptr : points) {
			object_center_world(0) += ptr(0);
			object_center_world(1) += ptr(1);
			object_center_world(2) += ptr(2);
		}
		object_center_world(0) /= points.size();
		object_center_world(1) /= points.size();
		object_center_world(2) /= points.size();
		//计算最远点
		predicted_size = 0.0;
		for (auto& ptr : points) {
			predicted_size = max(predicted_size, (object_center_world - ptr).norm());
		}
		predicted_size *= 17.0 / 16.0;
		//cout << "object's bbx solved within precentage "<< precent<< " with executed time " << clock() - now_time << " ms." << endl;
		cout << "object's pos is ("<< object_center_world(0) << "," << object_center_world(1) << "," << object_center_world(2) << ") and size is " << predicted_size << endl;
		/*int sample_num = 0;
		int viewnum = 0;
		//第一个视点固定为模型中心
		View view(Eigen::Vector3d(-0.065348 + object_center_world(0), 0.292504 + object_center_world(1), 0.0130882 + object_center_world(2)));
		if (!vaild_view(view)) cout << "check init view." << endl;
		views.push_back(view);
		views_key_set->insert(octo_model->coordToKey(view.init_pos(0), view.init_pos(1), view.init_pos(2)));
		viewnum++;
		while (viewnum != num_of_views) {
			//3倍BBX的一个采样区域
			double x = get_random_coordinate(object_center_world(0) - predicted_size * 4, object_center_world(0) + predicted_size * 4);
			double y = get_random_coordinate(object_center_world(1), object_center_world(1) + predicted_size * 4);
			double z = get_random_coordinate(object_center_world(2) - predicted_size * 4, object_center_world(2) + predicted_size * 4);
			View view(Eigen::Vector3d(x, y, z));
			view.id = viewnum;
			//cout << x<<" " << y << " " << z << endl;
			//符合条件的视点保留
			if (vaild_view(view)) {
				view.space_id = id;
				view.dis_to_obejct = (object_center_world - view.init_pos).norm();
				pair<int, double> local_path = get_local_path(Eigen::Vector3d(now_camera_pose_world(0, 3), now_camera_pose_world(1, 3), now_camera_pose_world(2, 3)).eval(), view.init_pos.eval(), object_center_world.eval(), predicted_size * sqrt(2)); //包围盒半径是半边长的根号2倍
				if (local_path.first < 0) cout << "local path wrong." << endl;
				view.robot_cost = local_path.second;
				views.push_back(view);
				views_key_set->insert(octo_model->coordToKey(x,y,z));
				viewnum++;
			}
			sample_num++;
			if (sample_num >= 10 * num_of_views) {
				cout << "lack of space to get view. error." << endl;
				break;
			}
		}
		cout << "view set is " << views_key_set->size() << endl;
		cout<< views.size() << " views getted with sample_times " << sample_num << endl;
		cout << "view_space getted form octomap with executed time " << clock() - now_time << " ms." << endl;*/
	}

	~View_Space() {
		delete views_key_set;
	}

	View_Space(int _id, Share_Data* _share_data, Voxel_Information* _voxel_information, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,int _first_view_id) {
		share_data = _share_data;
		object_changed = false;
		first_view_id = _first_view_id;
		id = _id;
		num_of_views = share_data->num_of_views;
		now_camera_pose_world = share_data->now_camera_pose_world;
		octo_model = share_data->octo_model;
		octomap_resolution = share_data->octomap_resolution;
		voxel_information = _voxel_information;
		viewer = share_data->viewer;
		views_key_set = new unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash>();
		//检测viewspace是否已经生成
		//if (share_data->method_of_IG == 7) {//sc-net
			ifstream fin( "../view_space.txt");
			if (fin.is_open()) { //存在文件就读视点集合
				int num;
				fin >> num;
				share_data->num_of_views = num;
				num_of_views = num;
				vector<Eigen::Vector3d> points;
				for (auto& ptr : cloud->points) {
					Eigen::Vector3d pt(ptr.x, ptr.y, ptr.z);
					points.push_back(pt);
				}
				get_view_space(points);
				views.clear();
				views_key_set->clear();
				for (int i = 0; i < num_of_views; i++) {
					double init_pos[3];
					fin >> init_pos[0] >> init_pos[1] >> init_pos[2];
					View view(Eigen::Vector3d(init_pos[0] + object_center_world(0), init_pos[1] + object_center_world(1), init_pos[2] + object_center_world(2)));
					view.id = i;
					view.space_id = id;
					view.dis_to_obejct = (object_center_world - view.init_pos).norm();
					views.push_back(view);
					views_key_set->insert(octo_model->coordToKey(init_pos[0], init_pos[1], init_pos[2]));
				}
				for (int i = 0; i < num_of_views; i++) {
					pair<int, double> local_path = get_local_path(views[first_view_id].init_pos.eval(), views[i].init_pos.eval(), object_center_world.eval(), predicted_size * sqrt(2)); //包围盒半径是半边长的根号2倍
					if (local_path.first < 0) cout << "local path wrong." << endl;
					views[i].robot_cost = local_path.second;
				}
				cout << "viewspace readed." << endl;
			}
			else {
				cout << "no view space. check!" << endl;
			}
		//}
		/*else { //search_method
			ifstream fin(share_data->pcd_file_path + share_data->name_of_pcd + ".txt");
			if (fin.is_open()) { //存在文件就读视点集合
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
					views_key_set->insert(octo_model->coordToKey(init_pos[0], init_pos[1], init_pos[2]));
				}
				cout << "viewspace readed." << endl;
			}
			else { //不存在就生成视点集合
				//获取点云BBX
				vector<Eigen::Vector3d> points;
				for (auto& ptr : cloud->points) {
					Eigen::Vector3d pt(ptr.x, ptr.y, ptr.z);
					points.push_back(pt);
				}
				//视点生成器
				get_view_space(points);
				share_data->access_directory(share_data->pcd_file_path);
				ofstream fout(share_data->pcd_file_path + share_data->name_of_pcd + ".txt");
				fout << num_of_views << '\n';
				fout << object_center_world(0) << ' ' << object_center_world(1) << ' ' << object_center_world(2) << '\n';
				fout << predicted_size << '\n';
				for (int i = 0; i < num_of_views; i++)
					fout << views[i].init_pos(0) << ' ' << views[i].init_pos(1) << ' ' << views[i].init_pos(2) << '\n';
				cout << "viewspace getted." << endl;
			}
		}*/	
		//更新一下数据区数据
		share_data->object_center_world = object_center_world;
		share_data->predicted_size = predicted_size;
		double map_size = predicted_size + 3.0 * octomap_resolution;
		share_data->map_size = map_size;
		//第一次的数据，根据BBX初始化地图
		double now_time = clock();
		for (int i = 0; i < 32; i++)
			for (int j = 0; j < 32; j++)
				for (int k = 0; k < 32; k++)
				{
					double x = share_data->object_center_world(0) - share_data->predicted_size + share_data->octomap_resolution * i;
					double y = share_data->object_center_world(1) - share_data->predicted_size + share_data->octomap_resolution * j;
					double z = max(share_data->min_z_table, share_data->object_center_world(2) - share_data->predicted_size) + share_data->octomap_resolution * k;
					octo_model->setNodeValue(x, y, z, (float)0, true); //初始化概率0.5，即logodds为0
				}		
		octo_model->updateInnerOccupancy();
		share_data->init_entropy = 0;
		share_data->voxels_in_BBX = 0;
		for (octomap::ColorOcTree::leaf_iterator it = octo_model->begin_leafs(), end = octo_model->end_leafs(); it != end; ++it)
		{
			//cout << it.getX() - object_center_world(0) << " " << it.getY() - object_center_world(1) << " " << it.getZ() - object_center_world(2)  << endl;
			double occupancy = (*it).getOccupancy();
			share_data->init_entropy += voxel_information->entropy(occupancy);
			share_data->voxels_in_BBX++;
		}
		voxel_information->init_mutex_voxels(share_data->voxels_in_BBX);
		cout << "Map_init has voxels(in BBX) " << share_data->voxels_in_BBX << " and entropy " << share_data->init_entropy << endl;
		//share_data->access_directory(share_data->save_path+ "/quantitative");
		//ofstream fout_map(share_data->save_path+"/quantitative/Map" + to_string(-1) + ".txt");
		//fout_map << 0 << '\t' << share_data->init_entropy << '\t' << 0 << '\t' << 1 << endl;

		//在点云上根据GT计算覆盖率,根据Octomap精度统计可见点个数
		int num = 0;
		unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
		for (int j = 0; j < share_data->cloud_final->points.size(); j++) {
			octomap::OcTreeKey key = share_data->ground_truth_model->coordToKey(share_data->cloud_final->points[j].x, share_data->cloud_final->points[j].y, share_data->cloud_final->points[j].z);
			if (voxel->find(key) == voxel->end()) {
				(*voxel)[key] = num++;
			}
		}
		cout << "Cloud_init has voxels " << voxel->size() << endl;
		//ofstream fout_cloud(share_data->save_path + "/quantitative/Cloud" + to_string(-1) + ".txt");
		//fout_cloud << voxel->size() << '\t' << 1.0 * voxel->size() / share_data->GT_points_number << '\t' << 1.0 * voxel->size() / share_data->cloud_points_number<< endl;
		delete voxel;
	}

	void update(int _id, Share_Data* _share_data, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,pcl::PointCloud<pcl::PointXYZRGB>::Ptr update_cloud) {
		share_data = _share_data;
		object_changed = false;
		id = _id;
		now_camera_pose_world = share_data->now_camera_pose_world;
		//更新视点标记
		for (int i = 0; i < views.size(); i++) {
			views[i].space_id = id;
			pair<int, double> local_path = get_local_path(Eigen::Vector3d(now_camera_pose_world(0, 3), now_camera_pose_world(1, 3), now_camera_pose_world(2, 3)).eval(), views[i].init_pos.eval(), object_center_world.eval(), predicted_size * sqrt(2)); //包围盒半径是半边长的根号2倍
			if (local_path.first < 0) cout << "local path wrong." << endl;
			views[i].robot_cost = local_path.second;
		}
		//插入点云至中间数据结构
		double now_time = clock();
		double map_size = predicted_size + 3.0 * octomap_resolution;
		share_data->map_size = map_size;
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

		//在地图上，统计信息熵
		for (int i = 0; i < 32; i++)
			for (int j = 0; j < 32; j++)
				for (int k = 0; k < 32; k++)
				{
					double x = share_data->object_center_world(0) - share_data->predicted_size + share_data->octomap_resolution * i;
					double y = share_data->object_center_world(1) - share_data->predicted_size + share_data->octomap_resolution * j;
					double z = max(share_data->min_z_table, share_data->object_center_world(2) - share_data->predicted_size) + share_data->octomap_resolution * k;
					auto node = octo_model->search(x, y, z);
					if (node == NULL) cout << "what?" << endl;
					double occupancy = node->getOccupancy();
					map_entropy += voxel_information->entropy(occupancy);
					if (occupancy > 0.5 && z > share_data->min_z_table + share_data->octomap_resolution) occupied_voxels++;
				}
		share_data->access_directory(share_data->save_path + "/octomaps");
		if(share_data->is_save)	share_data->octo_model->write(share_data->save_path + "/octomaps/octomap"+to_string(id)+".ot");
		
		if (id == 0) {
			share_data->access_directory(share_data->save_path + "/quantitative");
			ofstream fout_map(share_data->save_path+"/quantitative/Map" + to_string(-1) + ".txt");
			fout_map << 0 << '\t' << share_data->init_entropy << '\t' << 0 << '\t' << 1 << endl;
		}

		if (share_data->is_save) {
			//在点云上，统计重建体素个数
			for (octomap::ColorOcTree::leaf_iterator it = share_data->ground_truth_model->begin_leafs(), end = share_data->ground_truth_model->end_leafs(); it != end; ++it) {
				share_data->cloud_model->setNodeValue(it.getKey(), octomap::logodds(1.0), true);
				share_data->cloud_model->setNodeColor(it.getKey(), 192, 192, 192);
			}
			for (int j = 0; j < share_data->cloud_final->points.size(); j++) {
				octomap::OcTreeKey key = share_data->cloud_model->coordToKey(share_data->cloud_final->points[j].x, share_data->cloud_final->points[j].y, share_data->cloud_final->points[j].z);
				share_data->cloud_model->setNodeColor(key, 255, 64, 64);
			}
			share_data->cloud_model->updateInnerOccupancy();

			share_data->access_directory(share_data->save_path + "/octocloud");
			share_data->cloud_model->write(share_data->save_path + "/octocloud/octocloud" + to_string(id) + ".ot");
		}

		cout << "Map " << id << " has voxels " << occupied_voxels << ". Map " << id << " has entropy " << map_entropy << endl;
		cout << "Map " << id << " has voxels(rate) " << 1.0 * occupied_voxels / share_data->init_voxels << ". Map " << id << " has entropy(rate) " << map_entropy / share_data->init_entropy << endl;
		share_data->access_directory(share_data->save_path+"/quantitative");
		ofstream fout_map(share_data->save_path +"/quantitative/Map" + to_string(id) + ".txt");
		fout_map << occupied_voxels << '\t' << map_entropy << '\t' << 1.0 * occupied_voxels / share_data->init_voxels << '\t' << map_entropy / share_data->init_entropy << endl;

		//在点云上根据GT计算覆盖率,根据Octomap精度统计可见点个数
		int num = 0;
		unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
		for (int j = 0; j < share_data->cloud_final->points.size(); j++) {
			octomap::OcTreeKey key = share_data->ground_truth_model->coordToKey(share_data->cloud_final->points[j].x, share_data->cloud_final->points[j].y, share_data->cloud_final->points[j].z);
			if (voxel->find(key) == voxel->end()) {
				(*voxel)[key] = num++;
			}
		}
		cout << "Cloud " << id << " has voxels " << voxel->size() << endl;
		cout << "Cloud " << id << " has voxels(rate/GT_visible) " << 1.0 * voxel->size() / share_data->GT_points_number << " voxels(rate/GT) " << 1.0 * voxel->size() / share_data->cloud_points_number << endl; 
		ofstream fout_cloud(share_data->save_path + "/quantitative/Cloud" + to_string(id) + ".txt");
		fout_cloud << voxel->size() << '\t'  << 1.0 * voxel->size() / share_data->GT_points_number << '\t' << 1.0 * voxel->size() / share_data->cloud_points_number << endl;
		delete voxel;
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
	}

};
