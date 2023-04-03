#pragma once
#include <iostream> 
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <time.h>
#include <math.h>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/opencv.hpp>

#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>

#include <gurobi_c++.h>

using namespace std;

class Ray
{
public:
	octomap::OcTreeKey origin;
	octomap::OcTreeKey end;
	octomap::KeyRay* ray_set;
	octomap::KeyRay::iterator start;
	octomap::KeyRay::iterator stop;

	Ray(octomap::OcTreeKey _origin, octomap::OcTreeKey _end, octomap::KeyRay* _ray_set, octomap::KeyRay::iterator _start,octomap::KeyRay::iterator _stop){
		origin = _origin;
		end = _end;
		ray_set = _ray_set;
		start = _start;
		stop = _stop;
	}

	~Ray() {
		;
	}

	bool operator ==(const Ray& other) const {//用于查询 		
		return (origin == other.origin) && (end == other.end); 	
	}
};

class Ray_Hash
{
public:
	size_t operator() (const Ray& ray) const {//利用6个点的double来hash
		return octomap::OcTreeKey::KeyHash()(ray.origin) ^ octomap::OcTreeKey::KeyHash()(ray.end);
	}
};

class Ray_Information
{
public:
	Ray* ray;
	double information_gain;
	double visible;
	double object_visible;
	int voxel_num;
	bool previous_voxel_unknown;

	Ray_Information(Ray* _ray) {
		ray = _ray;
		information_gain = 0;
		visible = 1;
		object_visible = 1;
		previous_voxel_unknown = false;
		voxel_num = 0;
	}

	~Ray_Information() {
		delete ray;
	}

	void clear() {
		information_gain = 0;
		visible = 1;
		object_visible = 1;
		previous_voxel_unknown = false;
		voxel_num = 0;
	}
};

//void ray_graph_thread_process(int ray_id,Ray_Information** rays_info, unordered_map<int, vector<int>>* rays_to_viwes_map, unordered_map<octomap::OcTreeKey, unordered_set<int>, octomap::OcTreeKey::KeyHash>* end_id_map, Voxel_Information* voxel_information);
void information_gain_thread_process(Ray_Information** rays_info, unordered_map<int, vector<int>>* views_to_rays_map, View_Space* view_space, int pos);
void ray_expand_thread_process(int* ray_num, Ray_Information** rays_info, unordered_map<Ray, int, Ray_Hash>* rays_map, unordered_map<int, vector<int>>* views_to_rays_map, unordered_map<int, vector<int>>* rays_to_viwes_map, octomap::ColorOcTree* octo_model, Voxel_Information* voxel_information, View_Space* view_space, rs2_intrinsics* color_intrinsics, pcl::PointCloud<pcl::PointXYZ>::Ptr frontier,int pos);
void ray_cast_thread_process(int* ray_num, Ray_Information** rays_info, unordered_map<Ray, int, Ray_Hash>* rays_map, unordered_map<int, vector<int>>* views_to_rays_map, unordered_map<int, vector<int>>* rays_to_viwes_map, octomap::ColorOcTree* octo_model, Voxel_Information* voxel_information, View_Space* view_space, rs2_intrinsics* color_intrinsics,int pos);
vector<int> get_xmax_xmin_ymax_ymin_in_hull(vector<cv::Point2f>& hull, rs2_intrinsics& color_intrinsics);
bool is_pixel_in_convex(vector<cv::Point2f>& hull, cv::Point2f& pixel);
vector<cv::Point2f> get_convex_on_image(vector<Eigen::Vector4d>& convex_3d, Eigen::Matrix4d& now_camera_pose_world, rs2_intrinsics& color_intrinsics, int& pixel_interval, double& max_range, double& octomap_resolution);
octomap::point3d project_pixel_to_ray_end(int x,int y, rs2_intrinsics& color_intrinsics, Eigen::Matrix4d& now_camera_pose_world, float max_range = 1.0);
double information_function(short& method, double& ray_informaiton, double voxel_information, double& visible, bool& is_unknown, bool& previous_voxel_unknown, bool& is_endpoint, bool& is_occupied, double& object, double& object_visible);
void ray_information_thread_process(int ray_id, Ray_Information** rays_info, unordered_map<Ray, int, Ray_Hash>* rays_map, unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash>* occupancy_map, unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash>* object_weight, octomap::ColorOcTree* octo_model, Voxel_Information* voxel_information, View_Space* view_space, short method);
int frontier_check(octomap::point3d node, octomap::ColorOcTree* octo_model, Voxel_Information* voxel_information, double octomap_resolution);
double distance_function(double distance, double alpha);

class Views_Information
{
public:
	double cost_weight;
	Ray_Information** rays_info;
	unordered_map<int, vector<int>>* views_to_rays_map;
	unordered_map<int, vector<int>>* rays_to_viwes_map;
	unordered_map<Ray,int, Ray_Hash>* rays_map;
	unordered_map<octomap::OcTreeKey,double, octomap::OcTreeKey::KeyHash>* occupancy_map;
	unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash>* object_weight;
	long long max_num_of_rays;
	int ray_num;
	double alpha;
	int K = 6;
	rs2_intrinsics color_intrinsics;
	Voxel_Information* voxel_information;
	octomap::ColorOcTree* octo_model;
	double octomap_resolution;
	int method;
	int pre_edge_cnt;
	int edge_cnt;

	~Views_Information() {
		delete views_to_rays_map;
		delete rays_to_viwes_map;
		delete rays_map;
		delete occupancy_map;
		delete object_weight;
		for (int i = 0; i < ray_num; i++)
			delete rays_info[i];
		delete rays_info;
	}

	Views_Information(Share_Data* share_data, Voxel_Information* _voxel_information , View_Space* view_space,int iterations)
	{
		//更新内部数据
		voxel_information = _voxel_information;
		cost_weight = share_data->cost_weight;
		color_intrinsics = share_data->color_intrinsics;
		method = share_data->method_of_IG;
		octo_model = share_data->octo_model;
		octomap_resolution = share_data->octomap_resolution;
		voxel_information->octomap_resolution = octomap_resolution;
		alpha = 0.1 / octomap_resolution;
		voxel_information->skip_coefficient = share_data->skip_coefficient;
		if (method == APORA) {
			K = 1;
			alpha = 2;
		}
		//注意视点需要按照id排序来建立映射
		sort(view_space->views.begin(), view_space->views.end(), view_id_compare);
		double now_time = clock();
		views_to_rays_map = new unordered_map<int, vector<int>>();
		rays_to_viwes_map = new unordered_map<int, vector<int>>();
		rays_map = new unordered_map<Ray, int, Ray_Hash>();
		object_weight = new unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash>();
		occupancy_map = new unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash>();
		//定义frontier
		vector<octomap::point3d> points;
		pcl::PointCloud<pcl::PointXYZ>::Ptr edge(new pcl::PointCloud<pcl::PointXYZ>);
		double map_size = view_space->predicted_size;
		//查找地图中的edge
		for (octomap::ColorOcTree::leaf_iterator it = octo_model->begin_leafs(), end = octo_model->end_leafs(); it != end; ++it)
		{
			double occupancy = (*it).getOccupancy();
			//记录bbx中key到occ率的映射，用于重复查询
			(*occupancy_map)[it.getKey()] = occupancy;
			if (voxel_information->is_unknown(occupancy)) {
				auto coordinate = it.getCoordinate();
				if (coordinate.x() >= view_space->object_center_world(0) - map_size && coordinate.x() <= view_space->object_center_world(0) + map_size
					&& coordinate.y() >= view_space->object_center_world(1) - map_size && coordinate.y() <= view_space->object_center_world(1) + map_size
					&& coordinate.z() >= view_space->object_center_world(2) - map_size && coordinate.z() <= view_space->object_center_world(2) + map_size)
				{
					points.push_back(coordinate);
					if (frontier_check(coordinate, octo_model, voxel_information, octomap_resolution)==2) edge->points.push_back(pcl::PointXYZ(coordinate.x(), coordinate.y(), coordinate.z()));
				}
			}
		}
		pre_edge_cnt = 0x3f3f3f3f;
		edge_cnt = edge->points.size();
		//根据最邻近frontier，计算地图中该点的是物体表面的可能性
		if (edge->points.size() != 0) {
			pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
			kdtree.setInputCloud(edge);
			std::vector<int> pointIdxNKNSearch(K);
			std::vector<float> pointNKNSquaredDistance(K);
			for (int i = 0; i < points.size(); i++)
			{
				octomap::OcTreeKey key;   bool key_have = octo_model->coordToKeyChecked(points[i], key);
				if (key_have) {
					pcl::PointXYZ searchPoint(points[i].x(), points[i].y(), points[i].z());
					int num = kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance);
					if (num > 0) {
						double p_obj = 1;
						for (int j = 0; j < pointIdxNKNSearch.size(); j++) {
							p_obj *= distance_function(pointNKNSquaredDistance[j], alpha);
						}
						(*object_weight)[key] = p_obj;
					}
				}
			}
		}
		cout << "occupancy_map is " << occupancy_map->size() << endl;
		cout << "edge is " << edge->points.size() << endl;
		cout << "object_map is " << object_weight->size() << endl;
		//根据BBX计算最多有多少射线，射线个数最多为表面积大小*体积，用于分配指针内存
		double pre_line_point = 2.0 * map_size / octomap_resolution;
		//long long superficial = ceil(5.0 * pre_line_point * pre_line_point);
		long long volume = ceil(pre_line_point * pre_line_point * pre_line_point);
		//max_num_of_rays = superficial* volume * share_data->num_of_views;
		max_num_of_rays = volume * share_data->num_of_views;
		rays_info = new Ray_Information * [max_num_of_rays];
		cout << "full rays num is " << max_num_of_rays << endl;
		//计算BBX的八个顶点，用于划定射线范围
		vector<Eigen::Vector4d> convex_3d;
		double x1 = view_space->object_center_world(0) - map_size;
		double x2 = view_space->object_center_world(0) + map_size;
		double y1 = view_space->object_center_world(1) - map_size;
		double y2 = view_space->object_center_world(1) + map_size;
		double z1 = view_space->object_center_world(2) - map_size;
		double z2 = view_space->object_center_world(2) + map_size;
		convex_3d.push_back(Eigen::Vector4d(x1, y1, z1, 1));
		convex_3d.push_back(Eigen::Vector4d(x1, y2, z1, 1));
		convex_3d.push_back(Eigen::Vector4d(x2, y1, z1, 1));
		convex_3d.push_back(Eigen::Vector4d(x2, y2, z1, 1));
		convex_3d.push_back(Eigen::Vector4d(x1, y1, z2, 1));
		convex_3d.push_back(Eigen::Vector4d(x1, y2, z2, 1));
		convex_3d.push_back(Eigen::Vector4d(x2, y1, z2, 1));
		convex_3d.push_back(Eigen::Vector4d(x2, y2, z2, 1));
		voxel_information->convex = convex_3d;
		//分配视点的射线生成器
		thread** ray_caster = new thread *[view_space->views.size()];
		//射线初始下标从0开始
		ray_num = 0;
		for (int i = 0; i < view_space->views.size(); i++) {
			//对该视点分发生成射线的线程
			ray_caster[i] = new thread(ray_cast_thread_process, &ray_num, rays_info, rays_map, views_to_rays_map, rays_to_viwes_map, octo_model, voxel_information, view_space, &color_intrinsics, i);
		}
		//等待每个视点射线生成器计算完成
		for (int i = 0; i < view_space->views.size(); i++) {
			(*ray_caster[i]).join();
		}
		//释放内存
		for (int i = 0; i < view_space->views.size(); i++) {
			ray_caster[i]->~thread();
		}
		delete ray_caster;
		cout << "ray_num is " << ray_num << endl;
		cout << "All views' rays generated with executed time " << clock() - now_time << " ms. Startring compution." << endl;
		//为每条射线分配一个线程
		now_time = clock();
		thread** rays_process = new thread* [ray_num];
		for (int i = 0; i < ray_num; i++) {
			rays_process[i] = new thread(ray_information_thread_process, i, rays_info, rays_map, occupancy_map, object_weight, octo_model, voxel_information, view_space, method);
		}
		//等待射线计算完成
		for (int i = 0; i < ray_num; i++) {
			(*rays_process[i]).join();	
		}
		//释放内存
		for (int i = 0; i < ray_num; i++) {
			rays_process[i]->~thread();
		}
		delete rays_process;
		double cost_time = clock() - now_time;
		cout << "All rays' threads over with executed time " << cost_time << " ms." << endl;
		share_data->access_directory(share_data->save_path + "/run_time");
		ofstream fout(share_data->save_path + "/run_time/IG" + to_string(view_space->id) + ".txt");
		fout << cost_time << endl;
		//分配视点的信息统计器
		now_time = clock();
		thread** view_gain = new thread * [view_space->views.size()];
		for (int i = 0; i < view_space->views.size(); i++) {
			//对该视点分发信息统计的线程
			view_gain[i] = new thread(information_gain_thread_process, rays_info, views_to_rays_map, view_space, i);
		}
		//等待每个视点信息统计完成
		for (int i = 0; i < view_space->views.size(); i++) {
			(*view_gain[i]).join();
		}
		//释放内存
		for (int i = 0; i < view_space->views.size(); i++) {
			view_gain[i]->~thread();
		}
		delete view_gain;
		cout << "All views' gain threads over with executed time " << clock() - now_time << " ms." << endl;
	}
	
	void update(Share_Data* share_data, View_Space* view_space,int iterations) {
		//更新内部数据
		double now_time = clock();
		double map_size = view_space->predicted_size;
		//注意视点需要按照id排序来建立映射
		sort(view_space->views.begin(), view_space->views.end(), view_id_compare);
		//重新记录八叉树
		octo_model = share_data->octo_model;
		octomap_resolution = share_data->octomap_resolution;
		alpha = 0.1 / octomap_resolution;
		voxel_information->octomap_resolution = octomap_resolution;
		voxel_information->skip_coefficient = share_data->skip_coefficient;
		if (method == APORA) {
			K = 1;
			alpha = 2;
		}
		//清空视点信息
		for (int i = 0; i < view_space->views.size(); i++) {
			view_space->views[i].information_gain = 0;
			view_space->views[i].voxel_num = 0;
		}
		//避免重复search
		delete occupancy_map;
		occupancy_map = new unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash>();
		delete object_weight;
		object_weight = new unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash>();
		//更新frontier
		vector<octomap::point3d> points;
		pcl::PointCloud<pcl::PointXYZ>::Ptr edge(new pcl::PointCloud<pcl::PointXYZ>);
		for (octomap::ColorOcTree::leaf_iterator it = octo_model->begin_leafs(), end = octo_model->end_leafs(); it != end; ++it)
		{
			double occupancy = (*it).getOccupancy();
			(*occupancy_map)[it.getKey()] = occupancy;
			if (voxel_information->is_unknown(occupancy)) {
				auto coordinate = it.getCoordinate();
				if (coordinate.x() >= view_space->object_center_world(0) - map_size && coordinate.x() <= view_space->object_center_world(0) + map_size
					&& coordinate.y() >= view_space->object_center_world(1) - map_size && coordinate.y() <= view_space->object_center_world(1) + map_size
					&& coordinate.z() >= view_space->object_center_world(2) - map_size && coordinate.z() <= view_space->object_center_world(2) + map_size)
				{
					points.push_back(coordinate);
					if (frontier_check(coordinate, octo_model, voxel_information, octomap_resolution)==2) edge->points.push_back(pcl::PointXYZ(coordinate.x(), coordinate.y(), coordinate.z()));
				}
			}
		}
		edge_cnt = edge->points.size();
		if (edge_cnt > pre_edge_cnt) pre_edge_cnt = 0x3f3f3f3f;
		if (edge->points.size() != 0) {
			//计算frontier
			pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
			kdtree.setInputCloud(edge);
			std::vector<int> pointIdxNKNSearch(K);
			std::vector<float> pointNKNSquaredDistance(K);
			for (int i = 0; i < points.size(); i++)
			{
				octomap::OcTreeKey key;   bool key_have = octo_model->coordToKeyChecked(points[i], key);
				if (key_have) {
					pcl::PointXYZ searchPoint(points[i].x(), points[i].y(), points[i].z());
					int num = kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance);
					if (num > 0) {
						double p_obj = 1;
						for (int j = 0; j < pointIdxNKNSearch.size(); j++) {
							p_obj *= distance_function(pointNKNSquaredDistance[j], alpha);
						}
						(*object_weight)[key] = p_obj;
					}
				}
			}
		}
		cout << "edge is " << edge->points.size() << endl;
		cout << "object_map is " << object_weight->size() << endl;
		cout << "occupancy_map is " << occupancy_map->size() << endl;
		cout << "frontier updated with executed time " << clock() - now_time << " ms." << endl;
		//检测是否重生成
		now_time = clock();
		bool regenerate = false;
		if (view_space->object_changed) {
			regenerate = true;
		}
		//如果重生成，则更新数据结构
		if (regenerate) {
			//重计算最大射线数量，从0开始
			double pre_line_point = 2.0 * map_size / octomap_resolution;
			//long long superficial = ceil(5.0 * pre_line_point * pre_line_point);
			long long volume = ceil(pre_line_point * pre_line_point * pre_line_point);
			//max_num_of_rays = superficial* volume * share_data->num_of_views;
			max_num_of_rays = volume * share_data->num_of_views;
			delete[] rays_info;
			rays_info = new Ray_Information * [max_num_of_rays];
			cout << "full rays num is " << max_num_of_rays << endl;
			ray_num = 0;
			delete views_to_rays_map;
			views_to_rays_map = new unordered_map<int, vector<int>>();
			delete rays_to_viwes_map;
			rays_to_viwes_map = new unordered_map<int, vector<int>>();
			delete rays_map;
			rays_map = new unordered_map<Ray, int, Ray_Hash>();

			//计算BBX的八个顶点，用于划定射线范围
			vector<Eigen::Vector4d> convex_3d;
			double x1 = view_space->object_center_world(0) - map_size;
			double x2 = view_space->object_center_world(0) + map_size;
			double y1 = view_space->object_center_world(1) - map_size;
			double y2 = view_space->object_center_world(1) + map_size;
			double z1 = view_space->object_center_world(2) - map_size;
			double z2 = view_space->object_center_world(2) + map_size;
			convex_3d.push_back(Eigen::Vector4d(x1, y1, z1, 1));
			convex_3d.push_back(Eigen::Vector4d(x1, y2, z1, 1));
			convex_3d.push_back(Eigen::Vector4d(x2, y1, z1, 1));
			convex_3d.push_back(Eigen::Vector4d(x2, y2, z1, 1));
			convex_3d.push_back(Eigen::Vector4d(x1, y1, z2, 1));
			convex_3d.push_back(Eigen::Vector4d(x1, y2, z2, 1));
			convex_3d.push_back(Eigen::Vector4d(x2, y1, z2, 1));
			convex_3d.push_back(Eigen::Vector4d(x2, y2, z2, 1));
			voxel_information->convex = convex_3d;
			//分配视点的射线生成器
			thread** ray_caster = new thread * [view_space->views.size()];
			for (int i = 0; i < view_space->views.size(); i++) {
				//对该视点分发生成射线的线程
				ray_caster[i] = new thread(ray_cast_thread_process, &ray_num, rays_info, rays_map, views_to_rays_map, rays_to_viwes_map, octo_model, voxel_information, view_space, &color_intrinsics, i);
			}
			//等待每个视点射线生成器计算完成
			for (int i = 0; i < view_space->views.size(); i++) {
				(*ray_caster[i]).join();
			}
			//释放内存
			for (int i = 0; i < view_space->views.size(); i++) {
				ray_caster[i]->~thread();
			}
			delete ray_caster;
			cout << "ray_num is " << ray_num << endl;
			cout << "All views' rays generated with executed time " << clock() - now_time << " ms. Startring compution." << endl;
		}
		//为每条射线分配一个线程
		now_time = clock();
		thread** rays_process = new thread * [ray_num];
		for (int i = 0; i < ray_num; i++) {
			rays_info[i]->clear();
			rays_process[i] = new thread(ray_information_thread_process, i, rays_info, rays_map, occupancy_map, object_weight, octo_model, voxel_information, view_space, method);
		}
		//等待射线计算完成
		for (int i = 0; i < ray_num; i++) {
			(*rays_process[i]).join();
		}
		//释放内存
		for (int i = 0; i < ray_num; i++) {
			rays_process[i]->~thread();
		}
		delete rays_process;
		double cost_time = clock() - now_time;
		cout << "All rays' threads over with executed time " << cost_time << " ms." << endl;
		share_data->access_directory(share_data->save_path + "/run_time");
		ofstream fout(share_data->save_path + "/run_time/IG" + to_string(view_space->id) + ".txt");
		fout << cost_time << endl;
		now_time = clock();
		thread** view_gain = new thread * [view_space->views.size()];
		for (int i = 0; i < view_space->views.size(); i++) {
			//对该视点分发信息统计的线程
			view_gain[i] = new thread(information_gain_thread_process, rays_info, views_to_rays_map, view_space, i);
		}
		//等待每个视点信息统计完成
		for (int i = 0; i < view_space->views.size(); i++) {
			(*view_gain[i]).join();
		}
		//释放内存
		for (int i = 0; i < view_space->views.size(); i++) {
			view_gain[i]->~thread();
		}
		delete view_gain;
		cout << "All views' gain threads over with executed time " << clock() - now_time << " ms." << endl;
	}
};

void information_gain_thread_process(Ray_Information** rays_info, unordered_map<int, vector<int>>* views_to_rays_map, View_Space* view_space, int pos) {
	//视点的每个相关射线信息加入视点
	for (vector<int>::iterator it = (*views_to_rays_map)[pos].begin(); it != (*views_to_rays_map)[pos].end(); it++) {
		view_space->views[pos].information_gain += rays_info[*it]->information_gain;
		view_space->views[pos].voxel_num += rays_info[*it]->voxel_num;
	}
}

void ray_cast_thread_process(int* ray_num, Ray_Information** rays_info, unordered_map<Ray, int, Ray_Hash>* rays_map, unordered_map<int, vector<int>>* views_to_rays_map, unordered_map<int, vector<int>>* rays_to_viwes_map, octomap::ColorOcTree* octo_model, Voxel_Information* voxel_information, View_Space* view_space, rs2_intrinsics* color_intrinsics, int pos) {
	//获取视点位姿
	view_space->views[pos].get_next_camera_pos(view_space->now_camera_pose_world, view_space->object_center_world);
	Eigen::Matrix4d view_pose_world = (view_space->now_camera_pose_world * view_space->views[pos].pose.inverse()).eval();
	//将三维物体BBX根据视点位姿，投影至图片凸包区域
	double skip_coefficient = voxel_information->skip_coefficient;
	//根据能访问的体素来控制射线遍历，注意间隔跳跃参数
	int pixel_interval = color_intrinsics->width;
	//double max_range = 6.0 * view_space->predicted_size;
	double max_range = 1.0;
	vector<cv::Point2f> hull;
	hull = get_convex_on_image(voxel_information->convex, view_pose_world, *color_intrinsics, pixel_interval, max_range, voxel_information->octomap_resolution);
	//if (hull.size() != 4 && hull.size() != 5 && hull.size() != 6) cout << "hull wrong with size " << hull.size() << endl;
	//计算凸包的包围盒
	vector<int> boundary;
	boundary = get_xmax_xmin_ymax_ymin_in_hull(hull, *color_intrinsics);
	int xmax = boundary[0];
	int xmin = boundary[1];
	int ymax = boundary[2];
	int ymin = boundary[3];
	//cout << xmax << " " << xmin << " " << ymax << " " << ymin << " ," << pixel_interval <<endl;
	//中间数据结构
	vector<Ray*> rays;
	//int num = 0;
	//检查视点的key
	octomap::OcTreeKey key_origin;
	bool key_origin_have = octo_model->coordToKeyChecked(view_space->views[pos].init_pos(0), view_space->views[pos].init_pos(1), view_space->views[pos].init_pos(2), key_origin);
	if (key_origin_have) {
		octomap::point3d origin = octo_model->keyToCoord(key_origin);
		//遍历包围盒
		//srand(pos);
		//int rr = rand() % 256, gg = rand() % 256, bb = rand() % 256;
		for (int x = xmin; x <= xmax; x += (int)(pixel_interval * skip_coefficient))
			for (int y = ymin; y <= ymax; y += (int)(pixel_interval * skip_coefficient))
			{
				//num++;
				cv::Point2f pixel(x, y);
				//检查是否在凸包区域内部
				if (!is_pixel_in_convex(hull, pixel)) continue;
				//反向投影找到终点
				octomap::point3d end = project_pixel_to_ray_end(x, y, *color_intrinsics, view_pose_world, max_range);
				//显示一下
				//view_space->viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(origin(0), origin(1), origin(2)), pcl::PointXYZ(end(0), end(1), end(2)), rr, gg, bb, "line" + to_string(pos) + "-" + to_string(x) + "-" + to_string(y));
				octomap::OcTreeKey key_end;
				octomap::point3d direction = end - origin;
				octomap::point3d end_point;
				//越过未知区域，找到终点
				bool found_end_point = octo_model->castRay(origin, direction, end_point, true, max_range);
				if (!found_end_point) {//未找到终点，设置终点为最大距离
					end_point = origin + direction.normalized() * max_range; // use max range instead of stopping at the unknown       found_endpoint = true;     
				}
				//检查一下末端是否在地图限制范围内，且命中BBX
				bool key_end_have = octo_model->coordToKeyChecked(end_point, key_end);
				if (key_end_have) {
					//生成射线
					octomap::KeyRay* ray_set = new octomap::KeyRay();
					//获取射线数组，不包含末节点
					bool point_on_ray_getted = octo_model->computeRayKeys(origin, end_point, *ray_set);
					if (!point_on_ray_getted) cout << "Warning. ray cast with wrong max_range." << endl;
					if (ray_set->size() > 950) cout << ray_set->size() << " rewrite the vector size in octreekey.h." << endl;
					//把终点放入射线组
					ray_set->addKey(key_end);
					//第一个非空节点作为射线起点，尾巴开始最后一个非空元素作为射线终点
					octomap::KeyRay::iterator last = ray_set->end();
					last--;
					while (last != ray_set->begin() && (octo_model->search(*last) == NULL)) last--;
					//二分第一个非空元素
					octomap::KeyRay::iterator l = ray_set->begin();
					octomap::KeyRay::iterator r = last;
					octomap::KeyRay::iterator mid = l + (r - l) / 2;
					while (mid != r) {
						if (octo_model->search(*mid) != NULL)
							r = mid;
						else
							l = mid + 1;
						mid = l + (r - l) / 2;
					}
					octomap::KeyRay::iterator first = mid;
					while (first  != ray_set->end() && (octo_model->keyToCoord(*first).x() < view_space->object_center_world(0) - view_space->predicted_size || octo_model->keyToCoord(*first).x() > view_space->object_center_world(0) + view_space->predicted_size
						|| octo_model->keyToCoord(*first).y() < view_space->object_center_world(1) - view_space->predicted_size || octo_model->keyToCoord(*first).y() > view_space->object_center_world(1) + view_space->predicted_size
						|| octo_model->keyToCoord(*first).z() < view_space->object_center_world(2) - view_space->predicted_size || octo_model->keyToCoord(*first).z() > view_space->object_center_world(2) + view_space->predicted_size)) first++;
					//如果没有非空元素，直接丢弃射线
					if (last - first < 0) {
						delete ray_set;
						continue;
					}
					octomap::KeyRay::iterator stop = last;
					stop++;
					//显示一下
					//while (octo_model->keyToCoord(*first).x() < view_space->object_center_world(0) - view_space->predicted_size || octo_model->keyToCoord(*first).x() > view_space->object_center_world(0) + view_space->predicted_size
					//	|| octo_model->keyToCoord(*first).y() < view_space->object_center_world(1) - view_space->predicted_size || octo_model->keyToCoord(*first).y() > view_space->object_center_world(1) + view_space->predicted_size
					//	|| octo_model->keyToCoord(*first).z() < min(view_space->height_of_ground, view_space->object_center_world(2) - view_space->predicted_size) || octo_model->keyToCoord(*first).z() > view_space->object_center_world(2) + view_space->predicted_size) first++;
					//octomap::point3d ss = octo_model->keyToCoord(*first);
					//octomap::point3d ee = octo_model->keyToCoord(*last);
					//view_space->viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(ss(0), ss(1), ss(2)), pcl::PointXYZ(ee(0), ee(1), ee(2)), rr, gg, bb, "line" + to_string(pos) + "-" + to_string(x) + "-" + to_string(y));
					//将射线加入视点的集合，第一个元素与最后一个元素key+数组+头+尾
					Ray* ray = new Ray(*first, *last, ray_set, first, stop);
					rays.push_back(ray);
				}
			}
	}
	else {
		cout << pos << "th view out of map.check." << endl;
	}
	//cout << "rays " << rays.size() <<" num "<<num<< endl;
	//该视点射线下标的数组
	vector<int> ray_ids;
	ray_ids.resize(rays.size());
	//注意公用的数据结构要加锁
	voxel_information->mutex_rays.lock();
	//获取当前射线的位置
	int ray_id = (*ray_num);
	for (int i = 0; i < rays.size(); i++) {
		//对于这些射线，hash查询一下是否有重复的
		auto hash_this_ray = rays_map->find(*rays[i]);
		//如果没有重复的，就保存该射线
		if (hash_this_ray == rays_map->end()) {
			(*rays_map)[*rays[i]] = ray_id;
			ray_ids[i] = ray_id;
			//创造射线计算类
			rays_info[ray_id] = new Ray_Information(rays[i]);
			vector<int> view_ids;
			view_ids.push_back(pos);
			(*rays_to_viwes_map)[ray_id] = view_ids;
			ray_id++;
		}
		//如果有重复的，说明其他视点也算到了该射线，就把相应的id放入下标数组
		else {
			ray_ids[i] = hash_this_ray->second;
			delete rays[i]->ray_set;
			//其他视点已经记录的射线，把本视点的记录放进去
			vector<int> view_ids = (*rays_to_viwes_map)[ray_ids[i]];
			view_ids.push_back(pos);
			(*rays_to_viwes_map)[ray_ids[i]] = view_ids;
		}
	}
	//更新射线数目
	(*ray_num) = ray_id;
	//更新视点映射的射线数组
	(*views_to_rays_map)[pos] = ray_ids;
	//释放锁
	voxel_information->mutex_rays.unlock();
}

inline vector<int> get_xmax_xmin_ymax_ymin_in_hull(vector<cv::Point2f>& hull, rs2_intrinsics& color_intrinsics) {
	float xmax = 0, xmin = color_intrinsics.width - 1, ymax = 0, ymin = color_intrinsics.height - 1;
	for(int i = 0;i< hull.size();i++){
		xmax = max(xmax, hull[i].x);
		xmin = min(xmin, hull[i].x);
		ymax = max(ymax, hull[i].y);
		ymin = min(ymin, hull[i].y);
	}
	vector<int> boundary;
	boundary.push_back((int)floor(xmax));
	boundary.push_back((int)floor(xmin));
	boundary.push_back((int)floor(ymax));
	boundary.push_back((int)floor(ymin));
	return boundary;
}

inline bool is_pixel_in_convex(vector<cv::Point2f>& hull, cv::Point2f& pixel) {
	double hull_value = pointPolygonTest(hull, pixel, false);
	return hull_value >= 0;
}

inline vector<cv::Point2f> get_convex_on_image(vector<Eigen::Vector4d>& convex_3d, Eigen::Matrix4d& now_camera_pose_world, rs2_intrinsics& color_intrinsics,int& pixel_interval,double& max_range,double& octomap_resolution) {
	//投影立方体顶点至图像坐标系
	double now_range = 0;
	vector<cv::Point2f> contours;
	for (int i = 0; i < convex_3d.size(); i++) {
		Eigen::Vector4d vertex = now_camera_pose_world.inverse() * convex_3d[i];
		float point[3] = { vertex(0), vertex(1),vertex(2) };
		float pixel[2];
		rs2_project_point_to_pixel(pixel, &color_intrinsics, point);
		contours.push_back(cv::Point2f(pixel[0], pixel[1]));
		//cout << pixel[0] << " " << pixel[1] << endl;
		//计算一下最远点离开视点距离
		Eigen::Vector4d view_pos(now_camera_pose_world(0, 3), now_camera_pose_world(1, 3), now_camera_pose_world(2, 3), 1);
		now_range = max(now_range, (view_pos - convex_3d[i]).norm());
	}
	max_range = min(max_range, now_range);
	//计算凸包
	vector<cv::Point2f> hull;
	convexHull(contours, hull, false, true);
	if (!cv::isContourConvex(hull)) {
		cout << "no convex. check BBX." << endl;
		return contours;
	}
	//计算空间最远两点距离，计算像素最远两点距离，根据地图分辨率得到像素偏移
	double pixel_dis = 0;
	double space_dis = 0;
	for (int i = 0; i < hull.size(); i++)
		for (int j = 0; j < hull.size(); j++) if(i!=j){
			Eigen::Vector2d pixel_start(hull[i].x, hull[i].y);
			Eigen::Vector2d pixel_end(hull[j].x, hull[j].y);
			pixel_dis = max(pixel_dis,(pixel_start - pixel_end).norm());
			space_dis = max(space_dis, (convex_3d[i] - convex_3d[j]).norm());
		}
	pixel_interval = (int)(pixel_dis / space_dis * octomap_resolution);
	return hull;
}

inline octomap::point3d project_pixel_to_ray_end(int x,int y, rs2_intrinsics& color_intrinsics, Eigen::Matrix4d& now_camera_pose_world,float max_range) {
	float pixel[2] = { x ,y };
	float point[3];
	rs2_deproject_pixel_to_point(point, &color_intrinsics, pixel, max_range);
	Eigen::Vector4d point_world(point[0], point[1], point[2],1);
	point_world = now_camera_pose_world * point_world;
	return octomap::point3d(point_world(0), point_world(1), point_world(2));
}

void ray_information_thread_process(int ray_id, Ray_Information** rays_info, unordered_map<Ray, int, Ray_Hash>* rays_map, unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash>* occupancy_map, unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash>* object_weight, octomap::ColorOcTree* octo_model, Voxel_Information* voxel_information, View_Space* view_space, short method )
{
	//由于检查过，所以第一个节点就是非空节点
	octomap::KeyRay::iterator first = rays_info[ray_id]->ray->start;
	octomap::KeyRay::iterator last = rays_info[ray_id]->ray->stop;
	last--;
	for (octomap::KeyRay::iterator it = rays_info[ray_id]->ray->start; it != rays_info[ray_id]->ray->stop; ++it) {
		//从hash表里查询该key
		auto hash_this_key = (*occupancy_map).find(*it);
		//找不到节点就下一个
		if (hash_this_key== (*occupancy_map).end()) {
			if (method == RSE && it == last) rays_info[ray_id]->information_gain = 0;
			continue;
		}
		//读取节点概率值
		double occupancy = hash_this_key->second;
		//检查一下当前节点是否被占据
		bool voxel_occupied = voxel_information->is_occupied(occupancy);
		//检查一下节点是否未知
		bool voxel_unknown = voxel_information->is_unknown(occupancy);
		//读取一下节点为物体表面率
		double on_object = voxel_information->voxel_object(*it, object_weight);
		//如果被占据，那就是最后一个节点了
		if (voxel_occupied) last = it;
		//如果free，则初始节点要更新
		if (it==first && (!voxel_unknown&&!voxel_occupied)) first = it;
		//判断是否最后一个节点
		bool is_end = (it == last);
		//统计信息熵
		rays_info[ray_id]->information_gain = information_function(method, rays_info[ray_id]->information_gain, voxel_information->entropy(occupancy), rays_info[ray_id]->visible, voxel_unknown, rays_info[ray_id]->previous_voxel_unknown, is_end, voxel_occupied, on_object, rays_info[ray_id]->object_visible);
		rays_info[ray_id]->object_visible *= (1 - on_object);
		if (method == MCMF || method == GMC) rays_info[ray_id]->visible *= voxel_information->get_voxel_visible(occupancy);
		else rays_info[ray_id]->visible *= (1 - occupancy);
		rays_info[ray_id]->voxel_num++;
		//如果是最后了就退出
		if (is_end) break;
	}
	while (last - first < -1) first--;
	last++;
	//更新stop为最后一个节点后一个迭代器
	rays_info[ray_id]->ray->stop = last;
	//更新start为第一个迭代器
	rays_info[ray_id]->ray->start = first;
}

inline double information_function(short& method,double& ray_informaiton,double voxel_information,double& visible,bool& is_unknown, bool& previous_voxel_unknown,bool& is_endpoint,bool& is_occupied,double& object,double& object_visible) {
	double final_information = 0;
	switch (method) {
	case GMC:
		if (is_unknown) {
			final_information = ray_informaiton + object * visible * voxel_information;
		}
		else {
			final_information = ray_informaiton;
		}
		break;
	case MCMF:
		if (is_unknown) {
			final_information = ray_informaiton + object * visible * voxel_information;
		}
		else {
			final_information = ray_informaiton;
		}
		break;
	case OA:
		final_information = ray_informaiton + visible * voxel_information;
		break;
	case UV:
		if(is_unknown) final_information = ray_informaiton + visible * voxel_information;
		else final_information = ray_informaiton;
		break;
	case RSE:
		if (is_endpoint) {
			if (previous_voxel_unknown) {
				if (is_occupied) final_information = ray_informaiton + visible * voxel_information;
				else final_information = 0;
			}
			else final_information = 0;
		}
		else {
			if (is_unknown) {
				previous_voxel_unknown = true;
				final_information = ray_informaiton + visible * voxel_information;
			}
			else {
				previous_voxel_unknown = false;
				final_information = 0;
			}
		}
		break;
	case APORA:
		if (is_unknown) {
			final_information = ray_informaiton + object * object_visible * voxel_information;
		}
		else {
			final_information = ray_informaiton;
		}
		break;
	case Kr:
		if (is_endpoint) {
			if (is_occupied) final_information = ray_informaiton + voxel_information;
			else final_information = 0;
		}
		else final_information = ray_informaiton + voxel_information;
		break;
	}
	return final_information;
}

inline int frontier_check(octomap::point3d node, octomap::ColorOcTree* octo_model, Voxel_Information* voxel_information, double octomap_resolution) {
	int free_cnt = 0;
	int occupied_cnt = 0;
	for (int i = -1; i <= 1; i++)
		for (int j = -1; j <= 1; j++)
			for (int k = -1; k <= 1; k++) 
			{
				if (i == 0 && j == 0 && k == 0) continue;
				double x = node.x() + i * octomap_resolution;
				double y = node.y() + j * octomap_resolution;
				double z = node.z() + k * octomap_resolution;
				octomap::point3d neighbour(x, y, z);
				octomap::OcTreeKey neighbour_key;  bool neighbour_key_have = octo_model->coordToKeyChecked(neighbour, neighbour_key);
				if (neighbour_key_have) {
					octomap::ColorOcTreeNode* neighbour_voxel = octo_model->search(neighbour_key);
					if (neighbour_voxel != NULL) {
						free_cnt += voxel_information->voxel_free(neighbour_voxel) == true ? 1 : 0;
						occupied_cnt += voxel_information->voxel_occupied(neighbour_voxel) == true ? 1 : 0;
					}
				}
			}
	//edge
	if (free_cnt >= 1 && occupied_cnt >= 1) return 2;
	//边界
	if (free_cnt >= 1) return 1;
	//啥也不是
	return 0;
}

inline double distance_function(double distance,double alpha) {
	return exp(-pow2(alpha)*distance);
}

//Max Flow
class MCMF_solver {
public:
	struct Edge {
		int from, to, cap, flow;
		double cost;
		Edge(int u, int v, int c, int f, double w)
			: from(u), to(v), cap(c), flow(f), cost(w) {}
	};

	const int INF = 0x3f3f3f3f;
	int n, m;
	vector<Edge> edges;
	vector<vector<int>> G;
	vector<int> inq, p, a;
	vector<double> d;
	double eps = 1e-3;
	bool isZero(const double& val) { return abs(val) < eps; }

	void init(int n) {
		this->n = n;
		G.resize(n);
		d.resize(n);
		a.resize(n);
		p.resize(n);
		inq.resize(n);

		for (int i = 0; i < n; i++) {
			G[i].clear();
			d[i] = a[i] = p[i] = inq[i] = 0;
		}
		edges.clear();
	}

	void AddEdge(int from, int to, int cap, double cost) {
		//cerr << from << ' ' << to << ' ' << cap << ' ' << cost << endl;
		edges.push_back(Edge(from, to, cap, 0, cost));
		edges.push_back(Edge(to, from, 0, 0, -cost));
		m = edges.size();
		G[from].push_back(m - 2);
		G[to].push_back(m - 1);
	}

	bool BellmanFord(int s, int t, int& flow, double& cost) {
		for (int i = 0; i < n; i++) d[i] = INF;
		for (int i = 0; i < n; i++) inq[i] = 0;
		d[s] = 0;
		inq[s] = 1;
		p[s] = 0;
		a[s] = INF;
		queue<int> Q;
		Q.push(s);
		while (!Q.empty()) {
			int u = Q.front();
			Q.pop();
			inq[u] = 0;
			for (int i = 0; i < G[u].size(); i++) {
				Edge& e = edges[G[u][i]];
				if (e.cap > e.flow && d[e.to] > d[u] + e.cost) {
					d[e.to] = d[u] + e.cost;
					p[e.to] = G[u][i];
					a[e.to] = min(a[u], e.cap - e.flow);
					if (!inq[e.to]) {
						Q.push(e.to);
						inq[e.to] = 1;
					}
				}
			}
		}
		if (d[t] == INF) return false;  // 当没有可增广的路时退出
		flow += a[t];
		cost += d[t] * a[t];
		for (int u = t; u != s; u = edges[p[u]].from) {
			edges[p[u]].flow += a[t];
			edges[p[u] ^ 1].flow -= a[t];
		}
		return true;
	}

	vector<int> work(const vector<vector<pair<int, double>>>& vec) {
		int nn = vec.size();
		int S = nn, T = nn + 1;
		init(T + 2);

		vector<bool> vis(nn);
		for (int u = 0; u < nn; u++) {
			for (auto& e : vec[u]) {
				int v = e.first;
				double w = e.second;
				if (!isZero(w)) {
					if (!vis[u]) {
						AddEdge(S, u, 1, 0);
						vis[u] = true;
					}
				}
				else {
					if (!vis[v]) {
						AddEdge(v, T, INF, 0);
						vis[v] = true;
					}
				}
				AddEdge(u, v, INF, -w);
			}
		}
		int flow = 0;
		double cost = 0;
		while (BellmanFord(S, T, flow, cost))
			;

		//cerr << "flow = " << flow << endl;
		//cerr << "cost = " << cost << endl;

		vector<int> ret;
		//cerr << "ret: ";
		for (auto& e : edges)
			if (e.to == T)
				if (e.flow > 0) {
					ret.push_back(e.from);
					//cerr << "(" << e.from << ", " << e.flow << ") ";
				}
		//cerr << endl;

		return ret;
	}
};

/*
solving by Max Flow
*/

void adjacency_list_MF_thread_process(int ray_id, int* ny, int ray_index_shift, int voxel_index_shift, unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel_id_map, vector<vector<pair<int, double>>>* bipartite_list, View_Space* view_space, Views_Information* views_information, Voxel_Information* voxel_information, Share_Data* share_data);

class views_voxels_MF {
public:
	int nx, ny, nz;										//三边的点数，视点数nx，射线数ny,体素数nz
	vector<vector<pair<int, double>>>* bipartite_list;	//邻接表
	View_Space* view_space;
	Views_Information* views_information;
	Voxel_Information* voxel_information;
	Share_Data* share_data;
	unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel_id_map;	//体素下标
	MCMF_solver* mcmf;
	vector<int> view_id_set;

	void solve() {
		double now_time = clock();
		view_id_set = mcmf->work(*bipartite_list);
		double cost_time = clock() - now_time;
		cout << "flow network solved with executed time " << cost_time << " ms." << endl;
		cout << view_id_set.size() << " views getted by max flow." << endl;
		share_data->access_directory(share_data->save_path + "/run_time");
		ofstream fout(share_data->save_path + "/run_time/GO" + to_string(view_space->id) + ".txt");
		fout << cost_time << '\t' << view_id_set.size() << endl;
	}

	vector<int> get_view_id_set() {
		return view_id_set;
	}

	views_voxels_MF(int _nx, View_Space* _view_space, Views_Information* _views_information, Voxel_Information* _voxel_information, Share_Data* _share_data) {
		double now_time = clock();
		view_space = _view_space;
		views_information = _views_information;
		voxel_information = _voxel_information;
		share_data = _share_data;
		//视点按照id排序，并建立三分图邻接表
		sort(view_space->views.begin(), view_space->views.end(), view_id_compare);
		nx = _nx;
		ny = views_information->ray_num;
		bipartite_list = new vector<vector<pair<int, double>>>;
		bipartite_list->resize(nx + ny + share_data->voxels_in_BBX);
		//建立体素的id表
		voxel_id_map = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>;
		//并行遍历每条射线上的体素累加至对应视点
		nz = 0;
		thread** adjacency_list_process = new thread * [views_information->ray_num];
		for (int i = 0; i < views_information->ray_num; i++) {
			adjacency_list_process[i] = new thread(adjacency_list_MF_thread_process, i, &nz, nx, nx + ny, voxel_id_map, bipartite_list, view_space, views_information, voxel_information, share_data);
		}
		for (int i = 0; i < views_information->ray_num; i++) {
			(*adjacency_list_process[i]).join();
		}
		//释放内存
		for (int i = 0; i < views_information->ray_num; i++) {
			adjacency_list_process[i]->~thread();
		}
		delete adjacency_list_process;
		//输出一下具体的图大小
		if (nz != voxel_id_map->size()) cout << "node_z wrong." << endl;
		int num_of_all_edge = 0;
		int num_of_view_edge = 0;
		for (int i = 0; i < bipartite_list->size(); i++) {
			num_of_all_edge += (*bipartite_list)[i].size();
			if (i > nx && i < nx + ny) num_of_view_edge += (*bipartite_list)[i].size();
		}
		cout << "Full edge is " << num_of_all_edge << ". View edge(in) is " << num_of_view_edge << ". Voexl edge(out) is "<< num_of_all_edge - num_of_view_edge<< "."<< endl;
		cout << "adjacency list with interested voxels num " << ny << " getted with executed time " << clock() - now_time << " ms." << endl;
		mcmf = new MCMF_solver();
	}

	~views_voxels_MF() {
		delete bipartite_list;
		delete voxel_id_map;
		delete mcmf;
	}
};

void adjacency_list_MF_thread_process(int ray_id, int* nz, int ray_index_shift, int voxel_index_shift, unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel_id_map, vector<vector<pair<int, double>>>* bipartite_list, View_Space* view_space, Views_Information* views_information, Voxel_Information* voxel_information, Share_Data* share_data) {
	//该射线被哪些视点看到，加入图中
	vector<int> views_id = (*views_information->rays_to_viwes_map)[ray_id];
	for (int i = 0; i < views_id.size(); i++)
		(*bipartite_list)[ray_id + ray_index_shift].push_back(make_pair(views_id[i], 0.0));
	//仅保留感兴趣体素
	double visible = 1.0;
	octomap::KeyRay::iterator first = views_information->rays_info[ray_id]->ray->start;
	octomap::KeyRay::iterator last = views_information->rays_info[ray_id]->ray->stop;
	for (octomap::KeyRay::iterator it = views_information->rays_info[ray_id]->ray->start; it != views_information->rays_info[ray_id]->ray->stop; ++it) {
		//从hash表里查询该key
		auto hash_this_key = (*views_information->occupancy_map).find(*it);
		//找不到节点就下一个
		if (hash_this_key == (*views_information->occupancy_map).end()) continue;
		//读取节点概率值
		double occupancy = hash_this_key->second;
		//读取一下节点为物体表面率
		double on_object = voxel_information->voxel_object(*it, views_information->object_weight);
		//统计信息熵
		double information_gain = on_object * visible * voxel_information->entropy(occupancy);
		visible *= voxel_information->get_voxel_visible(occupancy);
		if (information_gain > share_data->interesting_threshold) {
			octomap::OcTreeKey node_y = *it;
			int voxel_id;
			voxel_information->mutex_rays.lock();
			auto hash_this_node = voxel_id_map->find(node_y);
			//如果没有记录，就视为新的体素
			if (hash_this_node == voxel_id_map->end()) {
				voxel_id = (*nz) + voxel_index_shift;
				(*voxel_id_map)[node_y] = voxel_id;
				(*nz)++;
			}
			else {
				voxel_id = hash_this_node->second;
			}
			voxel_information->mutex_rays.unlock();
			//对于每个视点，统计该体素的id与价值
			for (int i = 0; i < views_id.size(); i++)
			{
				(*voxel_information->mutex_voxels[voxel_id - voxel_index_shift]).lock();
				(*bipartite_list)[voxel_id].push_back(make_pair(ray_id + ray_index_shift, information_gain));
				(*voxel_information->mutex_voxels[voxel_id - voxel_index_shift]).unlock();
			}
		}
	}
}

/*
0-1 Integer linear program formulation of generalized maximum coverage problem
maximize the weighted sum of covered voxel elements in the selected view sets in which they are covered.
It subjects to (a) the cost of all voxels and the selected sets cannot exceed \beta\ast|X|\ast|V|;
(b) x_j can only be covered by at most one set;
(c) if \sum_{\forall\mathbb{v}_i\in\mathbb{V}} y_{ij}=1 then at least one set that can observe x_j is selected;
(d) if y_{ij}=1 then x_j is covered by \mathbb{v}_i;
and (e) if z_i=1 then \mathbb{v}_i is selected for the cover.
solving by Gurobi
*/

void adjacency_list_GMC_thread_process(int ray_id, int* ny, unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel_id_map, vector<vector<pair<int, double>>>* bipartite_list, View_Space* _view_space, Views_Information* _views_information, Voxel_Information* _voxel_information, Share_Data* _share_data);

class views_voxels_GMC {
public:
	int nx, ny;											//两边的点数，视点数nx，体素数ny
	vector<vector<pair<int, double>>>* bipartite_list;	//邻接表
	vector<vector<double>> bipartite_graph;				//二分图描述
	vector<double> coverage_weight;						//体素的覆盖权重
	vector<double> move_cost;							//视点移动代价
	View_Space* view_space;
	Views_Information* views_information;
	Voxel_Information* voxel_information;
	Share_Data* share_data;
	unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel_id_map;	//体素下标
	GRBEnv* env;
	GRBModel* model;
	vector<vector<GRBVar>> y;							//y_i_j表示j体素被i视点覆盖
	vector<GRBVar> z;									//z_i表示i视点被选取
	GRBLinExpr obj;
	vector<int> mask_x, mask_y;

	void solve() {
		double now_time = clock();
		// Optimize model
		model->optimize();
		share_data->access_directory(share_data->save_path + "/run_time");
		ofstream fout(share_data->save_path + "/run_time/GO" + to_string(view_space->id) + ".txt");
		fout << clock() - now_time << endl;
		// show sum of voxel weight
		cout << "Obj: " << model->get(GRB_DoubleAttr_ObjVal) << endl;
	}

	int get_voxel_num() {
		return ny;
	}

	vector<pair<int, int>> get_view_id_voxelnum_set() {
		vector<pair<int, int>> ans;
		int selected_num = 0;
		for (int i = 0; i < nx; i++) if (mask_x[i]) {
			if (z[i].get(GRB_DoubleAttr_X) == 1.0) {
				selected_num++;
				int voxelnum = 0;
				for (int j = 0; j < ny; j++)  if (mask_y[j]) {
					if (y[i][j].get(GRB_DoubleAttr_X) == 1.0) {
						voxelnum++;
					}
				}
				if (voxelnum != 0) {
					ans.push_back(make_pair(i, voxelnum));
					cout << "view " << i << " cover voxels " << voxelnum << endl;
				}
			}
		}
		cout << "Num: " << selected_num << endl;
		cout << "Num of used: " << ans.size() << endl;
		return ans;
	}

	views_voxels_GMC(int _nx, View_Space* _view_space, Views_Information* _views_information, Voxel_Information* _voxel_information, Share_Data* _share_data) {
		double now_time = clock();
		view_space = _view_space;
		views_information = _views_information;
		voxel_information = _voxel_information;
		share_data = _share_data;
		//视点按照id排序，并建立二分图(邻接表与邻接矩阵)
		sort(view_space->views.begin(), view_space->views.end(), view_id_compare);
		nx = _nx;
		bipartite_list = new vector<vector<pair<int, double>>>;
		bipartite_list->resize(nx);
		//建立体素的id表
		voxel_id_map = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>;
		//并行遍历每条射线上的体素累加至对应视点
		ny = 0;
		thread** adjacency_list_process = new thread * [views_information->ray_num];
		for (int i = 0; i < views_information->ray_num; i++) {
			adjacency_list_process[i] = new thread(adjacency_list_GMC_thread_process, i, &ny, voxel_id_map, bipartite_list, view_space, views_information, voxel_information, share_data);
		}
		for (int i = 0; i < views_information->ray_num; i++) {
			(*adjacency_list_process[i]).join();
		}
		//释放内存
		for (int i = 0; i < views_information->ray_num; i++) {
			adjacency_list_process[i]->~thread();
		}
		delete adjacency_list_process;
		if (ny != voxel_id_map->size()) cout << "node_y wrong." << endl;
		cout << "adjacency list with interested voxels num " << ny << " getted with executed time " << clock() - now_time << " ms." << endl;
		//获取视点移动权重
		move_cost.resize(nx);
		double max_dis = -1;
		for (int i = 0; i < nx; i++) {
			max_dis = max(max_dis, view_space->views[i].robot_cost);
		}
		for (int i = 0; i < nx; i++) {
			move_cost[i] = view_space->views[i].robot_cost / max_dis;
		}
		//将邻接表转为邻接矩阵
		now_time = clock();
		bipartite_graph.resize(nx);
		for (int i = 0; i < nx; i++) {
			bipartite_graph[i].resize(ny);
			for (int j = 0; j < ny; j++)
				bipartite_graph[i][j] = -1;
		}
		for (int i = 0; i < nx; i++) {
			// 过滤太远的视点
			//if (share_data->move_cost_on && move_cost[i]> share_data->move_rate) {
			if (move_cost[i] > share_data->move_rate) {
				continue;
			}
			//获取该视点的体素集合
			vector<pair<int, double>> voxels = (*bipartite_list)[i];
			for (int j = 0; j < voxels.size(); j++) {
				//cout << i <<" "<< voxels[j].first <<" "<< voxels[j].second << endl;
				//第一个值为id用于索引，第二个值为价值放入边中
				bipartite_graph[i][voxels[j].first] = voxels[j].second;
			}
		}
		cout << "bipartite_graph created with executed time " << clock() - now_time << " ms." << endl;
		//获取体素覆盖权重
		coverage_weight.resize(ny);
		for (unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>::iterator it = voxel_id_map->begin(); it != voxel_id_map->end(); it++) {
			octomap::OcTreeKey key = it->first;
			//从hash表里查询该key
			auto hash_this_key = (*views_information->occupancy_map).find(key);
			//找不到节点就下一个
			if (hash_this_key == (*views_information->occupancy_map).end()) {
				cout << "voxel has no object probility. check." << endl;
			}
			//读取节点概率值
			double occupancy = hash_this_key->second;
			//读取节点物体表面率
			double on_object = voxel_information->voxel_object(key, views_information->object_weight);
			coverage_weight[it->second] = occupancy * on_object;
		}
		cout << "All weights and cost getted." << endl;
		//建立对应的线性规划求解器
		now_time = clock();
		env = new GRBEnv();
		model = new GRBModel(*env);
		//标记无用变量
		mask_x.resize(nx);
		mask_y.resize(ny);
		for (int i = 0; i < nx; i++)
			mask_x[i] = 0;
		for (int j = 0; j < ny; j++)
			mask_y[j] = 0;
		for (int i = 0; i < nx; i++) {
			for (int j = 0; j < ny; j++)
				if (bipartite_graph[i][j] != -1) mask_x[i]++, mask_y[j]++;
		}
		// Create variables
		// (a)
		z.resize(nx);
		for (int i = 0; i < nx; i++) if (mask_x[i])
			z[i] = model->addVar(0.0, 1.0, 0.0, GRB_BINARY, "z" + to_string(i));
		// (b)
		y.resize(nx);
		for (int i = 0; i < nx; i++) if (mask_x[i]) {
			y[i].resize(ny);
			for (int j = 0; j < ny; j++) if (mask_y[j]) {
				y[i][j] = model->addVar(0.0, 1.0, 0.0, GRB_BINARY, "y" + to_string(i) + "," + to_string(j));
			}
		}
		cout << "Variables Created." << endl;
		// Set objective 
		for (int i = 0; i < nx; i++) if (mask_x[i]) {
			for (int j = 0; j < ny; j++) if (mask_y[j]) {
				obj += coverage_weight[j] * y[i][j];
			}
		}
		model->setObjective(obj, GRB_MAXIMIZE);
		cout << "Objective Set." << endl;
		// Add linear constraint
		// (c)
		GRBLinExpr subject_useful_views;
		for (int i = 0; i < nx; i++)  if (mask_x[i]) {
			GRBLinExpr subject_useful_views;
			subject_useful_views += z[i];
			for (int j = 0; j < ny; j++) if (mask_y[j]) {
				subject_useful_views -= y[i][j];
			}
			model->addConstr(subject_useful_views <= 0, "subject_useful_views");
		}
		// (d)
		for (int j = 0; j < ny; j++)  if (mask_y[j]) {
			GRBLinExpr subject_cover_by_at_least_one;
			for (int i = 0; i < nx; i++) if (mask_x[i])
				if (bipartite_graph[i][j] != -1) subject_cover_by_at_least_one += z[i];
			for (int i = 0; i < nx; i++) if (mask_x[i])
				subject_cover_by_at_least_one -= y[i][j];
			model->addConstr(subject_cover_by_at_least_one >= 0, "subject_covet_by_at_least_one" + to_string(j));
		}
		// (e)
		for (int j = 0; j < ny; j++) if (mask_y[j]) {
			GRBLinExpr subject_cover_by_one;
			for (int i = 0; i < nx; i++) if (mask_x[i]) {
				subject_cover_by_one += y[i][j];
			}
			model->addConstr(subject_cover_by_one <= 1, "subject_cover_by_one" + to_string(j));
		}
		// (f)
		GRBLinExpr subject_view_number;
		for (int i = 0; i < nx; i++) if (mask_x[i]) {
			subject_view_number += z[i];
		}
		//model->addConstr(subject_view_number <= share_data->num_of_max_iteration - 1 - share_data->iterations, "subject_view_number");
		// (g)
		GRBLinExpr subject_observation_cost;
		GRBLinExpr subject_full;
		for (int i = 0; i < nx; i++) if (mask_x[i]) {
			for (int j = 0; j < ny; j++) if (mask_y[j]) {
				subject_observation_cost += (1 - bipartite_graph[i][j]) * y[i][j];
				subject_full += y[i][j];
			}
		}
		model->addConstr(subject_observation_cost <= share_data->visble_rate * subject_full, "subject_observation_cost");
		cout << "Linear constraint added." << endl;
		model->set("TimeLimit", "100");
		cout << "TimeLimit is " << model->get(GRB_DoubleParam_TimeLimit) << " s." << endl;
		cout << "Integer linear program formulated with executed time " << clock() - now_time << " ms." << endl;
	}

	~views_voxels_GMC() {
		delete bipartite_list;
		delete voxel_id_map;
		delete env;
		delete model;
	}
};

void adjacency_list_GMC_thread_process(int ray_id, int* ny, unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel_id_map, vector<vector<pair<int, double>>>* bipartite_list, View_Space* view_space, Views_Information* views_information, Voxel_Information* voxel_information, Share_Data* share_data) {
	//该射线被哪些视点看到
	vector<int> views_id = (*views_information->rays_to_viwes_map)[ray_id];
	//仅保留感兴趣体素
	double visible = 1.0;
	octomap::KeyRay::iterator first = views_information->rays_info[ray_id]->ray->start;
	octomap::KeyRay::iterator last = views_information->rays_info[ray_id]->ray->stop;
	for (octomap::KeyRay::iterator it = views_information->rays_info[ray_id]->ray->start; it != views_information->rays_info[ray_id]->ray->stop; ++it) {
		//从hash表里查询该key
		auto hash_this_key = (*views_information->occupancy_map).find(*it);
		//找不到节点就下一个
		if (hash_this_key == (*views_information->occupancy_map).end()) continue;
		//读取节点概率值
		double occupancy = hash_this_key->second;
		//读取一下节点为物体表面率
		double on_object = voxel_information->voxel_object(*it, views_information->object_weight);
		//统计信息熵
		double information_gain = on_object * voxel_information->entropy(occupancy);
		if (information_gain >= share_data->skip_threshold) {
			octomap::OcTreeKey node_y = *it;
			int voxel_id;
			voxel_information->mutex_rays.lock();
			auto hash_this_node = voxel_id_map->find(node_y);
			//如果没有记录，就视为新的体素
			if (hash_this_node == voxel_id_map->end()) {
				(*voxel_id_map)[node_y] = *ny;
				voxel_id = *ny;
				(*ny)++;
			}
			else {
				voxel_id = hash_this_node->second;
			}
			//cerr << voxel_id << endl;
			voxel_information->mutex_rays.unlock();
			//对于每个视点，统计该体素的id与价值
			for (int i = 0; i < views_id.size(); i++)
			{
				(*voxel_information->mutex_views[views_id[i]]).lock();
				(*bipartite_list)[views_id[i]].push_back(make_pair(voxel_id, visible));
				//cerr << voxel_id<<" " << views_id[i] << endl;
				(*voxel_information->mutex_views[views_id[i]]).unlock();
			}
		}
		//更新下一视点可见性
		visible *= voxel_information->get_voxel_visible(occupancy);
	}
}
