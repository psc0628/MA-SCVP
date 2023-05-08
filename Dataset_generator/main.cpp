#include <windows.h>
#include <iostream>
#include <cstdio>
#include <thread>
#include <atomic>
#include <chrono>
typedef unsigned long long pop_t;

using namespace std;

#include "Share_Data.hpp"
#include "View_Space.hpp"
#include <gurobi_c++.h>

//Virtual_Perception_3D.hpp
void precept_thread_process(int i, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, octomap::point3d* _origin, octomap::point3d* _end, Eigen::Matrix4d* _view_pose_world, octomap::ColorOcTree* _ground_truth_model,Share_Data* share_data);

class Perception_3D {
public:
	Share_Data* share_data;
	octomap::ColorOcTree* ground_truth_model;
	int full_voxels;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

	Perception_3D(Share_Data* _share_data) {
		share_data = _share_data;
		ground_truth_model = share_data->ground_truth_model;
		full_voxels = share_data->full_voxels;
	}

	~Perception_3D() {
		
	}

	bool precept(View* now_best_view) { 
		double now_time = clock();
		//创建当前成像点云
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_parallel(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud_parallel->is_dense = false;
		cloud_parallel->points.resize(full_voxels);
		//获取视点位姿
		Eigen::Matrix4d view_pose_world;
		now_best_view->get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
		view_pose_world = (share_data->now_camera_pose_world * now_best_view->pose.inverse()).eval();
		//检查视点的key
		octomap::OcTreeKey key_origin;
		bool key_origin_have = ground_truth_model->coordToKeyChecked(now_best_view->init_pos(0), now_best_view->init_pos(1), now_best_view->init_pos(2), key_origin);
		if (key_origin_have) {
			octomap::point3d origin = ground_truth_model->keyToCoord(key_origin);
			//遍历每个体素
			octomap::point3d* end = new octomap::point3d[full_voxels];
			octomap::ColorOcTree::leaf_iterator it = ground_truth_model->begin_leafs();
			for (int i = 0; i < full_voxels; i++) {
				end[i] = it.getCoordinate();
				it++;
			}
			//ground_truth_model->write(share_data->save_path + "/test_camrea.ot");
			thread** precept_process = new thread * [full_voxels];
			for (int i = 0; i < full_voxels; i++) {
				precept_process[i] = new thread(precept_thread_process, i, cloud_parallel, &origin, &end[i], &view_pose_world, ground_truth_model, share_data);
			}
			for (int i = 0; i < full_voxels; i++)
				(*precept_process[i]).join();
			delete[] end;
			for (int i = 0; i < full_voxels; i++)
				precept_process[i]->~thread();
			delete[] precept_process;
		}
		else {
			cout << "View out of map.check." << endl;
		}
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud = temp;
		cloud->is_dense = false;
		cloud->points.resize(full_voxels);
		auto ptr = cloud->points.begin();
		int vaild_point = 0;
		auto p = cloud_parallel->points.begin();
		for (int i = 0; i < cloud_parallel->points.size(); i++, p++)
		{
			if ((*p).x == 0 && (*p).y == 0 && (*p).z == 0) continue;
			(*ptr).x = (*p).x;
			(*ptr).y = (*p).y;
			(*ptr).z = (*p).z;
			(*ptr).b = (*p).b;
			(*ptr).g = (*p).g;
			(*ptr).r = (*p).r;
			vaild_point++;
			ptr++;
		}
		cloud->width = vaild_point;
		cloud->height = 1;
		cloud->points.resize(vaild_point);
		//记录当前采集点云
		share_data->vaild_clouds++;
		share_data->clouds.push_back(cloud);
		//旋转至世界坐标系
		//*share_data->cloud_final += *cloud;
		cout << "virtual cloud get with executed time " << clock() - now_time << " ms." << endl;
		if (share_data->show) { //显示成像点云
			pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Camera"));
			viewer1->setBackgroundColor(255, 255, 255);
			viewer1->addCoordinateSystem(0.1);
			viewer1->initCameraParameters();
			viewer1->addPointCloud<pcl::PointXYZRGB>(cloud, "cloud");
			Eigen::Vector4d X(0.05, 0, 0, 1);
			Eigen::Vector4d Y(0, 0.05, 0, 1);
			Eigen::Vector4d Z(0, 0, 0.05, 1);
			Eigen::Vector4d O(0, 0, 0, 1);
			X = view_pose_world * X;
			Y = view_pose_world * Y;
			Z = view_pose_world * Z;
			O = view_pose_world * O;
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(-1));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(-1));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(-1));
			while (!viewer1->wasStopped())
			{
				viewer1->spinOnce(100);
				boost::this_thread::sleep(boost::posix_time::microseconds(100000));
			}
		}

		cloud_parallel->~PointCloud();
		return true;
	}
};

inline octomap::point3d project_pixel_to_ray_end(int x, int y, rs2_intrinsics& color_intrinsics, Eigen::Matrix4d& now_camera_pose_world, float max_range) {
	float pixel[2] = { x ,y };
	float point[3];
	rs2_deproject_pixel_to_point(point, &color_intrinsics, pixel, max_range);
	Eigen::Vector4d point_world(point[0], point[1], point[2], 1);
	point_world = now_camera_pose_world * point_world;
	return octomap::point3d(point_world(0), point_world(1), point_world(2));
}

void precept_thread_process(int i, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, octomap::point3d* _origin, octomap::point3d* _end, Eigen::Matrix4d* _view_pose_world,octomap::ColorOcTree* _ground_truth_model, Share_Data* share_data) {
	//num++;
	octomap::point3d origin = *_origin;
	Eigen::Matrix4d view_pose_world = *_view_pose_world;
	octomap::ColorOcTree* ground_truth_model = _ground_truth_model;
	pcl::PointXYZRGB point;
	point.x = 0; point.y = 0; point.z = 0;
	//投影检测是否在成像范围内
	Eigen::Vector4d end_3d(_end->x(), _end->y(), _end->z(),1);
	Eigen::Vector4d vertex = view_pose_world.inverse() * end_3d;
	float point_3d[3] = { vertex(0), vertex(1),vertex(2) };
	float pixel[2];
	rs2_project_point_to_pixel(pixel, &share_data->color_intrinsics, point_3d);
	if (pixel[0] < 0 || pixel[0]>share_data->color_intrinsics.width || pixel[1] < 0 || pixel[1]>share_data->color_intrinsics.height) {
		cloud->points[i] = point;
		return;
	}
	//反向投影找到终点
	octomap::point3d end = project_pixel_to_ray_end(pixel[0], pixel[1], share_data->color_intrinsics, view_pose_world, 1.0);
	octomap::OcTreeKey key_end;
	octomap::point3d direction = end - origin;
	octomap::point3d end_point;
	//越过未知区域，找到终点
	bool found_end_point = ground_truth_model->castRay(origin, direction, end_point, true, 1.0);
	if (!found_end_point) {//未找到终点，无观测数据
		cloud->points[i] = point;
		return;
	}
	if (end_point == origin) {
		cout << "view in the object. check!"<<endl;
		cloud->points[i] = point;
		return;
	}
	//检查一下末端是否在地图限制范围内
	bool key_end_have = ground_truth_model->coordToKeyChecked(end_point, key_end);
	if (key_end_have) {
		octomap::ColorOcTreeNode* node = ground_truth_model->search(key_end);
		if (node != NULL) {
			octomap::ColorOcTreeNode::Color color = node->getColor();
			point.x = end_point.x();
			point.y = end_point.y();
			point.z = end_point.z();
			point.b = color.b;
			point.g = color.g;
			point.r = color.r;
		}
	}
	cloud->points[i] = point;
}

//views_voxels_LM.hpp
class views_voxels_LM {
public:
	Share_Data* share_data;
	View_Space* view_space;
	vector<vector<bool>> graph;
	unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel_id_map;	//体素下标
	int num_of_voxel;
	set<int>* chosen_views;
	GRBEnv* env;
	GRBModel* model;
	vector<GRBVar> x;
	GRBLinExpr obj;

	void solve() {
		// Optimize model
		model->optimize();
		// show nonzero variables
		/*for (int i = 0; i < share_data->num_of_views; i++)
			if (x[i].get(GRB_DoubleAttr_X) == 1.0)
				cout << x[i].get(GRB_StringAttr_VarName) << " " << x[i].get(GRB_DoubleAttr_X) << endl;
		// show num of views
		cout << "Obj: " << model->get(GRB_DoubleAttr_ObjVal) << endl;*/
	}

	vector<int> get_view_id_set() {
		vector<int> ans;
		for (int i = 0; i < share_data->num_of_views; i++)
			if (x[i].get(GRB_DoubleAttr_X) == 1.0) ans.push_back(i);
		return ans;
	}

	views_voxels_LM(Share_Data* _share_data, View_Space* _view_space, set<int>* _chosen_views) {
		double now_time = clock();
		share_data = _share_data;
		view_space = _view_space;
		chosen_views = _chosen_views;
		//建立体素的id表
		num_of_voxel = 0;
		voxel_id_map = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
		for (int i = 0; i < share_data->voxels.size(); i++) {
			for (auto& it : *share_data->voxels[i]) {
				if (voxel_id_map->find(it.first) == voxel_id_map->end()) {
					(*voxel_id_map)[it.first] = num_of_voxel++;
				}
			}
		}
		//cout << num_of_voxel << " real | gt " << share_data->full_voxels << endl;
		graph.resize(num_of_voxel);
		for (int i = 0; i < share_data->num_of_views; i++) {
			graph[i].resize(num_of_voxel);
			for (int j = 0; j < num_of_voxel; j++) {
				graph[i][j] = 0;
			}
		}
		set<int> voxels_not_need;
		for (int i = 0; i < share_data->voxels.size(); i++) {
			for (auto& it : *share_data->voxels[i]) {
				graph[i][(*voxel_id_map)[it.first]] = 1;
				if (chosen_views->find(i) != chosen_views->end()) {
					voxels_not_need.insert((*voxel_id_map)[it.first]);
				}
			}
		}
		//建立对应的线性规划求解器
		now_time = clock();
		env = new GRBEnv();
		model = new GRBModel(*env);
		x.resize(share_data->num_of_views);
		// Create variables
		for (int i = 0; i < share_data->num_of_views; i++)
			x[i] = model->addVar(0.0, 1.0, 0.0, GRB_BINARY, "x" + to_string(i));
		// Set objective : \sum_{s\in S} x_s
		for (int i = 0; i < share_data->num_of_views; i++)
			obj += x[i];
		model->setObjective(obj, GRB_MINIMIZE);
		// Add linear constraint: \sum_{S:e\in S} x_s\geq1
		for (int j = 0; j < num_of_voxel; j++)
		{
			if (voxels_not_need.find(j) != voxels_not_need.end()) continue;
			GRBLinExpr subject_of_voxel;
			for (int i = 0; i < share_data->num_of_views; i++)
				if (graph[i][j] == 1) subject_of_voxel += x[i];
			model->addConstr(subject_of_voxel >= 1, "c" + to_string(j));
		}
		model->set("TimeLimit", "10");
		//cout << "Integer linear program formulated with executed time " << clock() - now_time << " ms." << endl;
	}

	~views_voxels_LM() {
		delete voxel_id_map;
		delete env;
		delete model;
	}
};

//NBV_Net_Labeler.hpp
class NBV_Net_Labeler 
{
public:
	Share_Data* share_data;
	View_Space* view_space;
	Perception_3D* percept;
	pcl::visualization::PCLVisualizer::Ptr viewer;
	int toward_state;
	int rotate_state;

	double check_size(double predicted_size, Eigen::Vector3d object_center_world, vector<Eigen::Vector3d>& points) {
		int vaild_points = 0;
		for (auto& ptr : points) {
			if (ptr(0) < object_center_world(0) - predicted_size || ptr(0) > object_center_world(0) + predicted_size) continue;
			if (ptr(1) < object_center_world(1) - predicted_size || ptr(1) > object_center_world(1) + predicted_size) continue;
			if (ptr(2) < object_center_world(2) - predicted_size || ptr(2) > object_center_world(2) + predicted_size) continue;
			vaild_points++;
		}
		return (double)vaild_points / (double)points.size();
	}

	NBV_Net_Labeler(Share_Data* _share_data, int _toward_state = 0, int _rotate_state = 0) {
		share_data = _share_data;
		toward_state = _toward_state;
		rotate_state = _rotate_state;
		share_data->save_path += share_data->name_of_pcd + "/rotate_" + to_string(rotate_state);
		share_data->save_path_nbvnet += share_data->name_of_pcd + "/rotate_" + to_string(rotate_state);
		share_data->save_path_pcnbv += share_data->name_of_pcd + "/rotate_" + to_string(rotate_state);
		cout << "toward_state is " << toward_state << " , rotate_state is " << rotate_state << endl;
		//初始化GT
		//旋转6个朝向之一
		pcl::transformPointCloud(*share_data->cloud_pcd, *share_data->cloud_pcd, share_data->get_toward_pose(toward_state));
		//旋转8个角度之一
		Eigen::Matrix3d rotation;
		rotation = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()) *
			Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
			Eigen::AngleAxisd(45 * rotate_state * acos(-1.0) / 180.0, Eigen::Vector3d::UnitZ());
		Eigen::Matrix4d T_pose(Eigen::Matrix4d::Identity(4, 4));
		T_pose(0, 0) = rotation(0, 0); T_pose(0, 1) = rotation(0, 1); T_pose(0, 2) = rotation(0, 2); T_pose(0, 3) = 0;
		T_pose(1, 0) = rotation(1, 0); T_pose(1, 1) = rotation(1, 1); T_pose(1, 2) = rotation(1, 2); T_pose(1, 3) = 0;
		T_pose(2, 0) = rotation(2, 0); T_pose(2, 1) = rotation(2, 1); T_pose(2, 2) = rotation(2, 2); T_pose(2, 3) = 0;
		T_pose(3, 0) = 0;			   T_pose(3, 1) = 0;			  T_pose(3, 2) = 0;			     T_pose(3, 3) = 1;
		pcl::transformPointCloud(*share_data->cloud_pcd, *share_data->cloud_pcd, T_pose);
		//share_data->access_directory(share_data->save_path);
		//pcl::io::savePCDFile<pcl::PointXYZ>(share_data->save_path+to_string(toward_state)+"_"+ to_string(rotate_state) +".pcd", *cloud_pcd);
		//GT cloud
		share_data->cloud_ground_truth->is_dense = false;
		share_data->cloud_ground_truth->points.resize(share_data->cloud_pcd->points.size());
		share_data->cloud_ground_truth->width = share_data->cloud_pcd->points.size();
		share_data->cloud_ground_truth->height = 1;
		auto ptr = share_data->cloud_ground_truth->points.begin();
		auto p = share_data->cloud_pcd->points.begin();
		float unit = 1.0;
		for (auto& ptr : share_data->cloud_pcd->points) {
			if (fabs(ptr.x) >= 10 || fabs(ptr.y) >= 10 || fabs(ptr.z) >= 10) {
				unit = 0.001;
				cout << "change unit from <mm> to <m>." << endl;
				break;
			}
		}
		//检查物体大小，统一缩放为0.10m左右
		vector<Eigen::Vector3d> points;
		for (auto& ptr : share_data->cloud_pcd->points) {
			Eigen::Vector3d pt(ptr.x * unit, ptr.y * unit, ptr.z * unit);
			points.push_back(pt);
		}
		Eigen::Vector3d object_center_world = Eigen::Vector3d(0, 0, 0);
		//计算点云质心
		for (auto& ptr : points) {
			object_center_world(0) += ptr(0);
			object_center_world(1) += ptr(1);
			object_center_world(2) += ptr(2);
		}
		object_center_world(0) /= points.size();
		object_center_world(1) /= points.size();
		object_center_world(2) /= points.size();
		//二分查找BBX半径，以BBX内点的个数比率达到0.90-0.95为终止条件
		/*double l = 0, r = 0, mid;
		for (auto& ptr : points) {
			r = max(r, (object_center_world - ptr).norm());
		}
		mid = (l + r) / 2;
		double precent = check_size(mid, object_center_world, points);
		double pre_precent = precent;
		while (precent > 0.999 || precent < 1.0) {
			if (precent > 0.999) {
				r = mid;
			}
			else if (precent < 1.0) {
				l = mid;
			}
			mid = (l + r) / 2;
			precent = check_size(mid, object_center_world, points);
			if (fabs(pre_precent - precent) < 0.001) break;
			pre_precent = precent;
		}*/
		//计算最远点
		double predicted_size = 0.0;
		for (auto& ptr : points) {
			predicted_size = max(predicted_size, (object_center_world - ptr).norm());
		}
		predicted_size *= 17.0/16.0;
		//predicted_size = 0.16;
		
		double scale = 1.0;
		if (share_data->mp_scale.find(share_data->name_of_pcd)!= share_data->mp_scale.end()) {
			scale = (predicted_size - share_data->mp_scale[share_data->name_of_pcd]) / predicted_size;
			cout << "object " << share_data->name_of_pcd << " large. change scale " << predicted_size << " to about " << predicted_size - share_data->mp_scale[share_data->name_of_pcd] << " m." << endl;
		}
		else {
			cout << "object " << share_data->name_of_pcd << " size is " << predicted_size << " m." << endl;
		}

		//动态分辨率
		double predicted_octomap_resolution = scale * predicted_size * 2.0 / 32.0;
		cout << "choose octomap_resolution: " << predicted_octomap_resolution << " m." << endl;
		share_data->octomap_resolution = predicted_octomap_resolution;
		share_data->octo_model = new octomap::ColorOcTree(share_data->octomap_resolution);
		share_data->GT_sample = new octomap::ColorOcTree(share_data->octomap_resolution);
		//测试BBX尺寸
		for (int i = 0; i < 32; i++)
			for (int j = 0; j < 32; j++)
				for (int k = 0; k < 32; k++)
				{
					double x = object_center_world(0) * scale * unit - scale * predicted_size + share_data->octomap_resolution * i;
					double y = object_center_world(1) * scale * unit - scale * predicted_size + share_data->octomap_resolution * j;
					double z = object_center_world(2) * scale * unit - scale * predicted_size + share_data->octomap_resolution * k;
					share_data->GT_sample->setNodeValue(x, y, z, share_data->GT_sample->getProbMissLog(), true); //初始化概率0
					//cout << x << " " << y << " " << z << endl;
				}

		//转换点云
		//double min_z = 0;
		double min_z = object_center_world(2) * scale * unit;
		for (int i = 0; i < share_data->cloud_pcd->points.size(); i++, p++)
		{
			(*ptr).x = (*p).x * scale * unit;
			(*ptr).y = (*p).y * scale * unit;
			(*ptr).z = (*p).z * scale * unit;
			(*ptr).b = 0;
			(*ptr).g = 0;
			(*ptr).r = 255;
			//GT插入点云
			octomap::OcTreeKey key;  bool key_have = share_data->ground_truth_model->coordToKeyChecked(octomap::point3d((*ptr).x, (*ptr).y, (*ptr).z), key);
			if (key_have) {
				octomap::ColorOcTreeNode* voxel = share_data->ground_truth_model->search(key);
				if (voxel == NULL) {
					share_data->ground_truth_model->setNodeValue(key, share_data->ground_truth_model->getProbHitLog(), true);
					share_data->ground_truth_model->integrateNodeColor(key, (*ptr).r, (*ptr).g, (*ptr).b);
				}
			}
			min_z = min(min_z, (double)(*ptr).z);
			//GT_sample插入点云
			octomap::OcTreeKey key_sp;  bool key_have_sp = share_data->GT_sample->coordToKeyChecked(octomap::point3d((*ptr).x, (*ptr).y, (*ptr).z), key_sp);
			if (key_have_sp) {
				octomap::ColorOcTreeNode* voxel_sp = share_data->GT_sample->search(key_sp);
				//if (voxel_sp == NULL) {
					share_data->GT_sample->setNodeValue(key_sp, share_data->GT_sample->getProbHitLog(), true);
					share_data->GT_sample->integrateNodeColor(key_sp, 255, 0, 0);
				//}
			}
			ptr++;
		}
		//记录桌面
		share_data->min_z_table = min_z - share_data->ground_truth_resolution;
		
		//share_data->access_directory(share_data->save_path);
		share_data->ground_truth_model->updateInnerOccupancy();
		//share_data->ground_truth_model->write(share_data->save_path + "/GT.ot");
		//GT_sample_voxels
		share_data->GT_sample->updateInnerOccupancy();
		//share_data->GT_sample->write(share_data->save_path + "/GT_sample.ot");
		share_data->init_voxels = 0;
		for (octomap::ColorOcTree::leaf_iterator it = share_data->GT_sample->begin_leafs(), end = share_data->GT_sample->end_leafs(); it != end; ++it) {
			share_data->init_voxels++;
		}
		//cout << "Map_GT_sample has voxels " << share_data->init_voxels << endl;
		//if (share_data->init_voxels != 32768) cout << "WARNING! BBX small." << endl;
		//ofstream fout(share_data->save_path + "/GT_size.txt");
		//fout << scale * predicted_size << endl;

		share_data->full_voxels = 0;
		for (octomap::ColorOcTree::leaf_iterator it = share_data->ground_truth_model->begin_leafs(), end = share_data->ground_truth_model->end_leafs(); it != end; ++it) {
			share_data->full_voxels++;
		}

		//初始化viewspace
		view_space = new View_Space(share_data);

		//相机类初始化
		percept = new Perception_3D(share_data);
		
		//srand(time(0));
	}

	int get_nbv_view_cases_and_distrubution() {
		double now_time = clock();
		//每个视点成像并统计体素
		int full_num = 0;
		unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* all_voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
		for (int i = 0; i < view_space->views.size(); i++) {
			percept->precept(&view_space->views[i]);
			//get voxel map
			int num = 0;
			unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
			for (int j = 0; j < share_data->clouds[i]->points.size(); j++) {
				octomap::OcTreeKey key = share_data->ground_truth_model->coordToKey(share_data->clouds[i]->points[j].x, share_data->clouds[i]->points[j].y, share_data->clouds[i]->points[j].z);
				if (voxel->find(key) == voxel->end()) {
					(*voxel)[key] = num++;
				}
				if (all_voxel->find(key) == all_voxel->end()) {
					(*all_voxel)[key] = full_num++;
				}
			}
			share_data->voxels.push_back(voxel);
		}
		delete all_voxel;
		cout << "All voxels(cloud) num is " << full_num << endl;
		cout << "all virtual cloud get with executed time " << clock() - now_time << " ms." << endl;

		now_time = clock();
		vector<vector<double>> distrubution; //i个视点个数时所能增加的比例
		distrubution.resize(share_data->num_of_views);
		//枚举每个视角启动
		vector<long long> nbv_view_cases;
		for (int k = 0; k < share_data->num_of_views; k++) {
			long long now_view_case = 1LL << k;
			while (true) {
				//计算当前view_case可见数量
				set<int> chosen_views;
				for (long long j = now_view_case, i = 0; j != 0; j >>= 1, i++)
					if (j & 1) { //j位，i视点
						chosen_views.insert(i);
					}
				int now_num_visible = 0;
				unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* observed_voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
				for (long long j = now_view_case, i = 0; j != 0; j >>= 1, i++)
					if (j & 1) { //j位，k视点
						for (auto& key : *share_data->voxels[i]) {
							if (observed_voxel->find(key.first) == observed_voxel->end()) {
								(*observed_voxel)[key.first] = now_num_visible++;
							}
						}
					}
				//cout <<"All visible voxels(cloud) num is "<< now_num_visible << endl;
				//统计分布
				distrubution[chosen_views.size() - 1].push_back(1.0 * now_num_visible / full_num);
				//如果完全覆盖，则处理分布后结束
				if (now_num_visible == full_num) {
					for (int k = chosen_views.size() - 1; k >= 1; k--) { //分布表每项最后一列是当前的情况，倒着减前一个
						distrubution[k][distrubution[k].size() - 1] -= distrubution[k - 1][distrubution[k - 1].size() - 1];
					}
					//for (int k = 0; k < chosen_views.size(); k++) cout << distrubution[k][distrubution[k].size() - 1] << endl;
					//cout << endl;
					break;
				}
				// 没覆盖则是一个view_case
				nbv_view_cases.push_back(now_view_case); 
				//通过打分计算下一个view_case
				vector<int> score;
				score.resize(share_data->num_of_views);
				for (int i = 0; i < share_data->num_of_views; i++)
					score[i] = 0;
				for (int i = 0; i < view_space->views.size(); i++)
					if (chosen_views.count(i) == 0) { //j位，i视点
						for (auto& key : *share_data->voxels[i]) {
							if (observed_voxel->find(key.first) == observed_voxel->end()) {
								score[i]++;
							}
						}
						//cout << i <<" : " << score[i] << endl;
					}
				int best_view_id = -1;
				int best_score = -1;
				for (int i = 0; i < view_space->views.size(); i++)
					if (chosen_views.count(i) == 0) { //j位，i视点
						if (score[i] > best_score) {
							best_score = score[i];
							best_view_id = i;
						}
					}
				delete observed_voxel;
				//更新view_case
				now_view_case |= (1LL << best_view_id);
			}
		}
		cout << "gt getted with executed time " << clock() - now_time << " ms." << endl;
		//保存分布与view_case
		share_data->access_directory(share_data->gt_path + share_data->name_of_pcd + "/rotate_" + to_string(rotate_state));
		for (int i = 0; i < share_data->num_of_views; i++) {
			if (distrubution[i].size() == 0) break;
			ofstream fout_distrubution(share_data->gt_path + share_data->name_of_pcd + "/rotate_" + to_string(rotate_state) + "/nbv_" + to_string(i) + ".txt");
			fout_distrubution << setprecision(10);
			for (int j = 0; j < distrubution[i].size(); j++) {
				fout_distrubution << distrubution[i][j] << "\n";
			}
		}
		ofstream fout_view_cases(share_data->gt_path + share_data->name_of_pcd + "/rotate_" + to_string(rotate_state) + "/view_cases.txt");
		for (int i = 0; i < nbv_view_cases.size(); i++) {
			fout_view_cases << nbv_view_cases[i] << "\n";
		}

		return 0;
	}

	int label_all_view_csaes() {
		double now_time = clock();
		//每个视点成像并统计体素
		int full_num = 0;
		unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* all_voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
		for (int i = 0; i < view_space->views.size(); i++) {
			percept->precept(&view_space->views[i]);
			//get voxel map
			int num = 0;
			unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
			for (int j = 0; j < share_data->clouds[i]->points.size(); j++) {
				octomap::OcTreeKey key = share_data->ground_truth_model->coordToKey(share_data->clouds[i]->points[j].x, share_data->clouds[i]->points[j].y, share_data->clouds[i]->points[j].z);
				if (voxel->find(key) == voxel->end()) {
					(*voxel)[key] = num++;
				}
				if (all_voxel->find(key) == all_voxel->end()) {
					(*all_voxel)[key] = full_num++;
				}
			}
			share_data->voxels.push_back(voxel);
		}
		delete all_voxel;
		cout << "All voxels(cloud) num is " << full_num << endl;
		cout << "all virtual cloud get with executed time " << clock() - now_time << " ms." << endl;

		now_time = clock();
		for (long long cas = 0; cas < share_data->view_cases.size(); cas++) { //方案cas
			//if (share_data->view_cases[cas] != 134217872) continue; //for testing
			//cout << cas << " case testing:" << endl;
			set<int> chosen_views;
			for (long long j = share_data->view_cases[cas], i = 0; j != 0; j >>= 1, i++)
				if (j & 1) { //j位，i视点
					chosen_views.insert(i);
				}
			int num = 0;
			unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* observed_voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
			for (long long j = share_data->view_cases[cas], i = 0; j != 0; j >>= 1, i++)
				if(j&1){ //j位，i视点
					for (auto &key: *share_data->voxels[i]) {
						if (observed_voxel->find(key.first) == observed_voxel->end()) {
							(*observed_voxel)[key.first] = num++;
						}
					}
				}
			//cout <<"All visible voxels(cloud) num is "<< num << endl;

			//NBV打分
			vector<int> score;
			score.resize(share_data->num_of_views);
			for (int i = 0; i < share_data->num_of_views; i++)
				score[i] = 0;
			//octomap::ColorOcTree* u_rest = new octomap::ColorOcTree(share_data->ground_truth_resolution);
			for(int i=0;i<view_space->views.size();i++)
				if(chosen_views.count(i)==0){ //j位，i视点
					for (auto &key : *share_data->voxels[i]) {
						if (observed_voxel->find(key.first) == observed_voxel->end()) {
							score[i]++;
							//u_rest->setNodeValue(key.first, octomap::logodds(1.0), true);
							//u_rest->integrateNodeColor(key.first, 255, 0, 0);
						}
					}
					//cout << i <<" : " << score[i] << endl;
				}
			//u_rest->updateInnerOccupancy();
			//u_rest->write(share_data->save_path_nbvnet + "/u_rest.ot");
			int best_view_id = -1;
			int best_score = -1;
			for (int i = 0; i < view_space->views.size(); i++)
				if (chosen_views.count(i) == 0) { //j位，i视点
					if (score[i] > best_score) {
						best_score = score[i];
						best_view_id = i;
					}
				}
			delete observed_voxel;

			//集合覆盖求解
			views_voxels_LM* SCOP_solver = new views_voxels_LM(share_data, view_space, &chosen_views);
			SCOP_solver->solve();
			vector<int> need_views = SCOP_solver->get_view_id_set();
			delete SCOP_solver;	

			//插入物体和桌面
			octomap::ColorOcTree* octo_model_test = new octomap::ColorOcTree(share_data->octomap_resolution);
			for (octomap::ColorOcTree::leaf_iterator it = share_data->ground_truth_model->begin_leafs(), end = share_data->ground_truth_model->end_leafs(); it != end; ++it) {
				octo_model_test->setNodeValue(it.getX(), it.getY(), it.getZ(), octo_model_test->getProbHitLog(), true);
			}
			octo_model_test->updateInnerOccupancy();
			for (double x = share_data->object_center_world(0) - 0.2; x <= share_data->object_center_world(0) + 0.2; x += share_data->octomap_resolution)
				for (double y = share_data->object_center_world(1) - 0.2; y <= share_data->object_center_world(1) + 0.2; y += share_data->octomap_resolution) {
					double z = share_data->min_z_table;
					octo_model_test->setNodeValue(x, y, z, octo_model_test->getProbHitLog(), true);
				}
			octo_model_test->updateInnerOccupancy();
			//octo_model_test->write(share_data->save_path + "/test.ot");
			int num_of_test = 0;
			for (octomap::ColorOcTree::leaf_iterator it = octo_model_test->begin_leafs(), end = octo_model_test->end_leafs(); it != end; ++it) {
				num_of_test++;
			}
			//对加入桌面后的视角成像
			Perception_3D test(share_data);
			test.ground_truth_model = octo_model_test;
			test.full_voxels = num_of_test;
			
			//初始化BBX为0.5
			octomap::ColorOcTree* octo_model = new octomap::ColorOcTree(share_data->octomap_resolution);
			for (int i = 0; i < 32; i++)
				for (int j = 0; j < 32; j++)
					for (int k = 0; k < 32; k++)
					{
						double x = share_data->object_center_world(0) - share_data->predicted_size + share_data->octomap_resolution * i;
						double y = share_data->object_center_world(1) - share_data->predicted_size + share_data->octomap_resolution * j;
						double z = max(share_data->min_z_table, share_data->object_center_world(2) - share_data->predicted_size) + share_data->octomap_resolution * k;
						//double z = share_data->object_center_world(2) - share_data->predicted_size + share_data->octomap_resolution * k;
						octo_model->setNodeValue(x, y, z, (float)0, true); //初始化概率0.5，即logodds为0
					}
			octo_model->updateInnerOccupancy();

			//插入成像点云
			for (long long j = share_data->view_cases[cas], i = 0; j != 0; j >>= 1, i++)
				if (j&1) { //j位，i视点
					test.precept(&view_space->views[i]);
					octomap::Pointcloud cloud_octo;
					for (auto &p : share_data->clouds[view_space->views.size()]->points) {
						cloud_octo.push_back(p.x, p.y, p.z);
					}
					octo_model->insertPointCloud(cloud_octo, octomap::point3d(view_space->views[i].init_pos(0), view_space->views[i].init_pos(1), view_space->views[i].init_pos(2)), -1, true, false);
					for (auto &p : share_data->clouds[view_space->views.size()]->points) {
						octo_model->integrateNodeColor(p.x, p.y, p.z, 255, 0, 0);
						if (p.z < share_data->min_z_table + share_data->octomap_resolution) octo_model->setNodeColor(p.x, p.y, p.z, 0, 0, 255);
					}
					octo_model->updateInnerOccupancy();
					share_data->clouds[view_space->views.size()]->~PointCloud();
					share_data->clouds.pop_back();
					
				}
			delete octo_model_test;
			octo_model->updateInnerOccupancy();
			octo_model->write(share_data->save_path_nbvnet + "/grid.ot");
			
			share_data->access_directory(share_data->save_path);
			share_data->access_directory(share_data->save_path_nbvnet);
			ofstream fout_grid(share_data->save_path + "/grid_" + to_string(cas) + ".txt"); //MA-SCVP
			ofstream fout_grid_nbvnet(share_data->save_path_nbvnet + "/grid_" + to_string(cas) + ".txt");
			ofstream fout_view_id(share_data->save_path_nbvnet + "/id_" + to_string(cas) + ".txt");
			ofstream fout_view_ids(share_data->save_path + "/ids_" + to_string(cas) + ".txt"); //MA-SCVP
			//octomap::ColorOcTree* octo_model_square = new octomap::ColorOcTree(share_data->octomap_resolution);
			for (int i = 0; i < 32; i++)
				for (int j = 0; j < 32; j++)
					for (int k = 0; k < 32; k++)
					{
						double x = share_data->object_center_world(0) - share_data->predicted_size + share_data->octomap_resolution * i;
						double y = share_data->object_center_world(1) - share_data->predicted_size + share_data->octomap_resolution * j;
						double z = max(share_data->min_z_table, share_data->object_center_world(2) - share_data->predicted_size) + share_data->octomap_resolution * k;
						//double z = share_data->object_center_world(2) - share_data->predicted_size + share_data->octomap_resolution * k;
						auto node = octo_model->search(x, y, z);
						if (node == NULL) cout << "what?" << endl;
						//fout_grid << x - share_data->object_center_world(0) << ' ' << y - share_data->object_center_world(1) << ' ' << z - share_data->object_center_world(2) << ' ' << node->getOccupancy() << '\n';
						fout_grid << node->getOccupancy() << '\n';
						fout_grid_nbvnet << node->getOccupancy() << '\n';
						//octo_model_square->setNodeValue(x, y, z, node->getLogOdds(), true);
						//if (node->getOccupancy() > 0.65) {
						//	if(z >= share_data->min_z_table + share_data->octomap_resolution) octo_model_square->integrateNodeColor(x, y, z, 255, 0, 0);
						//	else octo_model_square->integrateNodeColor(x, y, z, 0, 0, 255);
						//}
					}
			fout_view_id << best_view_id << '\n';
			for (int i = 0; i < need_views.size(); i++) { //MA-SCVP
				fout_view_ids << need_views[i] << '\n';
			}
			//octo_model_square->updateInnerOccupancy();
			//octo_model_square->write(share_data->save_path_nbvnet + "/square_grid.ot");
			//delete octo_model_square;
			delete octo_model;
			

			share_data->access_directory(share_data->save_path_pcnbv);
			ofstream fout_cloud(share_data->save_path_pcnbv + "/cloud_" + to_string(cas) + ".txt");
			ofstream fout_view_state(share_data->save_path + "/state_" + to_string(cas) + ".txt"); //MA-SCVP
			ofstream fout_view_state_pcnbv(share_data->save_path_pcnbv + "/state_" + to_string(cas) + ".txt");
			ofstream fout_view_score(share_data->save_path_pcnbv + "/score_" + to_string(cas) + ".txt");

			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGB>);
			vector<int> state;
			state.resize(share_data->num_of_views);
			for (int i = 0; i < share_data->num_of_views; i++)
				state[i] = 0;
			for (int i = 0; i < view_space->views.size(); i++) //读方案中选取的视点
				if (chosen_views.count(i) == 1) { //j位，i视点
					state[i] = 1;
					*cloud_out += *(share_data->clouds[i]);
				}
			//随机降采样点云
			pcl::RandomSample<pcl::PointXYZRGB> ran;
			ran.setInputCloud(cloud_out);
			ran.setSample(512); //设置下采样点云的点数
			ran.filter(*cloud_out);

			for (int k = 0; k < cloud_out->points.size(); k++) {
				fout_cloud << cloud_out->points[k].x << ' ' << cloud_out->points[k].y << ' ' << cloud_out->points[k].z << '\n';
			}
			for (int i = 0; i < view_space->views.size(); i++) {
				fout_view_state << state[i] << '\n';
				fout_view_state_pcnbv << state[i] << '\n';
				fout_view_score << 1.0 * score[i] / full_num << '\n';
			}
			cloud_out->~PointCloud();

			cout << "labed " << cas << " getted with executed time " << clock() - now_time << " ms." << endl;
			/*if (cas == 1) {
				//test.precept(&view_space->views[best_view_id]);
				pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Camera"));
				viewer1->setBackgroundColor(0, 0, 0);
				viewer1->addCoordinateSystem(0.1);
				viewer1->initCameraParameters();
				viewer1->addPointCloud<pcl::PointXYZRGB>(share_data->clouds[best_view_id], "cloud");
				while (!viewer1->wasStopped())
				{
					viewer1->spinOnce(100);
					boost::this_thread::sleep(boost::posix_time::microseconds(100000));
				}
			}*/
		}
		return 0;
	}

	~NBV_Net_Labeler() {
		delete view_space;
		delete percept;
	}
};

atomic<bool> stop = false;		//控制程序结束
Share_Data* share_data;			//共享数据区指针
NBV_Net_Labeler* labeler;

#define DebugOne 0
#define TestNBV 1
#define RunMaxLongTail 2
#define RunSub 3
#define CheckAll 4

int main()
{
	//Init
	ios::sync_with_stdio(false);
	int mode;
	cout << "input mode:";
	cin >> mode;
	//测试集
	vector<string> names;
	cout << "input models:" << endl;
	string name;
	while (cin >> name) {
		if (name == "-1") break;
		names.push_back(name);
	}
	//选取模式
	if (mode == DebugOne){
		for (int i = 0; i < 1; i++){
			for (int j = 0; j < 8; j++) {
				share_data = new Share_Data("../DefaultConfiguration.yaml", "", -1, -1);
				labeler = new NBV_Net_Labeler(share_data, i, j);
				if (share_data->init_voxels < 30) continue;
				labeler->label_all_view_csaes();
				delete labeler;
				delete share_data;
			}
		}
	}
	else if (mode == TestNBV){
		for (int i = 0; i < names.size(); i++) {
			for (int j = 0; j < 1; j++){
				for (int k = 0; k < 8; k++) {
					share_data = new Share_Data("../DefaultConfiguration.yaml", names[i], -1, -1);
					labeler = new NBV_Net_Labeler(share_data, j, k);
					if (share_data->init_voxels < 30) continue;
					labeler->get_nbv_view_cases_and_distrubution();
					delete labeler;
					delete share_data;
				}
			}
		}
	}
	else if (mode == RunMaxLongTail) {
		srand(time(0));
		cout << "RAND_MAX is " << RAND_MAX << endl;
		//cout << "input rot:"; int rot; cin >> rot;
		//生成样本
		int need_case_1 = 32;
		for (int i = 0; i < names.size(); i++) {
			for (int j = 0; j < 1; j++) {
				for (int k = 0; k < 8; k++) {
				//for (int k = rot; k < rot+1; k++) {
					//LongTailSampleMethod
					share_data = new Share_Data("../DefaultConfiguration.yaml", names[i], LongTailSampleMethod, need_case_1);
					//读取生成好的NBV重建分布
					vector<double> distrubution;
					distrubution.resize(32);
					for (int m = 0; m < 32; m++) distrubution[m] = 0.0;
					for (int m = 0; m < 32; m++) {
						ifstream fin(share_data->gt_path + names[i] + "/rotate_" + to_string(k) + "/nbv_" + to_string(m) + ".txt");
						if (fin.is_open()) {
							double surface_added;
							while (fin >> surface_added) {
								distrubution[m] += surface_added;
							}
						}
					}
					ofstream fout_distrubution(share_data->gt_path + names[i] + "/rotate_" + to_string(k) + "/" + to_string(need_case_1) + "_distrubution.txt");
					fout_distrubution << setprecision(10);
					for (int m = 0; m < 32; m++){
						distrubution[m] /= 32;
						fout_distrubution << distrubution[m] << '\n';
					}
					fout_distrubution.close();
					//读所有可行NBVcase
					ifstream fin_view_cases(share_data->gt_path + names[i] + "/rotate_" + to_string(k) + "/view_cases.txt");
					vector<vector<long long>> nbv_view_cases_by_iter;
					vector<vector<long long>> nbv_view_cases_by_view;
					nbv_view_cases_by_view.resize(32);
					long long now_case;
					vector<long long> now_view_cases;
					while (fin_view_cases >> now_case) {
						if ( __popcnt(now_case) == 1) {
							if (now_view_cases.size() != 0) nbv_view_cases_by_iter.push_back(now_view_cases);
							now_view_cases.clear();
						}
						now_view_cases.push_back(now_case);
						nbv_view_cases_by_view[__popcnt(now_case)].push_back(now_case);
					}
					if (now_view_cases.size() != 0) nbv_view_cases_by_iter.push_back(now_view_cases);
					//for (int m = 0; m < 32; m++) cout << nbv_view_cases_by_iter[m].size() << endl;
					//for (int m = 0; m < 32; m++) cout << nbv_view_cases_by_view[m].size() << endl;
					//生成长尾采样
					int sum_case_num = 0;
					vector<vector<long long>> out_ans_longtail;
					for (long long m = 1; m <= 31; m++) {
						vector<long long> ans;
						int need_case = ceil(1.0 * need_case_1 / distrubution[1] * distrubution[m]);
						sum_case_num += need_case;
						while (ans.size() != need_case) {
							long long n = ((long long)rand() << 30) + ((long long)rand() << 15) + (long long)rand();
							n %= (long long)nbv_view_cases_by_view[m].size();
							ans.push_back(nbv_view_cases_by_view[m][n]);
							nbv_view_cases_by_view[m].erase(nbv_view_cases_by_view[m].begin() + n);
						}
						out_ans_longtail.push_back(ans);
					}
					share_data->view_cases.clear();
					ofstream fout_longtail(share_data->gt_path + names[i] + "/rotate_" + to_string(k) + "/" + to_string(need_case_1) + "_longtail_sample.txt");
					for (int m = 0; m < out_ans_longtail.size(); m++) {
						//cout << m + 1 << " size is " << out_ans_longtail[m].size() << endl;
						for (int n = 0; n < out_ans_longtail[m].size(); n++) {
							fout_longtail << out_ans_longtail[m][n] << "\n";
							share_data->view_cases.push_back(out_ans_longtail[m][n]);
						}
					}
					fout_longtail.close();
					labeler = new NBV_Net_Labeler(share_data, j, k);
					labeler->label_all_view_csaes();
					delete labeler;
					delete share_data;

					//NBVSampleMethod
					share_data = new Share_Data("../DefaultConfiguration.yaml", names[i], NBVSampleMethod, need_case_1);
					//随机加一组case直至大于等于sum
					vector<long long> out_ans_nbv;
					while (out_ans_nbv.size() < sum_case_num) {
						long long m = ((long long)rand() << 30) + ((long long)rand() << 15) + (long long)rand();
						m %= (long long)nbv_view_cases_by_iter.size();
						//最后一组计算一下超过的数量，保证平衡
						int need_num = sum_case_num - out_ans_nbv.size();
						int real_num = out_ans_nbv.size() + nbv_view_cases_by_iter[m].size() - sum_case_num;
						//cout << need_num << " " << real_num << endl;
						if (need_num < real_num) break;
						for(int n=0;n< nbv_view_cases_by_iter[m].size();n++)
							out_ans_nbv.push_back(nbv_view_cases_by_iter[m][n]);
						nbv_view_cases_by_iter.erase(nbv_view_cases_by_iter.begin() + m);
					}
					share_data->view_cases.clear();
					ofstream fout_nbv(share_data->gt_path + names[i] + "/rotate_" + to_string(k) + "/" + to_string(need_case_1) + "_nbv_sample.txt");
					for (int m = 0; m < out_ans_nbv.size(); m++) {
						fout_nbv << out_ans_nbv[m] << "\n";
						share_data->view_cases.push_back(out_ans_nbv[m]);
					}
					fout_nbv.close();
					labeler = new NBV_Net_Labeler(share_data, j, k);
					labeler->label_all_view_csaes();
					delete labeler;
					delete share_data;
				}
			}
		}
	}
	else if (mode == RunSub) {
		srand(time(0));
		cout << "RAND_MAX is " << RAND_MAX << endl;
		//cout << "input rot:"; int rot; cin >> rot;
		//从已采样的数据中取子集，sampling_space_case_1母集合F1，need_case_1子集F1
		int sampling_space_case_1 = 32; // 32, 16
		int need_case_1 = 8; //16, 8
		for (int i = 0; i < names.size(); i++) {
			for (int j = 0; j < 1; j++) {
				for (int k = 0; k < 8; k++) {
					//LongTailSampleMethod
					share_data = new Share_Data("../DefaultConfiguration.yaml", names[i], LongTailSampleMethod, need_case_1);
					//读取生成好的NBV重建分布
					vector<double> distrubution;
					distrubution.resize(32);
					for (int m = 0; m < 32; m++) distrubution[m] = 0.0;
					for (int m = 0; m < 32; m++) {
						ifstream fin(share_data->gt_path + names[i] + "/rotate_" + to_string(k) + "/nbv_" + to_string(m) + ".txt");
						if (fin.is_open()) {
							double surface_added;
							while (fin >> surface_added) {
								distrubution[m] += surface_added;
							}
						}
					}
					ofstream fout_distrubution(share_data->gt_path + names[i] + "/rotate_" + to_string(k) + "/" + to_string(need_case_1) + "_distrubution.txt");
					fout_distrubution << setprecision(10);
					for (int m = 0; m < 32; m++) {
						distrubution[m] /= 32;
						fout_distrubution << distrubution[m] << '\n';
					}
					fout_distrubution.close();
					//读所有可行长尾采样case
					ifstream fin_view_cases(share_data->gt_path + names[i] + "/rotate_" + to_string(k) + "/" + to_string(sampling_space_case_1) + "_longtail_sample.txt");
					vector<vector<long long>> nbv_view_cases_by_view;
					nbv_view_cases_by_view.resize(32);
					unordered_map<long long, int> map_view_case_with_index;
					long long now_case;
					int index_of_view_case = 0;
					while (fin_view_cases >> now_case) {
						nbv_view_cases_by_view[__popcnt(now_case)].push_back(now_case);
						map_view_case_with_index[now_case] = index_of_view_case++;
					}
					fin_view_cases.close();
					//生成长尾采样
					int sum_case_num = 0;
					vector<vector<long long>> out_ans_longtail;
					for (long long m = 1; m <= 31; m++) {
						vector<long long> ans;
						int need_case = ceil(1.0 * need_case_1 / distrubution[1] * distrubution[m]);
						sum_case_num += need_case;
						while (ans.size() != need_case) {
							long long n = ((long long)rand() << 30) + ((long long)rand() << 15) + (long long)rand();
							n %= (long long)nbv_view_cases_by_view[m].size();
							ans.push_back(nbv_view_cases_by_view[m][n]);
							nbv_view_cases_by_view[m].erase(nbv_view_cases_by_view[m].begin() + n);
						}
						out_ans_longtail.push_back(ans);
					}
					share_data->view_cases.clear();
					//保存长尾采样
					ofstream fout_longtail(share_data->gt_path + names[i] + "/rotate_" + to_string(k) + "/" + to_string(need_case_1) + "_longtail_sample.txt");
					for (int m = 0; m < out_ans_longtail.size(); m++) {
						//cout << m + 1 << " size is " << out_ans_longtail[m].size() << endl;
						for (int n = 0; n < out_ans_longtail[m].size(); n++) {
							fout_longtail << out_ans_longtail[m][n] << "\n";
							share_data->view_cases.push_back(out_ans_longtail[m][n]);
						}
					}
					fout_longtail.close();
					labeler = new NBV_Net_Labeler(share_data, j, k);
					//复制对应文件，Longtail
					for (int m = 0; m < share_data->view_cases.size(); m++) {
						int cas = map_view_case_with_index[share_data->view_cases[m]];
						string path = share_data->pre_path + to_string(sampling_space_case_1) + "/MASCVP_LongTailSample/" + names[i] + "/rotate_" + to_string(k) + "/";
						string path_nbvnet = share_data->pre_path + to_string(sampling_space_case_1) + "/NBVNET_LongTailSample/" + names[i] + "/rotate_" + to_string(k) + "/";
						string path_pcnbv = share_data->pre_path + to_string(sampling_space_case_1) + "/PCNBV_LongTailSample/" + names[i] + "/rotate_" + to_string(k) + "/";
						share_data->access_directory(share_data->save_path);
						share_data->access_directory(share_data->save_path_nbvnet);
						share_data->access_directory(share_data->save_path_pcnbv);
						ofstream fout_grid(share_data->save_path + "/grid_" + to_string(m) + ".txt"); //MA-SCVP
						ofstream fout_grid_nbvnet(share_data->save_path_nbvnet + "/grid_" + to_string(m) + ".txt");
						ofstream fout_view_id(share_data->save_path_nbvnet + "/id_" + to_string(m) + ".txt");
						ofstream fout_view_ids(share_data->save_path + "/ids_" + to_string(m) + ".txt"); //MA-SCVP
						ofstream fout_cloud(share_data->save_path_pcnbv + "/cloud_" + to_string(m) + ".txt");
						ofstream fout_view_state(share_data->save_path + "/state_" + to_string(m) + ".txt"); //MA-SCVP
						ofstream fout_view_state_pcnbv(share_data->save_path_pcnbv + "/state_" + to_string(m) + ".txt");
						ofstream fout_view_score(share_data->save_path_pcnbv + "/score_" + to_string(m) + ".txt");
						double temp_double;
						int temp_int;
						//Copy MASCVP
						ifstream fin_grid(path + "grid_" + to_string(cas) + ".txt");
						while (fin_grid >> temp_double) fout_grid << temp_double << '\n';
						ifstream fin_state(path + "state_" + to_string(cas) + ".txt");
						while (fin_state >> temp_int) fout_view_state << temp_int << '\n';
						ifstream fin_id(path + "ids_" + to_string(cas) + ".txt");
						while (fin_id >> temp_int) fout_view_ids << temp_int << '\n';
						//Copy NBVNET
						ifstream fin_grid_nbvnet(path_nbvnet + "grid_" + to_string(cas) + ".txt");
						while (fin_grid_nbvnet >> temp_double) fout_grid_nbvnet << temp_double << '\n';
						ifstream fin_id_nbvnet(path_nbvnet + "id_" + to_string(cas) + ".txt");
						while (fin_id_nbvnet >> temp_int) fout_view_id << temp_int << '\n';
						//Copy PCNBV
						ifstream fin_cloud(path_pcnbv + "cloud_" + to_string(cas) + ".txt");
						int next_line = 0;
						while (fin_cloud >> temp_double) {
							fout_cloud << temp_double << (next_line % 3 == 2 ? '\n' : ' ');
							next_line++;
						}
						ifstream fin_state_pcnbv(path_pcnbv + "state_" + to_string(cas) + ".txt");
						while (fin_state_pcnbv >> temp_int) fout_view_state_pcnbv << temp_int << '\n';
						ifstream fin_score(path_pcnbv + "score_" + to_string(cas) + ".txt");
						while (fin_score >> temp_double) fout_view_score << temp_double << '\n';
					}
					delete labeler;
					delete share_data;

					//NBVSampleMethod
					share_data = new Share_Data("../DefaultConfiguration.yaml", names[i], NBVSampleMethod, need_case_1);
					//读所有可行NBVcase
					fin_view_cases.open(share_data->gt_path + names[i] + "/rotate_" + to_string(k) + "/" + to_string(sampling_space_case_1) + "_nbv_sample.txt");
					vector<vector<long long>> nbv_view_cases_by_iter;
					map_view_case_with_index.clear();
					index_of_view_case = 0;
					vector<long long> now_view_cases;
					while (fin_view_cases >> now_case) {
						if (__popcnt(now_case) == 1) {
							if (now_view_cases.size() != 0) nbv_view_cases_by_iter.push_back(now_view_cases);
							now_view_cases.clear();
						}
						now_view_cases.push_back(now_case);
						map_view_case_with_index[now_case] = index_of_view_case++;
					}
					if (now_view_cases.size() != 0) nbv_view_cases_by_iter.push_back(now_view_cases);
					fin_view_cases.close();
					//随机加一组case直至大于等于sum
					vector<long long> out_ans_nbv;
					while (out_ans_nbv.size() < sum_case_num) {
						long long m = ((long long)rand() << 30) + ((long long)rand() << 15) + (long long)rand();
						m %= (long long)nbv_view_cases_by_iter.size();
						//最后一组计算一下超过的数量，保证平衡
						int need_num = sum_case_num - out_ans_nbv.size();
						int real_num = out_ans_nbv.size() + nbv_view_cases_by_iter[m].size() - sum_case_num;
						//cout << need_num << " " << real_num << endl;
						if (need_num < real_num) break;
						for (int n = 0; n < nbv_view_cases_by_iter[m].size(); n++)
							out_ans_nbv.push_back(nbv_view_cases_by_iter[m][n]);
						nbv_view_cases_by_iter.erase(nbv_view_cases_by_iter.begin() + m);
					}
					share_data->view_cases.clear();
					//保存NBV采样
					ofstream fout_nbv(share_data->gt_path + names[i] + "/rotate_" + to_string(k) + "/" + to_string(need_case_1) + "_nbv_sample.txt");
					for (int m = 0; m < out_ans_nbv.size(); m++) {
						fout_nbv << out_ans_nbv[m] << "\n";
						share_data->view_cases.push_back(out_ans_nbv[m]);
					}
					fout_nbv.close();
					labeler = new NBV_Net_Labeler(share_data, j, k);
					//复制对应文件，nbv
					for (int m = 0; m < share_data->view_cases.size(); m++) {
						int cas = map_view_case_with_index[share_data->view_cases[m]];
						string path = share_data->pre_path + to_string(sampling_space_case_1) + "/MASCVP_NBVSample/" + names[i] + "/rotate_" + to_string(k) + "/";
						string path_nbvnet = share_data->pre_path + to_string(sampling_space_case_1) + "/NBVNET_NBVSample/" + names[i] + "/rotate_" + to_string(k) + "/";
						string path_pcnbv = share_data->pre_path + to_string(sampling_space_case_1) + "/PCNBV_NBVSample/" + names[i] + "/rotate_" + to_string(k) + "/";
						share_data->access_directory(share_data->save_path);
						share_data->access_directory(share_data->save_path_nbvnet);
						share_data->access_directory(share_data->save_path_pcnbv);
						ofstream fout_grid(share_data->save_path + "/grid_" + to_string(m) + ".txt"); //MA-SCVP
						ofstream fout_grid_nbvnet(share_data->save_path_nbvnet + "/grid_" + to_string(m) + ".txt");
						ofstream fout_view_id(share_data->save_path_nbvnet + "/id_" + to_string(m) + ".txt");
						ofstream fout_view_ids(share_data->save_path + "/ids_" + to_string(m) + ".txt"); //MA-SCVP
						ofstream fout_cloud(share_data->save_path_pcnbv + "/cloud_" + to_string(m) + ".txt");
						ofstream fout_view_state(share_data->save_path + "/state_" + to_string(m) + ".txt"); //MA-SCVP
						ofstream fout_view_state_pcnbv(share_data->save_path_pcnbv + "/state_" + to_string(m) + ".txt");
						ofstream fout_view_score(share_data->save_path_pcnbv + "/score_" + to_string(m) + ".txt");
						double temp_double;
						int temp_int;
						//Copy MASCVP
						ifstream fin_grid(path + "grid_" + to_string(cas) + ".txt");
						while (fin_grid >> temp_double) fout_grid << temp_double << '\n';
						ifstream fin_state(path + "state_" + to_string(cas) + ".txt");
						while (fin_state >> temp_int) fout_view_state << temp_int << '\n';
						ifstream fin_id(path + "ids_" + to_string(cas) + ".txt");
						while (fin_id >> temp_int) fout_view_ids << temp_int << '\n';
						//Copy NBVNET
						ifstream fin_grid_nbvnet(path_nbvnet + "grid_" + to_string(cas) + ".txt");
						while (fin_grid_nbvnet >> temp_double) fout_grid_nbvnet << temp_double << '\n';
						ifstream fin_id_nbvnet(path_nbvnet + "id_" + to_string(cas) + ".txt");
						while (fin_id_nbvnet >> temp_int) fout_view_id << temp_int << '\n';
						//Copy PCNBV
						ifstream fin_cloud(path_pcnbv + "cloud_" + to_string(cas) + ".txt");
						int next_line = 0;
						while (fin_cloud >> temp_double) {
							fout_cloud << temp_double << (next_line % 3 == 2 ? '\n' : ' ');
							next_line++;
						}
						ifstream fin_state_pcnbv(path_pcnbv + "state_" + to_string(cas) + ".txt");
						while (fin_state_pcnbv >> temp_int) fout_view_state_pcnbv << temp_int << '\n';
						ifstream fin_score(path_pcnbv + "score_" + to_string(cas) + ".txt");
						while (fin_score >> temp_double) fout_view_score << temp_double << '\n';
					}
					delete labeler;
					delete share_data;
				}
			}
		}
	}
	else if (mode == CheckAll) {
		share_data = new Share_Data("../DefaultConfiguration.yaml", "", -1, -1);
		for (int need_case_1 = 32; need_case_1 >= 8; need_case_1 -= 8) if (need_case_1 != 24 && need_case_1 != 16) {
			for (int i = 0; i < names.size(); i++) {
				for (int k = 0; k < 8; k++) {
					cout << "checking need_case_1 " << need_case_1 << " , object " << names[i] << " , rotate " << k << " :" << endl;
					cout << "checking LongTailSample:" << endl;
					string path = share_data->pre_path + to_string(need_case_1) + "/MASCVP_LongTailSample/" + names[i] + "/rotate_" + to_string(k) + "/";
					string path_nbvnet = share_data->pre_path + to_string(need_case_1) + "/NBVNET_LongTailSample/" + names[i] + "/rotate_" + to_string(k) + "/";
					string path_pcnbv = share_data->pre_path + to_string(need_case_1) + "/PCNBV_LongTailSample/" + names[i] + "/rotate_" + to_string(k) + "/";
					int cas_num = 0;
					long long case_temp;
					ifstream fin_case_num(share_data->pre_path + "NBV_GT_label/" + names[i] + "/rotate_" + to_string(k) + "/" + to_string(need_case_1) + "_longtail_sample.txt");
					while (fin_case_num >> case_temp) cas_num++;
					fin_case_num.close();
					cout << "case num is "<< cas_num << endl;
					if (cas_num == 0) cout << "case missing wrong! " << endl;
					for (int j = 0; j < cas_num; j++) {
						double temp;
						int num_gird, num_id, num_cloud, num_state, num_score;
						//Check MASCVP
						ifstream fin_grid(path + "grid_" + to_string(j) + ".txt");
						num_gird = 0;
						while (fin_grid >> temp) num_gird++;
						num_state = 0;
						ifstream fin_state(path + "state_" + to_string(j) + ".txt");
						while (fin_state >> temp) num_state++;
						num_id = 0;
						ifstream fin_id(path + "ids_" + to_string(j) + ".txt");
						while (fin_id >> temp) num_id++;
						//cout << num_gird << " " << num_id << endl;
						if (num_gird != 32 * 32 * 32 || num_state != 32 || num_id == 0) {
							cout << "MASCVP Object case " << j << " wrong." << endl;
						}
						//Check NBVNET
						ifstream fin_grid_nbvnet(path_nbvnet + "grid_" + to_string(j) + ".txt");
						num_gird = 0;
						while (fin_grid_nbvnet >> temp) num_gird++;
						num_id = 0;
						ifstream fin_id_nbvnet(path_nbvnet + "id_" + to_string(j) + ".txt");
						while (fin_id_nbvnet >> temp) num_id++;
						//cout << num_gird << " " << num_id << endl;
						if (num_gird != 32 * 32 * 32 || num_id != 1) {
							cout << "NBVNET Object case " << j << " wrong." << endl;
						}
						//Check PCNBV
						num_cloud = 0;
						ifstream fin_cloud(path_pcnbv + "cloud_" + to_string(j) + ".txt");
						while (fin_cloud >> temp) num_cloud++;
						num_state = 0;
						ifstream fin_state_pcnbv(path_pcnbv + "state_" + to_string(j) + ".txt");
						while (fin_state_pcnbv >> temp) num_state++;
						num_score = 0;
						ifstream fin_score(path_pcnbv + "score_" + to_string(j) + ".txt");
						while (fin_score >> temp) num_score++;
						//cout << num_state << " " << num_score << endl;
						if (num_state != 32 || num_score != 32 || num_cloud != 512 * 3) {
							cout << "PCNBV Object case " << j << " wrong." << endl;
						}
					}
					cout << "checking NBVSample:" << endl;
					path = share_data->pre_path + to_string(need_case_1) + "/MASCVP_NBVSample/" + names[i] + "/rotate_" + to_string(k) + "/";
					path_nbvnet = share_data->pre_path + to_string(need_case_1) + "/NBVNET_NBVSample/" + names[i] + "/rotate_" + to_string(k) + "/";
					path_pcnbv = share_data->pre_path + to_string(need_case_1) + "/PCNBV_NBVSample/" + names[i] + "/rotate_" + to_string(k) + "/";
					cas_num = 0;
					fin_case_num.open(share_data->pre_path + "NBV_GT_label/" + names[i] + "/rotate_" + to_string(k) + "/" + to_string(need_case_1) + "_nbv_sample.txt");
					while (fin_case_num >> case_temp) cas_num++;
					fin_case_num.close();
					cout << "case num is " << cas_num << endl;
					if (cas_num == 0) cout << "case missing wrong! " << endl;
					for (int j = 0; j < cas_num; j++) {
						double temp;
						int num_gird, num_id, num_cloud, num_state, num_score;
						//Check MASCVP
						ifstream fin_grid(path + "grid_" + to_string(j) + ".txt");
						num_gird = 0;
						while (fin_grid >> temp) num_gird++;
						num_state = 0;
						ifstream fin_state(path + "state_" + to_string(j) + ".txt");
						while (fin_state >> temp) num_state++;
						num_id = 0;
						ifstream fin_id(path + "ids_" + to_string(j) + ".txt");
						while (fin_id >> temp) num_id++;
						//cout << num_gird << " " << num_id << endl;
						if (num_gird != 32 * 32 * 32 || num_state != 32 || num_id == 0) {
							cout << "MASCVP Object case " << j << " wrong." << endl;
						}
						//Check NBVNET
						ifstream fin_grid_nbvnet(path_nbvnet + "grid_" + to_string(j) + ".txt");
						num_gird = 0;
						while (fin_grid_nbvnet >> temp) num_gird++;
						num_id = 0;
						ifstream fin_id_nbvnet(path_nbvnet + "id_" + to_string(j) + ".txt");
						while (fin_id_nbvnet >> temp) num_id++;
						//cout << num_gird << " " << num_id << endl;
						if (num_gird != 32 * 32 * 32 || num_id != 1) {
							cout << "NBVNET Object case " << j << " wrong." << endl;
						}
						//Check PCNBV
						num_cloud = 0;
						ifstream fin_cloud(path_pcnbv + "cloud_" + to_string(j) + ".txt");
						while (fin_cloud >> temp) num_cloud++;
						num_state = 0;
						ifstream fin_state_pcnbv(path_pcnbv + "state_" + to_string(j) + ".txt");
						while (fin_state_pcnbv >> temp) num_state++;
						num_score = 0;
						ifstream fin_score(path_pcnbv + "score_" + to_string(j) + ".txt");
						while (fin_score >> temp) num_score++;
						//cout << num_state << " " << num_score << endl;
						if (num_state != 32 || num_score != 32 || num_cloud != 512 * 3) {
							cout << "PCNBV Object case " << j << " wrong." << endl;
						}
					}
				}
			}
		}
	}
	cout << "System over." << endl;
	return 0;
}

/*
Armadillo
Asian_Dragon
Dragon
Stanford_Bunny
Happy_Buddha
Thai_Statue
Lucy
LM1
LM2
LM3
LM4
LM5
LM6
LM7
LM8
LM9
LM10
LM11
LM12
obj_000001
obj_000002
obj_000003
obj_000004
obj_000005
obj_000006
obj_000007
obj_000008
obj_000009
obj_000010
obj_000011
obj_000012
obj_000013
obj_000014
obj_000015
obj_000016
obj_000017
obj_000018
obj_000019
obj_000020
obj_000021
obj_000022
obj_000023
obj_000024
obj_000025
obj_000026
obj_000027
obj_000028
obj_000029
obj_000030
obj_000031
*/
