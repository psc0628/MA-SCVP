#include <windows.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
typedef unsigned long long pop_t;

using namespace std;

#include "Share_Data.hpp"
#include "View_Space.hpp"
#include "Information.hpp"

//Virtual_Perception_3D.hpp
void precept_thread_process(int i, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, octomap::point3d* _origin, octomap::point3d* _end, Eigen::Matrix4d* _view_pose_world, octomap::ColorOcTree* _ground_truth_model, Share_Data* share_data);

class Perception_3D {
public:
	Share_Data* share_data;
	octomap::ColorOcTree* ground_truth_model;
	int full_voxels;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

	Perception_3D(Share_Data* _share_data) {
		share_data = _share_data;
		ground_truth_model = new octomap::ColorOcTree(share_data->ground_truth_resolution);
		for (octomap::ColorOcTree::leaf_iterator it = share_data->ground_truth_model->begin_leafs(), end = share_data->ground_truth_model->end_leafs(); it != end; ++it) {
			ground_truth_model->setNodeValue(it.getX(), it.getY(), it.getZ(), it->getLogOdds(), true);
			ground_truth_model->setNodeColor(it.getX(), it.getY(), it.getZ(), 255, 0, 0);
		}
		for (double x = share_data->object_center_world(0) - 0.2; x <= share_data->object_center_world(0) + 0.2; x += share_data->ground_truth_resolution)
			for (double y = share_data->object_center_world(1) - 0.2; y <= share_data->object_center_world(1) + 0.2; y += share_data->ground_truth_resolution) {
				double z = share_data->min_z_table;
				ground_truth_model->setNodeValue(x, y, z, ground_truth_model->getProbHitLog(), true);
				ground_truth_model->setNodeColor(x, y, z, 0, 0, 255);
			}
		//ground_truth_model->write(share_data->save_path + "/GT_table.ot");
		full_voxels = 0;
		for (octomap::ColorOcTree::leaf_iterator it = ground_truth_model->begin_leafs(), end = ground_truth_model->end_leafs(); it != end; ++it) {
			full_voxels++;
		}
	}

	~Perception_3D() {
		;
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
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr no_table(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud = temp;
		cloud->is_dense = false;
		no_table->is_dense = false;
		cloud->points.resize(full_voxels);
		no_table->points.resize(full_voxels);
		auto ptr = cloud->points.begin();
		auto pt = no_table->points.begin();
		int vaild_point = 0;
		int table_point = 0;
		auto p = cloud_parallel->points.begin();
		for (int i = 0; i < cloud_parallel->points.size(); i++, p++)
		{
			if ((*p).x == 0 && (*p).y == 0 && (*p).z == 0) continue;
			if ((*p).z > share_data->min_z_table + share_data->ground_truth_resolution) {
				(*pt).x = (*p).x;
				(*pt).y = (*p).y;
				(*pt).z = (*p).z;
				(*pt).b = (*p).b;
				(*pt).g = (*p).g;
				(*pt).r = (*p).r;
				table_point++;
				pt++;
			}
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
		no_table->width = table_point;
		cloud->height = 1;
		no_table->height = 1;
		cloud->points.resize(vaild_point);
		no_table->points.resize(table_point);
		//记录当前采集点云
		share_data->vaild_clouds++;
		share_data->clouds.push_back(cloud);
		//旋转至世界坐标系
		*share_data->cloud_final += *no_table;
		//cout << "virtual cloud num is " << vaild_point << endl;
		//cout << "virtual cloud table num is " << table_point << endl;
		//cout << "virtual cloud get with executed time " << clock() - now_time << " ms." << endl;
		if (share_data->show) { //显示成像点云
			pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Camera"));
			viewer1->setBackgroundColor(255, 255, 255);
			//viewer1->addCoordinateSystem(0.1);
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
			
			/* // show rays
			//viewer1->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_ground_truth, "cloud_gt");
			viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(-1));
			viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(-1));
			viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(-1));
			for (int i = 0; i < cloud->points.size(); i+=1000) {
				viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z), 0.75, 0.75, 0, "point" + to_string(i));
			}
			viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
			*/
			while (!viewer1->wasStopped())
			{
				viewer1->spinOnce(100);
				boost::this_thread::sleep(boost::posix_time::microseconds(100000));
			}
		}

		cloud_parallel->~PointCloud();
		no_table->~PointCloud();
		cout <<"Virtual cloud getted with time "<< clock() - now_time<<" ms." << endl;
		return true;
	}
};

void precept_thread_process(int i, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, octomap::point3d* _origin, octomap::point3d* _end, Eigen::Matrix4d* _view_pose_world, octomap::ColorOcTree* _ground_truth_model, Share_Data* share_data) {
	//num++;
	octomap::point3d origin = *_origin;
	Eigen::Matrix4d view_pose_world = *_view_pose_world;
	octomap::ColorOcTree* ground_truth_model = _ground_truth_model;
	pcl::PointXYZRGB point;
	point.x = 0; point.y = 0; point.z = 0;
	//投影检测是否在成像范围内
	Eigen::Vector4d end_3d(_end->x(), _end->y(), _end->z(), 1);
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
		cout << "view in the object. check!" << endl;
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

//Global_Path_Planner.hpp
class Global_Path_Planner {
public:
	Share_Data* share_data;
	View_Space* view_space;
	View* now_view;
	bool solved;
	long long n,m;
	map<int, int>* view_id_in;
	map<int, int>* view_id_out;
	vector<vector<double>> graph;
	vector<vector<double>> dp;
	vector<vector<int>> path;
	double total_shortest;
	vector<int> global_path;

	Global_Path_Planner(Share_Data* _share_data, View_Space* _view_space, View* _now_view, vector<int>& view_set_label) {
		share_data = _share_data;
		view_space = _view_space;
		now_view = _now_view;
		solved = false;
		//构造下标映射
		view_id_in = new map<int, int>();
		view_id_out = new map<int, int>();
		(*view_id_in)[now_view->id] = 0;
		(*view_id_out)[0] = now_view->id;
		for (int i = 0; i < view_set_label.size(); i++) {
			(*view_id_in)[view_set_label[i]] = i+1;
			(*view_id_out)[i+1] = view_set_label[i];
		}
		//节点数与方案数
		n = view_set_label.size() + 1;
		m = 1LL << n;
		//local path 完全无向图
		graph.resize(n);
		for (int i = 0; i < n; i++)
			graph[i].resize(n);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++) {
				//交换id
				int u = (*view_id_out)[i];
				int v = (*view_id_out)[j];
				//求两点路径
				pair<int, double> local_path = get_local_path(view_space->views[u].init_pos.eval(), view_space->views[v].init_pos.eval(), view_space->object_center_world.eval(), view_space->predicted_size * sqrt(2)); //包围盒半径是半边长的根号2倍
				if (local_path.first < 0) cout << "local path wrong." << endl;
				graph[i][j] = local_path.second;
			}
		//初始化dp
		dp.resize(m);
		for (int i = 0; i < m; i++)
			dp[i].resize(n);
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				dp[i][j] = 1e10;
		dp[1][0] = 0;
		//初始化路径记录
		path.resize(m);
		for (int i = 0; i < m; i++)
			path[i].resize(n);
		cout << "Global_Path_Planner inited." << endl;
	}

	~Global_Path_Planner() {
		delete view_id_in;
		delete view_id_out;
		graph.clear();
		dp.clear();
		path.clear();
		global_path.clear();
	}

	double solve() {
		double now_time = clock();
		for (int i = 0; i < m; i++)  // i代表的是一个方案的集合，其中每个位置的0/1代表没有/有经过这个点(m=1<<n)
		{
			for (int j = 0; j < n; j++)  //枚举当前到了哪个点
			{
				if ((i >> j) & 1)  //如果i到达过j
				{
					for (int k = 0; k < n; k++)  //枚举到达j的点
					{
						if (i - (1 << j) >> k & 1)  //去除集合j的集合i
						{
							//如果更短则更新并记录转移路径
							if (dp[i][j] >= dp[i - (1LL << j)][k] + graph[k][j]) {
								dp[i][j] = dp[i - (1LL << j)][k] + graph[k][j];
								path[i][j] = k;
							}
							// dp[i][j] = min(dp[i][j], dp[i - (1 << j)][k] + mp[k][j]);
						}
					}
				}
			}
		}
		//默认起点为0,遍历每个可能终点找到最短哈密尔顿路径（全覆盖方案）
		int end_node;
		total_shortest = 1e20;
		for (int i = 1; i < n; i++) {
			if (total_shortest > dp[m - 1][i]) {
				total_shortest = dp[m - 1][i];
				end_node = i;
			}
		}
		//输出路径
		for (int i = (1 << n) - 1, j = end_node; i > 0;) {
			//注意下标交换
			global_path.push_back((*view_id_out)[j]);
			int ii = i - (1 << j);
			int jj = path[i][j];
			i = ii, j = jj;
		}
		//遍历路径需反向
		reverse(global_path.begin(), global_path.end());
		solved = true;
		double cost_time = clock() - now_time;
		cout << "Global Path length " << total_shortest << " getted with executed time " << cost_time << " ms." << endl;
		//保存
		share_data->access_directory(share_data->save_path + "/movement");
		ofstream fout_global_path(share_data->save_path + "/movement/global_path.txt");
		fout_global_path << total_shortest << '\t' << cost_time << '\t' << endl;
		return total_shortest;
	}

	vector<int> get_path_id_set() {
		if (!solved) cout << "call solve() first" << endl;
		cout << "Node ids on global_path form start to end are: ";
		for (int i = 0; i < global_path.size(); i++)
			cout << global_path[i] << " ";
		cout << endl;
		//删除出发点
		vector<int> ans;
		ans = global_path;
		ans.erase(ans.begin());
		return ans;
	}
};

#define Over 0
#define WaitData 1
#define WaitViewSpace 2
#define WaitInformation 3
#define WaitMoving 4

//NVB_Planner.hpp
void save_cloud_mid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, string name);
void create_view_space(View_Space** now_view_space, View* now_best_view, Share_Data* share_data, int iterations);
void create_views_information(Views_Information** now_views_infromation, View* now_best_view, View_Space* now_view_space, Share_Data* share_data, int iterations);
void move_robot(View* now_best_view, View_Space* now_view_space, Share_Data* share_data);
void show_cloud(pcl::visualization::PCLVisualizer::Ptr viewer);

class NBV_Planner
{
public:
	atomic<int> status;
	int iterations;
	Perception_3D* percept;
	Voxel_Information* voxel_information;
	View_Space* now_view_space;
	Views_Information* now_views_infromation;
	View* now_best_view;
	Share_Data* share_data;
	pcl::visualization::PCLVisualizer::Ptr viewer;

	~NBV_Planner() {
		delete percept;
		delete now_best_view;
		delete voxel_information;
		delete now_view_space;
		//只有搜索生成了information
		if (share_data->method_of_IG == 6 || share_data->method_of_IG == 7 || share_data->method_of_IG == 8 || share_data->method_of_IG == 9);
		else delete now_views_infromation;
	}

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

	NBV_Planner(Share_Data* _share_data, int _status = WaitData) {
		share_data = _share_data;
		iterations = 0;
		status = _status;
		share_data->now_view_space_processed = false;
		share_data->now_views_infromation_processed = false;
		share_data->move_on = false;
		voxel_information = new Voxel_Information(share_data->p_unknown_lower_bound, share_data->p_unknown_upper_bound);
		voxel_information->init_mutex_views(share_data->num_of_views);
		//初始化GT
		//share_data->access_directory(share_data->save_path);
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
		//检查物体大小，统一缩放为0.15m左右
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
		/*//二分查找BBX半径，以BBX内点的个数比率达到0.90-0.95为终止条件
		double l = 0, r = 0, mid;
		for (auto& ptr : points) {
			r = max(r, (object_center_world - ptr).norm());
		}
		mid = (l + r) / 2;
		double precent = check_size(mid, object_center_world, points);
		double pre_precent = precent;
		while (precent > 0.95 || precent < 1.0) {
			if (precent > 0.95) {
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
		predicted_size *= 17.0 / 16.0;
		//predicted_size = 0.1;

		double scale = 1.0;
		if (share_data->mp_scale.find(share_data->name_of_pcd) != share_data->mp_scale.end()) {
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
		share_data->octo_model->setOccupancyThres(0.65);
		share_data->GT_sample = new octomap::ColorOcTree(share_data->octomap_resolution);

		//转换点云
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
				if (voxel_sp == NULL) {
					share_data->GT_sample->setNodeValue(key_sp, share_data->GT_sample->getProbHitLog(), true);
					share_data->GT_sample->integrateNodeColor(key_sp, (*ptr).r, (*ptr).g, (*ptr).b);
				}
			}
			ptr++;
		}
		//cout << min_z << endl;
		//pcl::io::savePCDFile<pcl::PointXYZRGB>("C:\\Users\\yixinizhu\\Desktop\\" + share_data->name_of_pcd + ".pcd", *share_data->cloud_ground_truth);
		//记录桌面
		share_data->min_z_table = min_z - share_data->ground_truth_resolution;

		share_data->ground_truth_model->updateInnerOccupancy();
		//share_data->ground_truth_model->write(share_data->save_path + "/GT.ot");
		//GT_sample_voxels
		share_data->GT_sample->updateInnerOccupancy();
		//share_data->GT_sample->write(share_data->save_path + "/GT_sample.ot");
		share_data->init_voxels = 0;
		int full_voxels = 0;
		//在sample中统计总个数
		for (octomap::ColorOcTree::leaf_iterator it = share_data->GT_sample->begin_leafs(), end = share_data->GT_sample->end_leafs(); it != end; ++it){
			if(it.getY() > share_data->min_z_table + share_data->octomap_resolution)
				share_data->init_voxels++;
			full_voxels++;
		}
		cout << "Map_GT_sample has voxels " << share_data->init_voxels << endl;
		cout << "Map_GT_sample has voxels with bottom " << full_voxels << endl;
		share_data->init_voxels = full_voxels;
		//ofstream fout_sample(share_data->save_path + "/GT_sample_voxels.txt");
		//fout_sample << share_data->init_voxels << endl;
		//在GT中统计总个数
		share_data->cloud_points_number = 0;
		for (octomap::ColorOcTree::leaf_iterator it = share_data->ground_truth_model->begin_leafs(), end = share_data->ground_truth_model->end_leafs(); it != end; ++it) {
			share_data->cloud_points_number++;
		}
		cout << "Map_GT has voxels " << share_data->cloud_points_number << endl;
		//ofstream fout_gt(share_data->save_path + "/GT_voxels.txt");
		//fout_gt << share_data->cloud_points_number << endl;

		//初始化viewspace
		int first_view_id = share_data->first_view_id;
		now_view_space = new View_Space(iterations, share_data, voxel_information, share_data->cloud_ground_truth, first_view_id);
		//设置初始视点为统一的位置
		//if (share_data->method_of_IG == 7) { //SC-NET
			now_view_space->views[first_view_id].vis++;
			now_best_view = new View(now_view_space->views[first_view_id]);
		//}
		/*else {
			now_view_space->views[0].vis++;
			now_best_view = new View(now_view_space->views[0]);
		}*/
		//运动代价：视点id，当前代价，总体代价
		share_data->movement_cost = 0;
		//share_data->access_directory(share_data->save_path + "/movement");
		//ofstream fout_move(share_data->save_path + "/movement/path" + to_string(-1) + ".txt");
		//fout_move << 0 << '\t' << 0.0 << '\t' << 0.0 << endl;
		now_best_view->get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
		Eigen::Matrix4d view_pose_world = (share_data->now_camera_pose_world * now_best_view->pose.inverse()).eval();
		//相机类初始化
		percept = new Perception_3D(share_data);
		if (share_data->show) { //显示BBX、相机位置、GT
			pcl::visualization::PCLVisualizer::Ptr viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("Iteration"));
			viewer->setBackgroundColor(0, 0, 0);
			viewer->addCoordinateSystem(0.1);
			viewer->initCameraParameters();
			//第一帧相机位置
			Eigen::Vector4d X(0.05, 0, 0, 1);
			Eigen::Vector4d Y(0, 0.05, 0, 1);
			Eigen::Vector4d Z(0, 0, 0.05, 1);
			Eigen::Vector4d O(0, 0, 0, 1);
			X = view_pose_world * X;
			Y = view_pose_world * Y;
			Z = view_pose_world * Z;
			O = view_pose_world * O;
			viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(-1));
			viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(-1));
			viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(-1));
			//test_viewspace
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr test_viewspace(new pcl::PointCloud<pcl::PointXYZRGB>);
			test_viewspace->is_dense = false;
			test_viewspace->points.resize(now_view_space->views.size());
			auto pt = test_viewspace->points.begin();
			for (int i = 0; i < now_view_space->views.size(); i++, pt++) {
				(*pt).x = now_view_space->views[i].init_pos(0);
				(*pt).y = now_view_space->views[i].init_pos(1);
				(*pt).z = now_view_space->views[i].init_pos(2);
				//第一次显示所有点为白色
				(*pt).r = 255, (*pt).g = 255, (*pt).b = 255;
			}
			viewer->addPointCloud<pcl::PointXYZRGB>(test_viewspace, "test_viewspace");
			now_view_space->add_bbx_to_cloud(viewer);
			viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_ground_truth, "cloud_ground_truth");
			while (!viewer->wasStopped())
			{
				viewer->spinOnce(100);
				boost::this_thread::sleep(boost::posix_time::microseconds(100000));
			}
		}
	}

	int plan() {
		switch (status)
		{
		case Over:
			break;
		case WaitData:
			if (percept->precept(now_best_view)) {
				thread next_view_space(create_view_space, &now_view_space, now_best_view, share_data, iterations);
				next_view_space.detach();
				status = WaitViewSpace;
			}
			break;
		case WaitViewSpace:
			if (share_data->now_view_space_processed) {
				thread next_views_information(create_views_information, &now_views_infromation, now_best_view, now_view_space, share_data, iterations);
				next_views_information.detach();
				status = WaitInformation;
			}
			break;
		case WaitInformation:
			if (share_data->now_views_infromation_processed) {
				if (share_data->method_of_IG == 8) { //Random
					srand(clock());
					int next_id = rand() % share_data->num_of_views; //32是因数
					while (now_view_space->views[next_id].vis) { //随机到一个没有访问过的
						next_id = rand() % share_data->num_of_views;
					}
					now_view_space->views[next_id].vis++;
					now_view_space->views[next_id].can_move = true;
					delete now_best_view;
					now_best_view = new View(now_view_space->views[next_id]);
					cout << "choose the " << next_id << "th view." << endl;
					share_data->movement_cost += now_best_view->robot_cost;
					share_data->access_directory(share_data->save_path + "/movement");
					ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
					fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
				}
				else if (share_data->method_of_IG == 6) { //NBV-NET
					share_data->access_directory(share_data->nbv_net_path + "/log");
					ifstream ftest;
					do {
						ftest.open(share_data->nbv_net_path + "/log/ready.txt");
					} while (!ftest.is_open());
					ftest.close();
					ifstream fin(share_data->nbv_net_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + '_' + to_string(iterations) + ".txt");
					int id;
					fin >> id;
					cout <<"next view id is "<< id << endl;
					now_view_space->views[id].vis++;
					delete now_best_view;
					now_best_view = new View(now_view_space->views[id]);
					//运动代价：视点id，当前代价，总体代价
					share_data->movement_cost += now_best_view->robot_cost;
					share_data->access_directory(share_data->save_path + "/movement");
					ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
					fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
					//更新标志文件
					this_thread::sleep_for(chrono::seconds(1));
					int removed = remove((share_data->nbv_net_path + "/log/ready.txt").c_str());
					if (removed!=0) cout << "cannot remove ready.txt." << endl;
				}
				else if (share_data->method_of_IG == 9) { //PCNBV
					share_data->access_directory(share_data->pcnbv_path + "/log");
					ifstream ftest;
					do {
						ftest.open(share_data->pcnbv_path + "/log/ready.txt");
					} while (!ftest.is_open());
					ftest.close();
					ifstream fin(share_data->pcnbv_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + '_' + to_string(iterations) + ".txt");
					int id;
					fin >> id;
					cout << "next view id is " << id << endl;
					now_view_space->views[id].vis++;
					delete now_best_view;
					now_best_view = new View(now_view_space->views[id]);
					//运动代价：视点id，当前代价，总体代价
					share_data->movement_cost += now_best_view->robot_cost;
					share_data->access_directory(share_data->save_path + "/movement");
					ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
					fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
					//更新标志文件
					this_thread::sleep_for(chrono::seconds(1));
					int removed = remove((share_data->pcnbv_path + "/log/ready.txt").c_str());
					if (removed != 0) cout << "cannot remove ready.txt." << endl;
				}
				else if (share_data->method_of_IG == 7) { //SCVP
					if (iterations == 0 + share_data->num_of_nbvs_combined) {
						share_data->access_directory(share_data->sc_net_path + "/log");
						ifstream ftest;
						do {
							ftest.open(share_data->sc_net_path + "/log/ready.txt");
						} while (!ftest.is_open());
						ftest.close();
						ifstream fin(share_data->sc_net_path + "/log/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + ".txt");
						vector<int> view_set_label;
						int rest_view_id;
						while (fin >> rest_view_id) {
							view_set_label.push_back(rest_view_id);
						}
						//若有,则删除已访问视点
						set<int> vis_view_ids;
						for (int i = 0; i < now_view_space->views.size(); i++) {
							if (now_view_space->views[i].vis)	vis_view_ids.insert(i);
						}
						for (auto it = view_set_label.begin(); it != view_set_label.end(); ) {
							if(vis_view_ids.count((*it))){
								it = view_set_label.erase(it);
							}
							else {
								it++;
							}
						}
						//保存一下最终视点个数
						ofstream fout_all_needed_views(share_data->save_path + "/all_needed_views.txt");
						fout_all_needed_views << view_set_label.size() + 1 + share_data->num_of_nbvs_combined << endl;
						cout<< "All_needed_views is "<< view_set_label.size() + 1 + share_data->num_of_nbvs_combined << endl;
						//如果没有视点需要，这就直接退出
						if (view_set_label.size() == 0) {
							//更新标志文件
							this_thread::sleep_for(chrono::seconds(1));
							int removed = remove((share_data->sc_net_path + "/log/ready.txt").c_str());
							if (removed != 0) cout << "cannot remove ready.txt." << endl;
							//系统退出
							share_data->over = true;
							status = WaitMoving;
							break;
						}
						//规划路径
						Global_Path_Planner* gloabl_path_planner = new Global_Path_Planner(share_data, now_view_space, now_best_view, view_set_label);
						gloabl_path_planner->solve();
						//保存路径
						share_data->view_label_id = gloabl_path_planner->get_path_id_set();
						delete now_best_view;
						now_best_view = new View(now_view_space->views[share_data->view_label_id[iterations - share_data->num_of_nbvs_combined]]);
						//运动代价：视点id，当前代价，总体代价
						share_data->movement_cost += now_best_view->robot_cost;
						share_data->access_directory(share_data->save_path + "/movement");
						ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
						fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
						//更新标志文件
						this_thread::sleep_for(chrono::seconds(1));
						int removed = remove((share_data->sc_net_path + "/log/ready.txt").c_str());
						if (removed != 0) cout << "cannot remove ready.txt." << endl;
					}
					else {
						if (iterations == share_data->view_label_id.size() + share_data->num_of_nbvs_combined) {
							share_data->over = true;
							status = WaitMoving;
							break;
						}
						delete now_best_view;
						now_best_view = new View(now_view_space->views[share_data->view_label_id[iterations - share_data->num_of_nbvs_combined]]);
						//运动代价：视点id，当前代价，总体代价
						share_data->movement_cost += now_best_view->robot_cost;
						share_data->access_directory(share_data->save_path + "/movement");
						ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
						fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
					}
				}
				else{//搜索算法
					//对视点排序
					sort(now_view_space->views.begin(), now_view_space->views.end(), view_utility_compare);
					/*if (share_data->sum_local_information == 0) {
						cout << "randomly choose a view" << endl;
						srand(clock());
						random_shuffle(now_view_space->views.begin(), now_view_space->views.end());
					}*/
					//informed_viewspace
					if (share_data->show) { //显示BBX与相机位置
						pcl::visualization::PCLVisualizer::Ptr viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("Iteration" + to_string(iterations)));
						viewer->setBackgroundColor(0, 0, 0);
						viewer->addCoordinateSystem(0.1);
						viewer->initCameraParameters();
						//test_viewspace
						pcl::PointCloud<pcl::PointXYZRGB>::Ptr test_viewspace(new pcl::PointCloud<pcl::PointXYZRGB>);
						test_viewspace->is_dense = false;
						test_viewspace->points.resize(now_view_space->views.size());
						auto ptr = test_viewspace->points.begin();
						int needed = 0;
						for (int i = 0; i < now_view_space->views.size(); i++) {
							(*ptr).x = now_view_space->views[i].init_pos(0);
							(*ptr).y = now_view_space->views[i].init_pos(1);
							(*ptr).z = now_view_space->views[i].init_pos(2);
							//访问过的点记录为蓝色
							if (now_view_space->views[i].vis) (*ptr).r = 0, (*ptr).g = 0, (*ptr).b = 255;
							//在网络流内的设置为黄色
							else if (now_view_space->views[i].in_coverage[iterations] && i < now_view_space->views.size() / 10) (*ptr).r = 255, (*ptr).g = 255, (*ptr).b = 0;
							//在网络流内的设置为绿色
							else if (now_view_space->views[i].in_coverage[iterations]) (*ptr).r = 255, (*ptr).g = 0, (*ptr).b = 0;
							//前10%的权重的点设置为蓝绿色
							else if (i < now_view_space->views.size() / 10) (*ptr).r = 0, (*ptr).g = 255, (*ptr).b = 255;
							//其余点白色
							else (*ptr).r = 255, (*ptr).g = 255, (*ptr).b = 255;
							ptr++;
							needed++;
						}
						test_viewspace->points.resize(needed);
						viewer->addPointCloud<pcl::PointXYZRGB>(test_viewspace, "test_viewspace");
						bool best_have = false;
						for (int i = 0; i < now_view_space->views.size(); i++) {
							if (now_view_space->views[i].vis) {
								now_view_space->views[i].get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
								Eigen::Matrix4d view_pose_world = (share_data->now_camera_pose_world * now_view_space->views[i].pose.inverse()).eval();
								Eigen::Vector4d X(0.03, 0, 0, 1);
								Eigen::Vector4d Y(0, 0.03, 0, 1);
								Eigen::Vector4d Z(0, 0, 0.03, 1);
								Eigen::Vector4d O(0, 0, 0, 1);
								X = view_pose_world * X;
								Y = view_pose_world * Y;
								Z = view_pose_world * Z;
								O = view_pose_world * O;
								viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i));
								viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i));
								viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i));
							}
							else if (!best_have) {
								now_view_space->views[i].get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
								Eigen::Matrix4d view_pose_world = (share_data->now_camera_pose_world * now_view_space->views[i].pose.inverse()).eval();
								Eigen::Vector4d X(0.08, 0, 0, 1);
								Eigen::Vector4d Y(0, 0.08, 0, 1);
								Eigen::Vector4d Z(0, 0, 0.08, 1);
								Eigen::Vector4d O(0, 0, 0, 1);
								X = view_pose_world * X;
								Y = view_pose_world * Y;
								Z = view_pose_world * Z;
								O = view_pose_world * O;
								viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i));
								viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i));
								viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i));
								best_have = true;
							}
						}
						viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_final, "cloud_now_itreation");
						while (!viewer->wasStopped())
						{
							viewer->spinOnce(100);
							boost::this_thread::sleep(boost::posix_time::microseconds(100000));
						}
					}
					double max_utility = -1;
					for (int i = 0; i < now_view_space->views.size(); i++) {
						cout << "checking view " << i << endl;
						if (now_view_space->views[i].vis) continue;
						//if (!now_view_space->views[i].can_move) continue;
						delete now_best_view;
						now_best_view = new View(now_view_space->views[i]);
						max_utility = now_best_view->final_utility;
						now_view_space->views[i].vis++;
						now_view_space->views[i].can_move = true;
						cout << "choose the " << i << "th view." << endl;
						//运动代价：视点id，当前代价，总体代价
						share_data->movement_cost += now_best_view->robot_cost;
						share_data->access_directory(share_data->save_path + "/movement");
						ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
						fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
						break;
					}
					if (max_utility == -1) {
						cout << "Can't move to any viewport.Stop." << endl;
						status = Over;
						break;
					}
					cout << " next best view pos is (" << now_best_view->init_pos(0) << ", " << now_best_view->init_pos(1) << ", " << now_best_view->init_pos(2) << ")" << endl;
					cout << " next best view final_utility is " << now_best_view->final_utility << endl;
				}
				//进入运动模块
				thread next_moving(move_robot, now_best_view, now_view_space, share_data);
				next_moving.detach();
				status = WaitMoving;
			}
			break;
		case WaitMoving:
			//virtual move
			if (share_data->over) {
				cout << "Progress over.Saving octomap and cloud." << endl;
				status = Over;
				break;
			}
			if (share_data->move_on) {
				iterations++;
				share_data->now_view_space_processed = false;
				share_data->now_views_infromation_processed = false;
				share_data->move_on = false;
				status = WaitData;
			}
			break;
		}
		return status;
	}

	string out_status() {
		string status_string;
		switch (status)
		{
		case Over:
			status_string = "Over";
			break;
		case WaitData:
			status_string = "WaitData";
			break;
		case WaitViewSpace:
			status_string = "WaitViewSpace";
			break;
		case WaitInformation:
			status_string = "WaitInformation";
			break;
		case WaitMoving:
			status_string = "WaitMoving";
			break;
		}
		return status_string;
	}
};

atomic<bool> stop = false;		//控制程序结束
Share_Data* share_data;			//共享数据区指针
NBV_Planner* nbv_plan;

void save_cloud_mid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, string name) {
	//保存中间的点云的线程，目前不检查是否保存完毕
	share_data->save_cloud_to_disk(cloud, "/clouds", name);
	cout << name << " saved" << endl;
}

void create_view_space(View_Space** now_view_space, View* now_best_view, Share_Data* share_data, int iterations) {
	//计算关键帧相机位姿
	share_data->now_camera_pose_world = (share_data->now_camera_pose_world * now_best_view->pose.inverse()).eval();;
	//处理viewspace,如果不需要评估并且是one-shot路径就不更新OctoMap
	if (share_data->evaluate_one_shot == 0 && share_data->method_of_IG == 7 && iterations > 0 + share_data->num_of_nbvs_combined);
	else (*now_view_space)->update(iterations, share_data, share_data->cloud_final, share_data->clouds[iterations]);
	//保存中间迭代结果
	if(share_data->is_save)	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_mid(new pcl::PointCloud<pcl::PointXYZRGB>);
		*cloud_mid = *share_data->cloud_final;
		thread save_mid(save_cloud_mid, cloud_mid, "pointcloud" + to_string(iterations));
		save_mid.detach();
	}
	//更新标志位
	share_data->now_view_space_processed = true;
}

void create_views_information(Views_Information** now_views_infromation, View* now_best_view, View_Space* now_view_space, Share_Data* share_data, int iterations) {
	if (share_data->method_of_IG == 8) { //Random
		;
	}
	else if (share_data->method_of_IG == 6) { //NBV-NET
		//octotree
		share_data->access_directory(share_data->nbv_net_path + "/data");
		ofstream fout(share_data->nbv_net_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) +'_'+ to_string(iterations) + ".txt");
		for (int i = 0; i < 32; i++)
			for (int j = 0; j < 32; j++)
				for (int k = 0; k < 32; k++)
				{
					double x = share_data->object_center_world(0) - share_data->predicted_size + share_data->octomap_resolution * i;
					double y = share_data->object_center_world(1) - share_data->predicted_size + share_data->octomap_resolution * j;
					double z = max(share_data->min_z_table, share_data->object_center_world(2) - share_data->predicted_size) + share_data->octomap_resolution * k;
					auto node = share_data->octo_model->search(x, y, z);
					if (node == NULL) cout << "what?" << endl;
					fout << node->getOccupancy() << '\n';
				}
	}
	else if (share_data->method_of_IG == 9) { //PCNBV
		share_data->access_directory(share_data->pcnbv_path + "/data");
		ofstream fout_pointcloud(share_data->pcnbv_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_pc" + to_string(iterations) + ".txt");
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGB>);
		//随机降采样点云
		pcl::RandomSample<pcl::PointXYZRGB> ran;
		ran.setInputCloud(share_data->cloud_final);
		ran.setSample(1024); //设置下采样点云的点数
		ran.filter(*cloud_out);
		for (int i = 0; i < cloud_out->points.size(); i++){
			fout_pointcloud << cloud_out->points[i].x << ' '
				<< cloud_out->points[i].y << ' '
				<< cloud_out->points[i].z << '\n';
		}
		ofstream fout_viewstate(share_data->pcnbv_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_vs" + to_string(iterations) + ".txt");
		for (int i = 0; i < now_view_space->views.size(); i++) {
			if (now_view_space->views[i].vis)	fout_viewstate << 1 << '\n';
			else  fout_viewstate << 0 << '\n';
		}
	}
	else if (share_data->method_of_IG == 7) { //SCVP
		if (iterations == 0 + share_data->num_of_nbvs_combined) {
			//octotree
			share_data->access_directory(share_data->sc_net_path + "/data");
			//ofstream fout(share_data->sc_net_path + "/data/" + share_data->name_of_pcd + ".txt");
			ofstream fout(share_data->sc_net_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_voxel.txt");
			for (int i = 0; i < 32; i++)
				for (int j = 0; j < 32; j++)
					for (int k = 0; k < 32; k++)
					{
						double x = share_data->object_center_world(0) - share_data->predicted_size + share_data->octomap_resolution * i;
						double y = share_data->object_center_world(1) - share_data->predicted_size + share_data->octomap_resolution * j;
						double z = max(share_data->min_z_table, share_data->object_center_world(2) - share_data->predicted_size) + share_data->octomap_resolution * k;
						auto node = share_data->octo_model->search(x, y, z);
						if (node == NULL) cout << "what?" << endl;
						fout << node->getOccupancy() << '\n';
					}
			//view state
			ofstream fout_viewstate(share_data->sc_net_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_vs.txt");
			if (share_data->MA_SCVP_on) { //如果是MA-SCVP才输出视点向量
				for (int i = 0; i < now_view_space->views.size(); i++) {
					if (now_view_space->views[i].vis)	fout_viewstate << 1 << '\n';
					else  fout_viewstate << 0 << '\n';
				}
			}
		}
	}
	else { //搜索方法
		int num_of_cover = 1;
		int num_of_voxel = 0;
		//处理views_informaiton
		if (iterations == 0) (*now_views_infromation) = new Views_Information(share_data, nbv_plan->voxel_information, now_view_space, iterations);
		else (*now_views_infromation)->update(share_data, now_view_space, iterations);
		if (share_data->method_of_IG == GMC) {
			//处理GMC，获取全局优化函数
			views_voxels_GMC* max_cover_solver = new views_voxels_GMC(share_data->num_of_max_flow_node, now_view_space, *now_views_infromation, nbv_plan->voxel_information, share_data);
			max_cover_solver->solve();
			vector<pair<int, int>> coverage_view_id_voxelnum_set = max_cover_solver->get_view_id_voxelnum_set();
			num_of_cover = coverage_view_id_voxelnum_set.size();
			for (int i = 0; i < now_view_space->views.size(); i++)
				now_view_space->views[i].in_cover = 0;
			for (int i = 0; i < coverage_view_id_voxelnum_set.size(); i++) {
				now_view_space->views[coverage_view_id_voxelnum_set[i].first].in_cover = coverage_view_id_voxelnum_set[i].second;
				num_of_voxel += coverage_view_id_voxelnum_set[i].second;
			}
			delete max_cover_solver;
			//保证分母不为0，无实际意义
			num_of_voxel = max(num_of_voxel, 1);
		}
		else if (share_data->method_of_IG == MCMF) {
			//处理网络流，获取全局优化函数
			views_voxels_MF* set_cover_solver = new views_voxels_MF(share_data->num_of_max_flow_node, now_view_space, *now_views_infromation, nbv_plan->voxel_information, share_data);
			set_cover_solver->solve();
			vector<int> coverage_view_id_set = set_cover_solver->get_view_id_set();
			for (int i = 0; i < coverage_view_id_set.size(); i++)
				now_view_space->views[coverage_view_id_set[i]].in_coverage[iterations] = 1;
			delete set_cover_solver;
		}
		//综合计算局部贪心与全局优化，产生视点信息熵
		share_data->sum_local_information = 0;
		share_data->sum_global_information = 0;
		share_data->sum_robot_cost = 0;
		for (int i = 0; i < now_view_space->views.size(); i++) {
			share_data->sum_local_information += now_view_space->views[i].information_gain;
			share_data->sum_global_information += now_view_space->views[i].get_global_information();
			share_data->sum_robot_cost += now_view_space->views[i].robot_cost;
		}
		//保证分母不为0，无实际意义
		if (share_data->sum_local_information == 0) share_data->sum_local_information = 1.0;
		if (share_data->sum_global_information == 0) share_data->sum_global_information = 1.0;
		for (int i = 0; i < now_view_space->views.size(); i++) {
			if (share_data->move_cost_on == false) {
				if (share_data->method_of_IG == MCMF) now_view_space->views[i].final_utility = (1 - share_data->cost_weight) * now_view_space->views[i].information_gain / share_data->sum_local_information + share_data->cost_weight * now_view_space->views[i].get_global_information() / share_data->sum_global_information;
				else if (share_data->method_of_IG == Kr) now_view_space->views[i].final_utility = now_view_space->views[i].information_gain / now_view_space->views[i].voxel_num;
				else if (share_data->method_of_IG == GMC) now_view_space->views[i].final_utility = (1 - share_data->cost_weight) * now_view_space->views[i].information_gain / share_data->sum_local_information + share_data->cost_weight * now_view_space->views[i].in_cover / num_of_voxel;
				else now_view_space->views[i].final_utility = now_view_space->views[i].information_gain;
			}
			else {
				if (share_data->method_of_IG == MCMF) now_view_space->views[i].final_utility = (1 - share_data->move_weight)* ((1 - share_data->cost_weight) * now_view_space->views[i].information_gain / share_data->sum_local_information + share_data->cost_weight * now_view_space->views[i].get_global_information() / share_data->sum_global_information) + share_data->move_weight * (share_data->robot_cost_negtive == true ? -1 : 1) * now_view_space->views[i].robot_cost / share_data->sum_robot_cost;
				else if (share_data->method_of_IG == Kr) now_view_space->views[i].final_utility = (1 - share_data->move_weight) * now_view_space->views[i].information_gain / now_view_space->views[i].voxel_num + share_data->move_weight * (share_data->robot_cost_negtive == true ? -1 : 1) * now_view_space->views[i].robot_cost / share_data->sum_robot_cost;
				else if (share_data->method_of_IG == GMC) now_view_space->views[i].final_utility = (1 - share_data->move_weight) * (1 - share_data->cost_weight) * now_view_space->views[i].information_gain / share_data->sum_local_information + share_data->cost_weight * now_view_space->views[i].in_cover / num_of_voxel + share_data->move_weight * (share_data->robot_cost_negtive == true ? -1 : 1) * now_view_space->views[i].robot_cost / share_data->sum_robot_cost;
				else now_view_space->views[i].final_utility = (1 - share_data->move_weight) * now_view_space->views[i].information_gain + share_data->move_weight * (share_data->robot_cost_negtive == true ? -1 : 1) * now_view_space->views[i].robot_cost / share_data->sum_robot_cost;
			}
		}
	}
	//更新标志位
	share_data->now_views_infromation_processed = true;
}

void move_robot(View* now_best_view, View_Space* now_view_space, Share_Data* share_data) {
	if (nbv_plan->iterations + 1 == share_data->num_of_nbvs_combined) { //Combined+MASCVP切换
		share_data->method_of_IG = SCVP;
		sort(now_view_space->views.begin(), now_view_space->views.end(), view_id_compare);
	}
	if (share_data->num_of_max_iteration > 0 && nbv_plan->iterations + 1 >= share_data->num_of_max_iteration) share_data->over = true;
	if (!share_data->move_wait) share_data->move_on = true;
}

void show_cloud(pcl::visualization::PCLVisualizer::Ptr viewer) {
	//pcl显示点云
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void get_command()
{	//从控制台读取指令字符串
	string cmd;
	while (!stop && !share_data->over)
	{
		cout << "Input command 1.stop 2.over 3.next_itreation :" << endl;
		cin >> cmd;
		if (cmd == "1") stop = true;
		else if (cmd == "2") share_data->over = true;
		else if (cmd == "3") share_data->move_on = true;
		else cout << "Wrong command.Retry :" << endl;
	}
	cout << "get_command over." << endl;
}

void get_run()
{
	//NBV规划期初始化
	nbv_plan = new NBV_Planner(share_data);
	//主控循环
	string status="";
	//实时读取与规划
	while (!stop && nbv_plan->plan()) {
		//如果状态有变化就输出
		if (status != nbv_plan->out_status()) {
			status = nbv_plan->out_status();
			cout << "NBV_Planner's status is " << status << endl;
		}
	}
	delete nbv_plan;
}

#define DebugOne 0
#define DebugAll 1
#define TestAll 2
#define GetGTPoints 3

int mode = DebugAll;

int main()
{
	//Init
	ios::sync_with_stdio(false);
	cout << "input mode:";
	cin >> mode;
	vector<int> rotate_states;
	//rotate_states.push_back(0);
	//rotate_states.push_back(1);
	//rotate_states.push_back(2);
	rotate_states.push_back(3);
	//rotate_states.push_back(4);
	//rotate_states.push_back(5);
	rotate_states.push_back(6);
	//rotate_states.push_back(7);
	vector<int> first_view_ids;
	first_view_ids.push_back(0);
	first_view_ids.push_back(2);
	first_view_ids.push_back(4);
	first_view_ids.push_back(14);
	first_view_ids.push_back(27);
	//for (int i = 0; i < 32; i++)
	//	first_view_ids.push_back(i);
	int combined_test_on;
	cout << "combined on:";
	cin >> combined_test_on;
	//scvp，7需要先跑，其他方法读取个数，若后跑就按默认值
	vector<int> methods;
	methods.push_back(7);
	methods.push_back(3);
	methods.push_back(4);
	methods.push_back(0);
	methods.push_back(6);
	int method_id;
	cout << "thread for method id:";
	cin >> method_id;
	int move_test_on;
	cout << "move test on :";
	cin >> move_test_on;
	//测试集
	vector<string> names;
	cout << "input models:" << endl;
	string name;
	while (cin >> name) {
		if (name == "-1") break;
		names.push_back(name);
	}
	//选取模式
	if (mode == DebugOne)
	{
		//数据区初始化
		share_data = new Share_Data("../DefaultConfiguration.yaml");
		//控制台读取指令线程
		thread cmd(get_command);
		//NBV系统运行线程
		thread runner(get_run);
		//等待线程结束
		runner.join();
		cmd.join();
		delete share_data;
	}
	else if (mode == DebugAll){
		//测试0视点,自选方法
		for (int i = 0; i < names.size(); i++) {
			//数据区初始化
			share_data = new Share_Data("../DefaultConfiguration.yaml", names[i]);
			//NBV系统运行线程
			thread runner(get_run);
			//等待线程结束
			runner.join();
			delete share_data;
		}
	}
	else if (mode == TestAll) {
		//测试所有物体、视点、方法
		for (int i = 0; i < names.size(); i++) {
			for (int j = 0; j < rotate_states.size(); j++) {
				for (int k = 0; k < first_view_ids.size(); k++) {
					//数据区初始化
					share_data = new Share_Data("../DefaultConfiguration.yaml", names[i], rotate_states[j], first_view_ids[k], method_id, move_test_on, combined_test_on);
					//NBV系统运行线程
					thread runner(get_run);
					//等待线程结束
					runner.join();
					delete share_data;
				}
			}
		}
	}
	else if (mode == GetGTPoints) {
		for (int i = 0; i < names.size(); i++) {
			cout << "Get GT visible pointcloud number of model " << names[i] << endl;
			
			/*
			// 获取物体尺寸
			share_data = new Share_Data("../DefaultConfiguration.yaml", names[i], 0);
			nbv_plan = new NBV_Planner(share_data);
			ofstream write;
			write.open("D:\\论文\\MA-SCVP\\size.txt", ios::app);
			write << share_data->predicted_size << endl;
			write.close();
			*/

			//for (int rotate_state = 0; rotate_state < 8; rotate_state++) {
			for (int x = 0; x < rotate_states.size(); x++) {
				int rotate_state = rotate_states[x];
				//数据区初始化
				share_data = new Share_Data("../DefaultConfiguration.yaml", names[i], rotate_state);
				//NBV规划期初始化
				nbv_plan = new NBV_Planner(share_data);

				/*
				// 获取视点空间重建率
				ifstream read("D:\\Data\\MA-SCVP\\GT_points_num\\" + names[i] + "_r" + to_string(rotate_state) + ".txt");
				int SC16, SC32, SC64;
				read >> SC32;
				read.close();
				read.open("D:\\Data\\MA-SCVP\\GT_points_num\\\\GT_points_num_64\\" + names[i] + "_r" + to_string(rotate_state) + ".txt");
				read >> SC64;
				read.close();
				read.open("D:\\Data\\MA-SCVP\\GT_points_num\\\\GT_points_num_16\\" + names[i] + "_r" + to_string(rotate_state) + ".txt");
				read >> SC16;
				ofstream write;
				write.open("D:\\论文\\MA-SCVP\\SC_viewspace.txt", ios::app);
				write << names[i] + "_r" + to_string(rotate_state) << "\t" << SC16 << "\t" << SC32 << "\t" << SC64 << "\t" << share_data->cloud_points_number << endl;
				write.close();
				*/
				
				//获取全部点云
				for (int i = 0; i < nbv_plan->now_view_space->views.size(); i++) {
					nbv_plan->percept->precept(&nbv_plan->now_view_space->views[i]);
				}
				//根据Octomap精度统计可见点个数
				double now_time = clock();
				int num = 0;
				unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
				for (int j = 0; j < share_data->cloud_final->points.size(); j++) {
					octomap::OcTreeKey key = share_data->ground_truth_model->coordToKey(share_data->cloud_final->points[j].x, share_data->cloud_final->points[j].y, share_data->cloud_final->points[j].z);
					if (voxel->find(key) == voxel->end()) {
						(*voxel)[key] = num++;
					}
				}
				cout << "GT visible pointcloud get with executed time " << clock() - now_time << " ms." << endl;
				/*
				for (octomap::ColorOcTree::leaf_iterator it = nbv_plan->percept->ground_truth_model->begin_leafs(), end = nbv_plan->percept->ground_truth_model->end_leafs(); it != end; ++it) {
					if (voxel->find(it.getKey()) != voxel->end()) {
						nbv_plan->percept->ground_truth_model->setNodeColor(it.getKey(), 255, 0, 0);
					}
				}
				nbv_plan->percept->ground_truth_model->write("D:\\论文\\MA-SCVP\\inner.ot");
				*/
				
				//保存GT
				share_data->access_directory(share_data->gt_path + "/GT_points_num/");
				ofstream fout_GT_points_num(share_data->gt_path + "/GT_points_num/" + names[i] + "_r" + to_string(rotate_state) + ".txt");
				fout_GT_points_num << voxel->size() << endl;
				cout << "Rotate " << rotate_state << " GT_points_num is " << voxel->size() << " ,rate is " << 1.0 * voxel->size() / share_data->cloud_points_number << endl;
				delete voxel;
				delete nbv_plan;
				delete share_data;
				
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
Horse
Lion
LM1
LM2
LM3
LM4
LM5
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
obj_000032

LM6
obj_000010
obj_000011
*/
