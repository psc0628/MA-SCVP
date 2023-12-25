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
	pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp_rgb;
	pcl::PassThrough<pcl::PointXYZRGB> pass;
	pcl::VoxelGrid<pcl::PointXYZRGB> sor;
	bool precepted;

	Perception_3D(Share_Data* _share_data) {
		share_data = _share_data;
        icp_rgb.setTransformationEpsilon(1e-10);
        icp_rgb.setEuclideanFitnessEpsilon(1e-6);
        icp_rgb.setMaximumIterations(100000);
		icp_rgb.setMaxCorrespondenceDistance(share_data->icp_distance);		// 设置ICP匹配时两点之间的最大距离 5cm
		pass.setFilterLimitsNegative(false);   			// 设置字段范围内的是保留（false）还是过滤掉（true）
        sor.setLeafSize(0.005f, 0.005f, 0.005f);
		precepted = false;
	}

	~Perception_3D() {
		;
	}

	bool is_precept() {
		return precepted;
	}

	void precept(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc) {
		// read point cloud
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
		*cloud = *pc;

		if (share_data->show) { //显示成像点云
			pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Camera" + to_string(share_data->vaild_clouds)));
			viewer1->setBackgroundColor(255, 255, 255);
			viewer1->addCoordinateSystem(0.1);
			viewer1->initCameraParameters();
			viewer1->setCameraPosition(0.8, 0.8, 0.8, share_data->object_center_world(0), share_data->object_center_world(1), share_data->object_center_world(2), 0, 0, 1);
			viewer1->addPointCloud<pcl::PointXYZRGB>(cloud, "cloud");

			while (!viewer1->wasStopped())
			{
				viewer1->spinOnce(100);
				boost::this_thread::sleep(boost::posix_time::microseconds(100000));
			}
		}

		// keep cloud in the tablesapce
		pass.setInputCloud(cloud);          
        pass.setFilterFieldName("x");         
        pass.setFilterLimits(share_data->table_x_min, share_data->table_x_max);
        pass.filter(*cloud);
        pass.setInputCloud(cloud);         
        pass.setFilterFieldName("y");         
		pass.setFilterLimits(share_data->table_y_min, share_data->table_y_max);    
        pass.filter(*cloud);
        pass.setInputCloud(cloud);              
        pass.setFilterFieldName("z");          
        pass.setFilterLimits(share_data->min_z_table - 0.01, share_data->height_to_filter_arm);      
        pass.filter(*cloud);

		if(share_data->cloud_scene->points.size() != 0) {
			cout<<"Starting ICP"<<endl;
			// do voxel grid filter
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_downsample(new pcl::PointCloud<pcl::PointXYZRGB>);
			*cloud_downsample = *cloud;

			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_secne_downsample(new pcl::PointCloud<pcl::PointXYZRGB>);
			*cloud_secne_downsample = *share_data->cloud_scene;

			sor.setInputCloud(cloud_downsample);
			sor.filter(*cloud_downsample);

			sor.setInputCloud(cloud_secne_downsample);
			sor.filter(*cloud_secne_downsample);

			// do icp
			icp_rgb.setInputSource(cloud_downsample);
			icp_rgb.setInputTarget(cloud_secne_downsample);
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_icp(new pcl::PointCloud<pcl::PointXYZRGB>);
			icp_rgb.align(*cloud_icp);
			Eigen::Matrix4d T_icp(4, 4);
			T_icp = icp_rgb.getFinalTransformation().cast<double>();
			cout << "ICP matraix: " << T_icp << endl;
			pcl::transformPointCloud(*cloud, *cloud, T_icp);
		}

		// update clouds
		share_data->vaild_clouds++;
		share_data->clouds.push_back(cloud);
		*share_data->cloud_scene += *cloud;
		
		// pass through 2 times BBX size to get the final cloud, assume the object size will not be larger than 2 times init BBX size
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr no_table(new pcl::PointCloud<pcl::PointXYZRGB>);
		*no_table = *cloud;
		pass.setInputCloud(no_table);          
        pass.setFilterFieldName("x");         
        pass.setFilterLimits(share_data->map_center(0) - 2 * share_data->map_size, share_data->map_center(0) + 2 * share_data->map_size);       
        pass.filter(*no_table);
        pass.setInputCloud(no_table);         
        pass.setFilterFieldName("y");         
        pass.setFilterLimits(share_data->map_center(1) - 2 * share_data->map_size, share_data->map_center(1) + 2 * share_data->map_size);          
        pass.filter(*no_table);
        pass.setInputCloud(no_table);             
        pass.setFilterFieldName("z");   
		pass.setFilterLimits(max(share_data->height_of_ground, share_data->map_center(2) - 2 * share_data->map_size), share_data->map_center(2) + 2 * share_data->map_size); 
        pass.filter(*no_table);
		*share_data->cloud_final += *no_table;

		// update flag
		precepted = true;

		if (share_data->show) { //显示成像点云
			pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Camera" + to_string(share_data->vaild_clouds)));
			viewer1->setBackgroundColor(255, 255, 255);
			viewer1->addCoordinateSystem(0.1);
			viewer1->initCameraParameters();
			viewer1->setCameraPosition(0.8, 0.8, 0.8, share_data->object_center_world(0), share_data->object_center_world(1), share_data->object_center_world(2), 0, 0, 1);
			viewer1->addPointCloud<pcl::PointXYZRGB>(cloud, "cloud");

			while (!viewer1->wasStopped())
			{
				viewer1->spinOnce(100);
				boost::this_thread::sleep(boost::posix_time::microseconds(100000));
			}
		}


	}
};

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
		//�����±�ӳ��
		view_id_in = new map<int, int>();
		view_id_out = new map<int, int>();
		(*view_id_in)[now_view->id] = 0;
		(*view_id_out)[0] = now_view->id;
		for (int i = 0; i < view_set_label.size(); i++) {
			(*view_id_in)[view_set_label[i]] = i+1;
			(*view_id_out)[i+1] = view_set_label[i];
		}
		//�ڵ����뷽����
		n = view_set_label.size() + 1;
		m = 1LL << n;
		//local path ��ȫ����ͼ
		graph.resize(n);
		for (int i = 0; i < n; i++)
			graph[i].resize(n);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++) {
				//����id
				int u = (*view_id_out)[i];
				int v = (*view_id_out)[j];
				//������·��
				pair<int, double> local_path = get_local_path(view_space->views[u].init_pos.eval(), view_space->views[v].init_pos.eval(), view_space->object_center_world.eval(), view_space->predicted_size + share_data->safe_distance); //��Χ�а뾶�ǰ�߳��ĸ���2��
				if (local_path.first < 0) cout << "local path wrong." << endl;
				graph[i][j] = local_path.second;
			}
		//��ʼ��dp
		dp.resize(m);
		for (int i = 0; i < m; i++)
			dp[i].resize(n);
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				dp[i][j] = 1e10;
		dp[1][0] = 0;
		//��ʼ��·����¼
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
		for (int i = 0; i < m; i++)  // i��������һ�������ļ��ϣ�����ÿ��λ�õ�0/1����û��/�о��������(m=1<<n)
		{
			for (int j = 0; j < n; j++)  //ö�ٵ�ǰ�����ĸ���
			{
				if ((i >> j) & 1)  //���i�����j
				{
					for (int k = 0; k < n; k++)  //ö�ٵ���j�ĵ�
					{
						if (i - (1 << j) >> k & 1)  //ȥ������j�ļ���i
						{
							//�����������²���¼ת��·��
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
		//Ĭ�����Ϊ0,����ÿ�������յ��ҵ���̹��ܶ���·����ȫ���Ƿ�����
		int end_node;
		total_shortest = 1e20;
		for (int i = 1; i < n; i++) {
			if (total_shortest > dp[m - 1][i]) {
				total_shortest = dp[m - 1][i];
				end_node = i;
			}
		}
		//���·��
		for (int i = (1 << n) - 1, j = end_node; i > 0;) {
			//ע���±꽻��
			global_path.push_back((*view_id_out)[j]);
			int ii = i - (1 << j);
			int jj = path[i][j];
			i = ii, j = jj;
		}
		//����·���跴��
		reverse(global_path.begin(), global_path.end());
		solved = true;
		double cost_time = clock() - now_time;
		cout << "Global Path length " << total_shortest << " getted with executed time " << cost_time << " ms." << endl;
		//����
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
		//ɾ��������
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
		//ֻ������������information
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

		//filter init cloud
		pcl::PassThrough<pcl::PointXYZRGB> pass;
		pass.setFilterLimitsNegative(false); 
		pass.setInputCloud(share_data->cloud_now);          
        pass.setFilterFieldName("x");         
        pass.setFilterLimits(share_data->table_x_min, share_data->table_x_max);   
        pass.filter(*share_data->cloud_now);
        pass.setInputCloud(share_data->cloud_now);         
        pass.setFilterFieldName("y");         
		pass.setFilterLimits(share_data->table_y_min, share_data->table_y_max);     
        pass.filter(*share_data->cloud_now);
        pass.setInputCloud(share_data->cloud_now);              
        pass.setFilterFieldName("z");          
        pass.setFilterLimits(share_data->height_of_ground, share_data->height_to_filter_arm);      
        pass.filter(*share_data->cloud_now);
	
		// get viewspace
		int first_view_id = share_data->first_view_id;
		now_view_space = new View_Space(iterations, share_data, voxel_information, share_data->cloud_now, first_view_id);
		//���ó�ʼ�ӵ�Ϊͳһ��λ��
		now_view_space->views[first_view_id].vis++;
		now_best_view = new View(now_view_space->views[first_view_id]);

		//�˶����ۣ��ӵ�id����ǰ���ۣ��������
		share_data->movement_cost = 0;
		//share_data->access_directory(share_data->save_path + "/movement");
		//ofstream fout_move(share_data->save_path + "/movement/path" + to_string(-1) + ".txt");
		//fout_move << 0 << '\t' << 0.0 << '\t' << 0.0 << endl;
		now_best_view->get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
		Eigen::Matrix4d view_pose_world = (share_data->now_camera_pose_world * now_best_view->pose.inverse()).eval();
		//������ʼ��
		percept = new Perception_3D(share_data);

		if (share_data->show) { //show cloud now
			pcl::visualization::PCLVisualizer::Ptr viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("initial"));
			viewer->setBackgroundColor(255, 255, 255);
			viewer->addCoordinateSystem(0.1);
			viewer->initCameraParameters();
			pcl::visualization::Camera cam;
			viewer->getCameraParameters(cam);
			cam.window_size[0] = 1920;
			cam.window_size[1] = 1080;
			viewer->setCameraParameters(cam);
			viewer->setCameraPosition(-0.2, 0.8, 1.0, share_data->object_center_world(0), share_data->object_center_world(1), share_data->object_center_world(2), 0, 0, 1);
			viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_now, "cloud_now)");
			while (!viewer->wasStopped())
			{
				viewer->spinOnce(100);
				boost::this_thread::sleep(boost::posix_time::microseconds(100000));
			}
		}

		if (share_data->show) { //show gt cloud
			pcl::visualization::PCLVisualizer::Ptr viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("GT"));
			viewer->setBackgroundColor(255, 255, 255);
			viewer->addCoordinateSystem(0.1);
			viewer->initCameraParameters();
			viewer->setCameraPosition(0.8, 0.8, 0.8, share_data->object_center_world(0), share_data->object_center_world(1), share_data->object_center_world(2), 0, 0, 1);
			viewer->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_ground_truth, "cloud_ground_truth)");
			while (!viewer->wasStopped())
			{
				viewer->spinOnce(100);
				boost::this_thread::sleep(boost::posix_time::microseconds(100000));
			}
		}

		//filter gt cloud
		pcl::VoxelGrid<pcl::PointXYZRGB> sor;
		sor.setLeafSize(share_data->ground_truth_resolution, share_data->ground_truth_resolution, share_data->ground_truth_resolution);
		sor.setInputCloud(share_data->cloud_ground_truth);
		sor.filter(*share_data->cloud_ground_truth);
		cout << "filerted cloud_ground_truth->points.size() is " << share_data->cloud_ground_truth->points.size() << endl;

		share_data->cloud_ground_truth_downsampled = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
		sor.setLeafSize(0.005f, 0.005f, 0.005f);
		sor.setInputCloud(share_data->cloud_ground_truth);
		sor.filter(*share_data->cloud_ground_truth_downsampled);
		cout << "filerted cloud_ground_truth_downsampled->points.size() is " << share_data->cloud_ground_truth_downsampled->points.size() << endl;

		auto ptr = share_data->cloud_ground_truth->points.begin();
		for (int i = 0; i < share_data->cloud_ground_truth->points.size(); i++,ptr++) {
			//GT�������
			octomap::OcTreeKey key;  bool key_have = share_data->ground_truth_model->coordToKeyChecked(octomap::point3d((*ptr).x, (*ptr).y, (*ptr).z), key);
			if (key_have) {
				octomap::ColorOcTreeNode* voxel = share_data->ground_truth_model->search(key);
				if (voxel == NULL) {
					share_data->ground_truth_model->setNodeValue(key, share_data->ground_truth_model->getProbHitLog(), true);
					share_data->ground_truth_model->integrateNodeColor(key, (*ptr).r, (*ptr).g, (*ptr).b);
				}
				if(share_data->voxel_gt->find(key) == share_data->voxel_gt->end()) {
					(*share_data->voxel_gt)[key] = 1;
				}
			}
			//GT_sample�������
			octomap::OcTreeKey key_sp;  bool key_have_sp = share_data->GT_sample->coordToKeyChecked(octomap::point3d((*ptr).x, (*ptr).y, (*ptr).z), key_sp);
			if (key_have_sp) {
				octomap::ColorOcTreeNode* voxel_sp = share_data->GT_sample->search(key_sp);
				if (voxel_sp == NULL) {
					share_data->GT_sample->setNodeValue(key_sp, share_data->GT_sample->getProbHitLog(), true);
					share_data->GT_sample->integrateNodeColor(key_sp, (*ptr).r, (*ptr).g, (*ptr).b);
				}
				if(share_data->voxel_gt_sample->find(key_sp) == share_data->voxel_gt_sample->end()) {
					(*share_data->voxel_gt_sample)[key_sp] = 1;
				}
			}
			ptr++;
		}

		share_data->access_directory(share_data->save_path);
		share_data->ground_truth_model->updateInnerOccupancy();
		//share_data->ground_truth_model->write(share_data->save_path + "/GT.ot");
		//GT_sample_voxels
		share_data->GT_sample->updateInnerOccupancy();
		//share_data->GT_sample->write(share_data->save_path + "/GT_sample.ot");
		share_data->init_voxels = 0;
		int full_voxels = 0;
		//��sample��ͳ���ܸ���
		for (auto it = share_data->voxel_gt_sample->begin(); it != share_data->voxel_gt_sample->end(); it++) {
			if(share_data->GT_sample->keyToCoord(it->first).z() > share_data->min_z_table + share_data->octomap_resolution)
				share_data->init_voxels++;
			full_voxels++;
		}
		cout << "Map_GT_sample has voxels " << share_data->init_voxels << endl;
		cout << "Map_GT_sample has voxels with bottom " << full_voxels << endl;
		share_data->init_voxels = full_voxels;
		//ofstream fout_sample(share_data->save_path + "/GT_sample_voxels.txt");
		//fout_sample << share_data->init_voxels << endl;
		//��GT��ͳ���ܸ���
		share_data->cloud_points_number = share_data->voxel_gt->size();
		cout << "Map_GT has voxels " << share_data->cloud_points_number << endl;
		//ofstream fout_gt(share_data->save_path + "/GT_voxels.txt");
		//fout_gt << share_data->cloud_points_number << endl;

		if (share_data->show) { //��ʾBBX�����λ�á�GT
			pcl::visualization::PCLVisualizer::Ptr viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("GT"));
			viewer->setBackgroundColor(255, 255, 255);
			viewer->addCoordinateSystem(0.1);
			viewer->initCameraParameters();
			viewer->setCameraPosition(0.8, 0.8, 0.8, share_data->object_center_world(0), share_data->object_center_world(1), share_data->object_center_world(2), 0, 0, 1);
			//��һ֡���λ��
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
				//��һ����ʾ���е�Ϊ��ɫ
				(*pt).r = 255, (*pt).g = 0, (*pt).b = 0;
			}
			viewer->addPointCloud<pcl::PointXYZRGB>(test_viewspace, "test_viewspace");
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "test_viewspace");
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
			if (percept->is_precept()) {
				percept->precepted = false;
				thread next_view_space(create_view_space, &now_view_space, now_best_view, share_data, iterations);
				next_view_space.join();
				status = WaitViewSpace;
			}
			break;
		case WaitViewSpace:
			if (share_data->now_view_space_processed) {
				thread next_views_information(create_views_information, &now_views_infromation, now_best_view, now_view_space, share_data, iterations);
				next_views_information.join();
				status = WaitInformation;
			}
			break;
		case WaitInformation:
			if (share_data->now_views_infromation_processed) {
				if (share_data->method_of_IG == 8) { //Random
					srand(clock());
					int next_id = rand() % share_data->num_of_views; //32������
					while (now_view_space->views[next_id].vis) { //�����һ��û�з��ʹ���
						next_id = rand() % share_data->num_of_views;
					}
					now_view_space->views[next_id].vis++;
					now_view_space->views[next_id].can_move = true;
					delete now_best_view;
					now_best_view = new View(now_view_space->views[next_id]);
					now_best_view->get_next_camera_pos(share_data->now_camera_pose_world,share_data->object_center_world);
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
					now_best_view->get_next_camera_pos(share_data->now_camera_pose_world,share_data->object_center_world);
					//�˶����ۣ��ӵ�id����ǰ���ۣ��������
					share_data->movement_cost += now_best_view->robot_cost;
					share_data->access_directory(share_data->save_path + "/movement");
					ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
					fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
					//���±�־�ļ�
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
					now_best_view->get_next_camera_pos(share_data->now_camera_pose_world,share_data->object_center_world);
					//�˶����ۣ��ӵ�id����ǰ���ۣ��������
					share_data->movement_cost += now_best_view->robot_cost;
					share_data->access_directory(share_data->save_path + "/movement");
					ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
					fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
					//���±�־�ļ�
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
						//����,��ɾ���ѷ����ӵ�
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
						//����һ�������ӵ����
						ofstream fout_all_needed_views(share_data->save_path + "/all_needed_views.txt");
						fout_all_needed_views << view_set_label.size() + 1 + share_data->num_of_nbvs_combined << endl;
						cout<< "All_needed_views is "<< view_set_label.size() + 1 + share_data->num_of_nbvs_combined << endl;
						//���û���ӵ���Ҫ�����ֱ���˳�
						if (view_set_label.size() == 0) {
							//���±�־�ļ�
							this_thread::sleep_for(chrono::seconds(1));
							int removed = remove((share_data->sc_net_path + "/log/ready.txt").c_str());
							if (removed != 0) cout << "cannot remove ready.txt." << endl;
							//ϵͳ�˳�
							share_data->over = true;
							status = WaitMoving;
							break;
						}
						//�滮·��
						Global_Path_Planner* gloabl_path_planner = new Global_Path_Planner(share_data, now_view_space, now_best_view, view_set_label);
						gloabl_path_planner->solve();
						//����·��
						share_data->view_label_id = gloabl_path_planner->get_path_id_set();
						delete now_best_view;
						now_best_view = new View(now_view_space->views[share_data->view_label_id[iterations - share_data->num_of_nbvs_combined]]);
						now_best_view->get_next_camera_pos(share_data->now_camera_pose_world,share_data->object_center_world);
						//�˶����ۣ��ӵ�id����ǰ���ۣ��������
						share_data->movement_cost += now_best_view->robot_cost;
						share_data->access_directory(share_data->save_path + "/movement");
						ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
						fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
						//���±�־�ļ�
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
						now_best_view->get_next_camera_pos(share_data->now_camera_pose_world,share_data->object_center_world);
						//�˶����ۣ��ӵ�id����ǰ���ۣ��������
						share_data->movement_cost += now_best_view->robot_cost;
						share_data->access_directory(share_data->save_path + "/movement");
						ofstream fout_move(share_data->save_path + "/movement/path" + to_string(iterations) + ".txt");
						fout_move << now_best_view->id << '\t' << now_best_view->robot_cost << '\t' << share_data->movement_cost << endl;
					}
				}
				else{//�����㷨
					//���ӵ�����
					sort(now_view_space->views.begin(), now_view_space->views.end(), view_utility_compare);
					/*if (share_data->sum_local_information == 0) {
						cout << "randomly choose a view" << endl;
						srand(clock());
						random_shuffle(now_view_space->views.begin(), now_view_space->views.end());
					}*/
					//informed_viewspace
					if (share_data->show) { //��ʾBBX�����λ��
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
							//���ʹ��ĵ��¼Ϊ��ɫ
							if (now_view_space->views[i].vis) (*ptr).r = 0, (*ptr).g = 0, (*ptr).b = 255;
							//���������ڵ�����Ϊ��ɫ
							else if (now_view_space->views[i].in_coverage[iterations] && i < now_view_space->views.size() / 10) (*ptr).r = 255, (*ptr).g = 255, (*ptr).b = 0;
							//���������ڵ�����Ϊ��ɫ
							else if (now_view_space->views[i].in_coverage[iterations]) (*ptr).r = 255, (*ptr).g = 0, (*ptr).b = 0;
							//ǰ10%��Ȩ�صĵ�����Ϊ����ɫ
							else if (i < now_view_space->views.size() / 10) (*ptr).r = 0, (*ptr).g = 255, (*ptr).b = 255;
							//������ɫ
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
						now_best_view->get_next_camera_pos(share_data->now_camera_pose_world,share_data->object_center_world);
						max_utility = now_best_view->final_utility;
						now_view_space->views[i].vis++;
						now_view_space->views[i].can_move = true;
						cout << "choose the " << i << "th view." << endl;
						//�˶����ۣ��ӵ�id����ǰ���ۣ��������
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
				//�����˶�ģ��
				thread next_moving(move_robot, now_best_view, now_view_space, share_data);
				next_moving.join();
				status = WaitMoving;
			}
			break;
		case WaitMoving:
			//if the method is not (combined) one-shot and random, then use f_voxel to decide whether to stop
			if(!(share_data->Combined_on == true || share_data->method_of_IG == 7 || share_data->method_of_IG == 8) && share_data->f_voxels.size() != iterations + 1){
				//compute f_voxels
				int f_voxels_num = 0;
				for (octomap::ColorOcTree::leaf_iterator it = share_data->octo_model->begin_leafs(), end = share_data->octo_model->end_leafs(); it != end; ++it) {
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
				cout << "share_data->f_voxels.size() is " << share_data->f_voxels.size() << endl;

				//save f_voxels_num for each iteration because the old f_voxels_nums may change according to the new octomap
				share_data->access_directory(share_data->save_path + "/f_voxels");
				ofstream fout_f_voxels_num(share_data->save_path + "/f_voxels/f_num" + to_string(iterations) + ".txt");
				for(int i = 0; i < share_data->f_voxels.size(); i++){
					fout_f_voxels_num << share_data->f_voxels[i] << endl;
				}

			    // check if the f_voxels_num is stable
				if (share_data->f_stop_iter == -1) {
					if (share_data->f_voxels.size() > 2) {
						bool f_voxels_change = false;
						//三次扫描过程中，连续两个f变化都小于阈值就结束
						if (fabs(share_data->f_voxels[share_data->f_voxels.size() - 1] - share_data->f_voxels[share_data->f_voxels.size() - 2]) >= share_data->voxels_in_BBX * share_data->f_stop_threshold) {
							f_voxels_change = true;
						}
						if (fabs(share_data->f_voxels[share_data->f_voxels.size() - 2] - share_data->f_voxels[share_data->f_voxels.size() - 3]) >= share_data->voxels_in_BBX * share_data->f_stop_threshold) {
							f_voxels_change = true;
						}
						if (!f_voxels_change) {
							cout << "two f_voxels change smaller than threshold. Record." << endl;
							share_data->f_stop_iter = iterations;

							ofstream fout_f_stop_views(share_data->save_path + "/f_voxels/f_stop_views.txt");
							fout_f_stop_views << 1 << "\t" << share_data->f_stop_iter + 1 << endl; //1 means f_voxels stop
						}
					}
					if (share_data->over == true && share_data->f_stop_iter == -1) {
						cout << "Max iter reached. Record." << endl;
						share_data->f_stop_iter = iterations;

						ofstream fout_f_stop_views(share_data->save_path + "/f_voxels/f_stop_views.txt");
						fout_f_stop_views << 0 << "\t" << share_data->f_stop_iter + 1 << endl; //0 means over
					}
				}
				if (share_data->f_stop_iter_lenient == -1) {
					if (share_data->f_voxels.size() > 2) {
						bool f_voxels_change = false;
						//三次扫描过程中，连续两个f变化都小于阈值就结束
						if (fabs(share_data->f_voxels[share_data->f_voxels.size() - 1] - share_data->f_voxels[share_data->f_voxels.size() - 2]) >= share_data->voxels_in_BBX * share_data->f_stop_threshold_lenient) {
							f_voxels_change = true;
						}
						if (fabs(share_data->f_voxels[share_data->f_voxels.size() - 2] - share_data->f_voxels[share_data->f_voxels.size() - 3]) >= share_data->voxels_in_BBX * share_data->f_stop_threshold_lenient) {
							f_voxels_change = true;
						}
						if (!f_voxels_change) {
							cout << "two f_voxels change smaller than threshold_lenient. Record." << endl;
							share_data->f_stop_iter_lenient = iterations;

							ofstream fout_f_stop_views(share_data->save_path + "/f_voxels/f_lenient_stop_views.txt");
							fout_f_stop_views << 1 << "\t" << share_data->f_stop_iter_lenient + 1 << endl; //1 means f_voxels stop
						}
					}
					if (share_data->over == true && share_data->f_stop_iter_lenient == -1) {
						cout << "Max iter reached. Record." << endl;
						share_data->f_stop_iter_lenient = iterations;

						ofstream fout_f_stop_views(share_data->save_path + "/f_voxels/f_lenient_stop_views.txt");
						fout_f_stop_views << 0 << "\t" << share_data->f_stop_iter_lenient + 1 << endl; //0 means over
					}
				}
				if(iterations + 1 == share_data->mascvp_nbv_needed_views){
					cout << "mascvp_nbv_needed_views reached. Record." << endl;

					ofstream fout_mascvp_stop_views(share_data->save_path + "/f_voxels/mascvp_nbv_needed_views.txt");
					fout_mascvp_stop_views << 1 << "\t" << iterations + 1 << endl; //1 means f_voxels stop
				}
			}
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

bool stop = false;		//���Ƴ������
Share_Data* share_data;			//����������ָ��
NBV_Planner* nbv_plan;

void save_cloud_mid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, string name) {
	//�����м�ĵ��Ƶ��̣߳�Ŀǰ������Ƿ񱣴����
	//share_data->save_cloud_to_disk(cloud, "/clouds", name);
	
}

void create_view_space(View_Space** now_view_space, View* now_best_view, Share_Data* share_data, int iterations) {
	//更新当前机器人位置
	share_data->now_camera_pose_world = (share_data->now_camera_pose_world * now_best_view->pose.inverse()).eval();
	//保存当前机器人位置
	share_data->nbvs_pose_world.push_back(share_data->now_camera_pose_world);
	share_data->access_directory(share_data->save_path + "/nbvs_pose_world");
	ofstream fout_nbvs_pose_world(share_data->save_path + "/nbvs_pose_world/" + to_string(iterations) + ".txt");
	fout_nbvs_pose_world << share_data->now_camera_pose_world <<endl;
	//处理viewspace,如果不需要评估并且是one-shot路径就不更新OctoMap
	(*now_view_space)->update(iterations, share_data, share_data->cloud_final, share_data->clouds[iterations]);
	//保存当前viewspace
	if(share_data->is_save)	{
		share_data->access_directory(share_data->save_path + "/clouds");
		pcl::io::savePCDFileBinary<pcl::PointXYZRGB>(share_data->save_path + "/clouds/" + "object" + to_string(iterations) + ".pcd", *share_data->cloud_final);
		// pcl::io::savePCDFileBinary<pcl::PointXYZRGB>(share_data->save_path + "/clouds/" + "scene" + to_string(iterations) + ".pcd", *share_data->cloud_scene);
		cout << "pointcloud" + to_string(iterations) << " saved" << endl;
	}
	//更新标志
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
					double x = share_data->map_center(0) - share_data->map_size + share_data->octomap_resolution * i;
					double y = share_data->map_center(1) - share_data->map_size + share_data->octomap_resolution * j;
					double z = max(share_data->min_z_table, share_data->map_center(2) - share_data->map_size) + share_data->octomap_resolution * k;
					auto node = share_data->octo_model->search(x, y, z);
					if (node == NULL) fout << 0.307692 << '\n'; //超出地图的一般认为是空的
					else fout << node->getOccupancy() << '\n';
				}
	}
	else if (share_data->method_of_IG == 9) { //PCNBV
		share_data->access_directory(share_data->pcnbv_path + "/data");
		ofstream fout_pointcloud(share_data->pcnbv_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_pc" + to_string(iterations) + ".txt");
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGB>);
		//�������������
		pcl::RandomSample<pcl::PointXYZRGB> ran;
		ran.setInputCloud(share_data->cloud_final);
		ran.setSample(1024); //�����²������Ƶĵ���
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
						double x = share_data->map_center(0) - share_data->map_size + share_data->octomap_resolution * i;
						double y = share_data->map_center(1) - share_data->map_size + share_data->octomap_resolution * j;
						double z = max(share_data->min_z_table, share_data->map_center(2) - share_data->map_size) + share_data->octomap_resolution * k;
						auto node = share_data->octo_model->search(x, y, z);
						if (node == NULL) fout << 0.307692 << '\n'; //超出地图的一般认为是空的
						else fout << node->getOccupancy() << '\n';
					}
			//view state
			ofstream fout_viewstate(share_data->sc_net_path + "/data/" + share_data->name_of_pcd + "_r" + to_string(share_data->rotate_state) + "_v" + to_string(share_data->first_view_id) + "_vs.txt");
			if (share_data->MA_SCVP_on) { //�����MA-SCVP������ӵ�����
				for (int i = 0; i < now_view_space->views.size(); i++) {
					if (now_view_space->views[i].vis)	fout_viewstate << 1 << '\n';
					else  fout_viewstate << 0 << '\n';
				}
			}
		}
	}
	else { //��������
		int num_of_cover = 1;
		int num_of_voxel = 0;
		//����views_informaiton
		if (iterations == 0) (*now_views_infromation) = new Views_Information(share_data, nbv_plan->voxel_information, now_view_space, iterations);
		else (*now_views_infromation)->update(share_data, now_view_space, iterations);
		if (share_data->method_of_IG == GMC) {
			//����GMC����ȡȫ���Ż�����
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
			//��֤��ĸ��Ϊ0����ʵ������
			num_of_voxel = max(num_of_voxel, 1);
		}
		else if (share_data->method_of_IG == MCMF) {
			//��������������ȡȫ���Ż�����
			views_voxels_MF* set_cover_solver = new views_voxels_MF(share_data->num_of_max_flow_node, now_view_space, *now_views_infromation, nbv_plan->voxel_information, share_data);
			set_cover_solver->solve();
			vector<int> coverage_view_id_set = set_cover_solver->get_view_id_set();
			for (int i = 0; i < coverage_view_id_set.size(); i++)
				now_view_space->views[coverage_view_id_set[i]].in_coverage[iterations] = 1;
			delete set_cover_solver;
		}
		//�ۺϼ���ֲ�̰����ȫ���Ż��������ӵ���Ϣ��
		share_data->sum_local_information = 0;
		share_data->sum_global_information = 0;
		share_data->sum_robot_cost = 0;
		for (int i = 0; i < now_view_space->views.size(); i++) {
			share_data->sum_local_information += now_view_space->views[i].information_gain;
			share_data->sum_global_information += now_view_space->views[i].get_global_information();
			share_data->sum_robot_cost += now_view_space->views[i].robot_cost;
		}
		//��֤��ĸ��Ϊ0����ʵ������
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
	//���±�־λ
	share_data->now_views_infromation_processed = true;
}

void move_robot(View* now_best_view, View_Space* now_view_space, Share_Data* share_data) {
	// update macvp combined information
	if (nbv_plan->iterations + 1 == share_data->num_of_nbvs_combined) { //Combined+MASCVP�л�
		share_data->method_of_IG = SCVP;
		sort(now_view_space->views.begin(), now_view_space->views.end(), view_id_compare);
	}
	// check if iteration is over
	if (share_data->num_of_max_iteration > 0 && nbv_plan->iterations + 1 >= share_data->num_of_max_iteration) share_data->over = true;
	// if over return
	if (share_data->over) {
		share_data->waypoints.clear();
		share_data->move_on = true;
	}	
	// update flag outside
	// share_data->move_on = true;
}

void show_cloud(pcl::visualization::PCLVisualizer::Ptr& viewer) {
	//pcl��ʾ����
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void get_command()
{	//�ӿ���̨��ȡָ���ַ���
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
	//NBV�滮�ڳ�ʼ��
	nbv_plan = new NBV_Planner(share_data);
	//����ѭ��
	string status="";
	//ʵʱ��ȡ��滮
	while (!stop && nbv_plan->plan()) {
		//���״̬�б仯�����
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

int main_base()
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
	//rotate_states.push_back(6);
	//rotate_states.push_back(7);
	vector<int> first_view_ids;
	first_view_ids.push_back(0);
	//first_view_ids.push_back(2);
	//first_view_ids.push_back(4);
	//first_view_ids.push_back(14);
	//first_view_ids.push_back(27);
	//for (int i = 0; i < 32; i++)
	//	first_view_ids.push_back(i);
	int combined_test_on;
	cout << "combined on:";
	cin >> combined_test_on;
	//scvp��7��Ҫ���ܣ�����������ȡ�����������ܾͰ�Ĭ��ֵ
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
	//���Լ�
	vector<string> names;
	cout << "input models:" << endl;
	string name;
	while (cin >> name) {
		if (name == "-1") break;
		names.push_back(name);
	}
	//ѡȡģʽ
	if (mode == DebugOne)
	{
		//��������ʼ��
		share_data = new Share_Data("../DefaultConfiguration.yaml");
		//����̨��ȡָ���߳�
		thread cmd(get_command);
		//NBVϵͳ�����߳�
		thread runner(get_run);
		//�ȴ��߳̽���
		runner.join();
		cmd.join();
		delete share_data;
	}
	else if (mode == DebugAll){
		//����0�ӵ�,��ѡ����
		for (int i = 0; i < names.size(); i++) {
			//��������ʼ��
			share_data = new Share_Data("../DefaultConfiguration.yaml", names[i]);
			//NBVϵͳ�����߳�
			thread runner(get_run);
			//�ȴ��߳̽���
			runner.join();
			delete share_data;
		}
	}
	else if (mode == TestAll) {
		//�����������塢�ӵ㡢����
		for (int i = 0; i < names.size(); i++) {
			for (int j = 0; j < rotate_states.size(); j++) {
				for (int k = 0; k < first_view_ids.size(); k++) {
					//��������ʼ��
					share_data = new Share_Data("../DefaultConfiguration.yaml", names[i], rotate_states[j], first_view_ids[k], method_id, move_test_on, combined_test_on);
					//NBVϵͳ�����߳�
					thread runner(get_run);
					//�ȴ��߳̽���
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
			// ��ȡ����ߴ�
			share_data = new Share_Data("../DefaultConfiguration.yaml", names[i], 0);
			nbv_plan = new NBV_Planner(share_data);
			ofstream write;
			write.open("D:\\����\\MA-SCVP\\size.txt", ios::app);
			write << share_data->predicted_size << endl;
			write.close();
			*/

			//for (int rotate_state = 0; rotate_state < 8; rotate_state++) {
			for (int x = 0; x < rotate_states.size(); x++) {
				int rotate_state = rotate_states[x];
				//��������ʼ��
				share_data = new Share_Data("../DefaultConfiguration.yaml", names[i], rotate_state);
				//NBV�滮�ڳ�ʼ��
				nbv_plan = new NBV_Planner(share_data);

				//��ȡȫ������
				for (int i = 0; i < nbv_plan->now_view_space->views.size(); i++) {
					//nbv_plan->percept->precept(&nbv_plan->now_view_space->views[i]);
				}

				int num = 0;
				unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
				for (int j = 0; j < share_data->cloud_final->points.size(); j++) {
					octomap::OcTreeKey key = share_data->ground_truth_model->coordToKey(share_data->cloud_final->points[j].x, share_data->cloud_final->points[j].y, share_data->cloud_final->points[j].z);
					if (voxel->find(key) == voxel->end()) {
						(*voxel)[key] = num++;
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
