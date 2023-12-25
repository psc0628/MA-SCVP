# View_planning_simulator
This is the view planning simulation system for object reconstruction.  
## Installion
These libraries need to be installed: opencv 4.4.0, PCL 1.9.1, Eigen 3.3.9, OctoMap 1.9.6, Gurobi 10.0.0.  
Note that Gurobi is free for academic use.  
Our codes can be compiled by Visual Studio 2022 with c++ 14 and run on Windows 11.  
For other system, please check the file read/write or multithreading functions in the codes.  
## Note
Change "const static size_t maxSize = 100000;" to "const static size_t maxSize = 1000" in file OcTreeKey.h, so that the code will run faster.  
## Prepare
Make sure "model_path" in DefaultConfiguration.yaml contains these processed 3D models.  
You may find our pre-porcessed 3D models at [Kaggle](https://www.kaggle.com/datasets/sicongpan/ma-scvp-dataset).  
Or use [this sampling method](https://github.com/PointCloudLibrary/pcl/blob/master/tools/mesh_sampling.cpp) to process your own 3D object model from *.obj or *.ply to *.pcd.  
The "pre_path" in DefaultConfiguration.yaml is the results saving path.  
Move GT_points_num folder to "gt_path" in DefaultConfiguration.yaml.  
The "nbv_net_path", "pcnbv_path", "sc_net_path" should be the downloaded network paths.  
## Usage
The mode of the system should be input in the Console.  
Then input 1 for combined pipeline and 0 for single method.  
Next input the method id you want to test. 0: MCMF, 3: RSE, 4: APORA, 6: NBVNET, 7: (MA-)SCVP, 8: Random, 9: PCNBV, 10: GMC.  
And if the method is search-based, you can input 1 for movement cost version and 0 otherwise.  
Finally give the object model names in the Console (-1 to break input).  
### Change in methods and setup
If you run with our combined pipeline, you can change the number of NBVs by "num_of_nbvs_combined" in DefaultConfiguration.yaml.  
If you run with combined pipeline, you have to input an NBV method, i.e., do not input 7 .  
If you run with NBV methods, you can change the number of maximum itearitons by "num_of_max_iteration" in DefaultConfiguration.yaml. 
After you run a network test, you have to remove "data" and "log" folder in the network path for re-testing.   
### Mode 0
The system will genertate the ground truth point clouds of all visble voxels for all input objects. This will speed up evluation.  
### Mode 1
The system will test the object "name_of_pcd" by "method_of_IG" method in DefaultConfiguration.yaml with rotation 0 and intital view 0.  
### Mode 2
The system will test all input objects by the input method with rotations and intital views in lines 1369-1411 of main.cpp.  



