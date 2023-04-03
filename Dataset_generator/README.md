# Dataset_generator
Our view planning datasets can be accessed at [Kaggle](https://www.kaggle.com/datasets/sicongpan/ma-scvp-dataset).  
These codes are for the generation of your own dataset.  
## Installion
These libraries need to be installed: opencv 4.4.0, PCL 1.9.1, Eigen 3.3.9, OctoMap 1.9.6, Gurobi 10.0.0.  
Note that Gurobi is free for academic use.  
Our codes can be compiled by Visual Studio 2022 with c++ 14 and run on Windows 11.  
For other system, please check the file read/write or multithreading functions in the codes.  
## Note
Change "const static size_t maxSize = 100000;" to "const static size_t maxSize = 1000" in file OcTreeKey.h, so that the code will run faster.
## Usage
The mode of the system should be input in the Console.  
Then give the object model names in the Console (-1 to break input).  
Make sure "model_path" in DefaultConfiguration.yaml contains these processed 3D models.  
You may find our pre-porcessed 3D models at [Kaggle](https://www.kaggle.com/datasets/sicongpan/ma-scvp-dataset).  
Or use [this sampling method](https://github.com/PointCloudLibrary/pcl/blob/master/tools/mesh_sampling.cpp) to process your own 3D object model from *.obj or *.ply to *.pcd.  
The "pre_path" in DefaultConfiguration.yaml is the dataset saving path.  
# Mode 0
The system will label the object "name_of_pcd" in DefaultConfiguration.yaml with a set of view cases in test_view_cases.txt.  
# Mode 1
The system will perform NBV reconstruction on all object cases to get the sampling space and the longtail distribution.  
# Mode 2
This mode only works after running the Mode 1.  
The system will genertate both NBVR and Longtail32 datasets on all object cases.  
# Mode 3
This mode only works after running the Mode 2.  
The system will genertate both NBVR and Longtail8 subsets.
# Mode 4
This mode only works after running the Mode 3.  
The system will check the datasets are correct or not.  
