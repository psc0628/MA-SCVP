# The demo of trained MA-SCVP network
## Installion
Install pytorch 1.8.1 with python 3.8.8. Or you can download our packed conda enviroment pytorch.tar.gz at [Kaggle](www.kaggle.com/datasets/sicongpan/ma-scvp-dataset).
## Usage
```bash
python infer.py ma-scvp_longtail32.pth.tar Dragon
```
## Explanation
The MA-SCVP network takes the 32x32x32 occupancy grid (the bounding box in the scene shown in left) and the view state vector (red-green-blue views shown in right) as input.  
<img src="https://github.com/psc0628/MA-SCVP/blob/main/Demo/Dragon_voxelscene.png" width="300px"> <img src="https://github.com/psc0628/MA-SCVP/blob/main/Demo/Dragon_viewstate.png" width="240px">  
The MA-SCVP network outputs the ideally smallest view set (red-green-blue views) to cover all remaining object surfaces.   
<img src="https://github.com/psc0628/MA-SCVP/blob/main/Demo/Dragon_cover.png" width="450px">