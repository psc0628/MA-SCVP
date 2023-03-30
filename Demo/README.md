# The demo of MA-SCVP network
## Installion
These libraries need to be installed: python 3.8.8, pytorch 1.8.1.
## Usage
```bash
python infer.py ma-scvp_longtail32.pth.tar Dragon
```
## Explanation
The MA-SCVP network takes the 32x32x32 occupancy grid and the view state vector as input
![image](https://github.com/psc0628/MA-SCVP/blob/main/Demo/Dragon_voxelscene.png) ![image](https://github.com/psc0628/MA-SCVP/blob/main/Demo/Dragon_viewstate.png)
The MA-SCVP network outputs the ideally smallest view set to cover all remaining object surfaces.
![image](https://github.com/psc0628/MA-SCVP/blob/main/Demo/Dragon_cover.png)