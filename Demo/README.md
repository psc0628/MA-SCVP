# The demo of trained MA-SCVP network
## Installion
These libraries need to be installed: python 3.8.8, pytorch 1.8.1.
## Usage
```bash
python infer.py ma-scvp_longtail32.pth.tar Dragon
```
## Explanation
The MA-SCVP network takes the 32x32x32 occupancy grid and the view state vector as input.  
<img src="https://github.com/psc0628/MA-SCVP/blob/main/Demo/Dragon_voxelscene.png" width="400px"> <img src="https://github.com/psc0628/MA-SCVP/blob/main/Demo/Dragon_viewstate.png" width="300px">
The MA-SCVP network outputs the ideally smallest view set to cover all remaining object surfaces.   
<img src="https://github.com/psc0628/MA-SCVP/blob/main/Demo/Dragon_cover.png" width="400px">