# PCNBV network
Thanks to [PCNBV](https://github.com/zhanhz/PC-NBV_pytorch).
## Usage
Run default tests.  
```bash
python run_test_rotate_view_parallel.py
```
Input the names of objects to be tested and end with "-1".  
## Change test setup
Change the number of maximum iteration in line 106 (if you run with our pipeline, this should equals to the number of NBVs).  
Change the pre-trained models in line 9.  
Change the rotation set and initial view set in lines 18-60 which corresponds to the view planning tests.  