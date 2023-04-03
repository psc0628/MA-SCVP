import sys
import os
import time
import numpy as np

name_of_objects = []
with open('object_name.txt', 'r',encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        name_of_objects.append(line)
        
rotate_ids = []
#rotate_ids.append(0)
#rotate_ids.append(1)
#rotate_ids.append(2)
rotate_ids.append(3)
#rotate_ids.append(4)
#rotate_ids.append(5)
rotate_ids.append(6)
#rotate_ids.append(7)

first_view_ids = []
first_view_ids.append(0)
first_view_ids.append(2)
first_view_ids.append(4)
first_view_ids.append(14)
first_view_ids.append(27)

for object_name in name_of_objects:
    print('testing '+ object_name)
    for rotate_id in rotate_ids:
        for view_id in first_view_ids:
            while os.path.isfile('./data/'+object_name+'_r'+str(rotate_id)+'_v'+str(view_id)+'_vs.txt')==False:
                pass
            time.sleep(1)
            os.system('python eval_single_file_mascvp.py ' + './pth/last.pth.tar ' + object_name+'_r'+str(rotate_id)+'_v'+str(view_id))
            f = open('./log/ready.txt','a')
            f.close()
            print('testing '+ object_name+'_r'+str(rotate_id) +'_v'+str(view_id) + ' over.')
print('all over.')
