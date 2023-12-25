import sys
import os
import time
import numpy as np

name_of_objects = []
user_input = input('input object name:')
while user_input != '-1':
    name_of_objects.append(user_input)
    user_input = input('input object name:')
        
rotate_ids = []
rotate_ids.append(0)
# rotate_ids.append(1)
rotate_ids.append(2)
rotate_ids.append(3)
rotate_ids.append(4)
rotate_ids.append(5)
# rotate_ids.append(6)
# rotate_ids.append(7)

first_view_ids = []
first_view_ids.append(0)
first_view_ids.append(1)
first_view_ids.append(2)
# first_view_ids.append(3)
first_view_ids.append(4)
# first_view_ids.append(5)
first_view_ids.append(6)
first_view_ids.append(7)
first_view_ids.append(8)
# first_view_ids.append(9)
# first_view_ids.append(10)
# first_view_ids.append(11)
first_view_ids.append(12)
# first_view_ids.append(13)
first_view_ids.append(14)
# first_view_ids.append(15)
first_view_ids.append(16)
# first_view_ids.append(17)
first_view_ids.append(18)
first_view_ids.append(19)
first_view_ids.append(20)
first_view_ids.append(21)
first_view_ids.append(22)
# first_view_ids.append(23)
first_view_ids.append(24)
first_view_ids.append(25)
# first_view_ids.append(26)
# first_view_ids.append(27)
first_view_ids.append(28)
first_view_ids.append(29)
first_view_ids.append(30)
# first_view_ids.append(31)

for object_name in name_of_objects:
    print('testing '+ object_name)
    for rotate_id in rotate_ids:
        for view_id in first_view_ids:
            while os.path.isfile('./data/'+object_name+'_r'+str(rotate_id)+'_v'+str(view_id)+'_vs.txt')==False:
                pass
            time.sleep(1)
            os.system('python infer.py ' + './pth/longtail32_lambda2.0_bsize8_lr0.0004.pth.tar ' + object_name+'_r'+str(rotate_id)+'_v'+str(view_id))
            # os.system('python infer.py ' + './pth/nbvsample32_lambda3.0_bsize8_lr0.0004.pth.tar ' + object_name+'_r'+str(rotate_id)+'_v'+str(view_id))
            f = open('./log/' + object_name+'_r'+str(rotate_id)+'_v'+str(view_id) + '_ready.txt','a')
            f.close()
            print('testing '+ object_name+'_r'+str(rotate_id) +'_v'+str(view_id) + ' over.')
print('all over.')
