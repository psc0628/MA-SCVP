import torch
import numpy as np

from models.pc_nbv import AutoEncoder

#DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
DEVICE = torch.device('cpu')

def infer_once(accumulate_pointcloud, view_states, network):
    network.eval()
    network.to(DEVICE)
    accumulate_pointcloud = accumulate_pointcloud.to(DEVICE)
    view_states = view_states.to(DEVICE)
    accumulate_pointcloud = accumulate_pointcloud.permute(0, 2, 1)
    
    _, pred_value = network(accumulate_pointcloud, view_states)

    return pred_value


if __name__ == '__main__':
    model_param_path = "./log/300.pth"
    cloud_name = "./cloud_0.txt"
    score_name = "./score_0.txt"
    
    network = AutoEncoder(views=32)
    network.load_state_dict(torch.load(model_param_path),map_location = torch.device('cpu'))
    
    accumulate_pointcloud = np.genfromtxt(cloud_name, dtype=np.float32).reshape(-1, 3)
    view_state = np.genfromtxt(score_name, dtype=np.float32)

    accumulate_pointcloud = torch.from_numpy(accumulate_pointcloud).unsqueeze(0)
    view_state = torch.from_numpy(view_state).unsqueeze(0)
    print(accumulate_pointcloud.shape, view_state.shape)
    print(torch.argmax(infer_once(accumulate_pointcloud, view_state, network), dim=1))
    
    
