import torch
import torch.nn as nn
import numpy as np
from icecream import ic

def flatten(l):
    return [item for sublist in l for item in sublist]

def refactor(batch_dict, lidar_agent_indicator):
        agent_num = len(lidar_agent_indicator)
        proposal_agentids_sample_list = batch_dict['agentid_fused'] # [sample1, sample2, ..., sample{batchnum}]

        lidar_matrix_list = []
        camera_matrix_list = []

        # scatter agentid
        for proposal_agentids_list in proposal_agentids_sample_list: # [[0,1,2],[1,2],[0,2],...]
            proposal_num = len(proposal_agentids_list)

            sp_row = [[i]*len(proposal_agentids_list[i]) for i in range(len(proposal_agentids_list))]
            sp_row = flatten(sp_row)
            sp_col = torch.cat(proposal_agentids_list).tolist()

            indice = np.array([sp_row, sp_col], dtype=np.int32)
            value = np.ones_like(sp_row)

            lidar_matrix = torch.sparse_coo_tensor(indice, value, (proposal_num, agent_num), device=lidar_agent_indicator.device).to_dense()
            camera_matrix = torch.sparse_coo_tensor(indice, value, (proposal_num, agent_num), device=lidar_agent_indicator.device).to_dense()

            lidar_mask = (lidar_agent_indicator)
            camera_mask = (1 - lidar_agent_indicator)

            lidar_matrix *= lidar_mask
            camera_matrix *= camera_mask

            lidar_matrix_list.append(lidar_matrix)
            camera_matrix_list.append(camera_matrix)

        batch_dict['lidar_matrix_list'] = lidar_matrix_list
        batch_dict['camera_matrix_list'] = camera_matrix_list

        return batch_dict