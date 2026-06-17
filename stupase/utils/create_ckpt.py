import torch
from omegaconf import OmegaConf


train_cfg_path = "/work/user_data/xiaobin/Experiments/exp_StuPASE/exp_flow_2026-06-10-10h36m/config.yaml"
train_ckpt_path = "/work/user_data/xiaobin/Experiments/exp_StuPASE/exp_flow_2026-06-10-10h36m/checkpoints/best_model_177.tar"
save_ckpt_path = "/work/user_data/xiaobin/Pre-trained/StuPASE_new/CFM.pt"

config = OmegaConf.load(train_cfg_path)
config =  OmegaConf.to_container(config)
state_dict = torch.load(train_ckpt_path, map_location='cpu')

new_dict = {}

if 'generator' in state_dict.keys():
    model_dict = state_dict['generator']
elif 'model' in state_dict.keys():
    model_dict = state_dict['model']
else:
    raise ValueError("Keys mismatch!")

new_dict['model'] = model_dict
new_dict['cfg'] = config['cfm_config']

torch.save(new_dict, save_ckpt_path)
print("Successfully created a pre-trained checkpoint:", save_ckpt_path)
print("Keys:", new_dict.keys())


