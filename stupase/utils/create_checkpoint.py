import torch
from omegaconf import OmegaConf


cfg_path = "/work/user_data/xiaobin/Experiments/exp_PASE/exp_vocoder_L24_2025-12-04-12h28m/config.yaml"
ckpt_path = "/work/user_data/xiaobin/Experiments/exp_PASE/exp_vocoder_L24_2025-12-04-12h28m/checkpoints/model_001.tar"
save_path = "/work/user_data/xiaobin/Pre-trained/PASE/Vocos_L24_test.tar"

config = OmegaConf.load(cfg_path)
config =  OmegaConf.to_container(config)
state_dict = torch.load(ckpt_path, map_location='cpu')

# print(state_dict.keys())
# exit()
new_dict = {}

if 'generator' in state_dict.keys():
    model_dict = state_dict['generator']
elif 'model' in state_dict.keys():
    model_dict = state_dict['model']
else:
    raise ValueError("Keys mismatch!")

new_dict['model'] = model_dict
new_dict['cfg'] = config['vocoder_config']
# new_dict['info'] = config

print(new_dict.keys())

torch.save(new_dict, save_path)

