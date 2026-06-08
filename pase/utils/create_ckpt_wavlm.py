import os
import torch
from models.wavlm.feature_extractor import WavLM_feat

model = WavLM_feat()
ckpt_finetuned = "/work/user_data/xiaobin/Experiments/exp_PASE_new/exp_DeWavLM_2026-03-24-14h57m/checkpoints/model_200.tar"
checkpoint = torch.load(ckpt_finetuned, map_location="cpu")

model.wavlm.load_state_dict(checkpoint['model'])

state_dict = {
    "cfg": model.wavlm.cfg.__dict__,
    "model": model.wavlm.state_dict(),
        
    }

save_path = "/work/user_data/xiaobin/Pre-trained/PASE_new"
os.makedirs(save_path, exist_ok=True)
wavlm_ckpt_path_new = f"{save_path}/DeWavLM.tar"
torch.save(state_dict, wavlm_ckpt_path_new)
print("Successfully created a pre-trained WavLM checkpoint:", wavlm_ckpt_path_new)