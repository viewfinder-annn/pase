# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""
URGENT 2025 speech data, for (single-stream) vocoder training and validation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import random
import pandas as pd
from typing import Union, List, Dict
from torch.utils import data
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf
from utils import simulate_utils


def parse_scp(scp_paths: List[str]) -> Dict[str, Dict]:
    data_dict = {}
    for scp in scp_paths:
        with open(scp, "r", encoding="utf-8") as f:
            for line in f:
                # The format of each line in the SCP file is: uid fs audio_path
                uid, fs, audio_path = line.strip().split()
                data_dict[uid] = {
                    "fs": int(fs),
                    "path": audio_path,
                }
    return data_dict


class URGENT2Dataset(data.Dataset):
    def __init__(
        self,
        speech_scps: List,
        wav_len=4, 
        num_per_epoch=10000, 
        random_start=False,
        default_fs = 16000,
        mode='train',
        seed=0
    ):
        super().__init__()
        assert mode in ['train', 'validation']
        
        self.speech_dict = parse_scp(speech_scps)
                           
        self.wav_len = wav_len
        self.num_per_epoch = num_per_epoch
        self.random_start = random_start
        self.default_fs = default_fs
        
        self.mode = mode
        self.seed = seed

        speech_uids = list(self.speech_dict.keys())
        
        rng = np.random.default_rng(self.seed)
        rng.shuffle(speech_uids)
        
        self.meta = []
        for i, s_uid in enumerate(speech_uids):
            item = {"id": f"utt_{i}", "speech_uid": s_uid}
            self.meta.append(item)

        print(f"[{mode}] #speech={len(speech_uids)}")
        self.sample_data_per_epoch(mode)
    
    def sample_data_per_epoch(self, mode='train'):
        if mode == 'train':
            self.meta_selected = random.sample(self.meta, self.num_per_epoch)
        else:  # select fixed data when in validation or test
            self.meta_selected = self.meta[:self.num_per_epoch]
    
        
    def __getitem__(self, idx):
        info = self.meta_selected[idx]
        
        uid = info["id"]
        fs = self.default_fs
        rng = np.random.default_rng(self.seed + idx if self.mode != "train" else random.randint(0, 10**9))

        speech_path = self.speech_dict[info["speech_uid"]]["path"]
        try:
            speech_sample = simulate_utils.read_audio(speech_path, force_1ch=True, fs=fs)[0]
        except:
            print(speech_path)
        
        orig_len = speech_sample.shape[1]
        
        # select a segmen with a fixed duration in seconds
        if self.wav_len != 0:  # wav_len=0 means no cut or padding, use in test
            seg_len = int(self.wav_len*fs)
            if seg_len < orig_len:
                start_point = rng.integers(0, orig_len-seg_len) if self.random_start else 0
                speech_sample = speech_sample[:, start_point: start_point + seg_len]
            elif seg_len > orig_len:
                pad_points = seg_len - orig_len
                speech_sample = np.pad(speech_sample, ((0, 0), (0, pad_points)), constant_values=0)
        
        # normalization
        scale = rng.uniform(0.5, 0.95)
        speech_sample = speech_sample / (np.max(np.abs(speech_sample)) + 1e-9) * scale
        
        info = {'id': uid, 'fs': fs, 'length': orig_len}

        return speech_sample.astype(np.float32), info
    
    
    def __len__(self):
        return len(self.meta_selected)

   
 
if __name__ == "__main__":
    import os
    from tqdm import tqdm
    from omegaconf import OmegaConf
    import soundfile as sf
    
    config = OmegaConf.load('configs/cfg_train_vocoder.yaml')

    train_dataset = URGENT2Dataset(**config['train_dataset'])
    train_dataloader = data.DataLoader(train_dataset, **config['train_dataloader'])

    shape0 = None

    tmp_dir = "/work/user_data/xiaobin/Datasets/dataloader_samples/train_samples"
    os.makedirs(tmp_dir, exist_ok=True)
    os.system(f"rm {tmp_dir}/*")
    
    train_dataloader.dataset.sample_data_per_epoch()
    for step, (clean, info) in enumerate(tqdm(train_dataloader)):
        if shape0 is None:
            shape0 = clean.shape
            print(shape0)
        shape = clean.shape
        assert shape == shape0
        
        if step < 10:
            sf.write(f"{tmp_dir}/{info['id'][0]}_clean.wav", clean[0].numpy().squeeze(), int(info['fs'][0]))
        if step == 10:
            break
    
    valid_dataset = URGENT2Dataset(**config['validation_dataset'])
    valid_dataloader = data.DataLoader(valid_dataset, **config['validation_dataloader'])

    tmp_dir = '/work/user_data/xiaobin/Datasets/dataloader_samples/valid_samples'
    os.makedirs(tmp_dir, exist_ok=True)
    os.system(f"rm {tmp_dir}/*")

    shape0 = None
    for step, (clean, info) in enumerate(tqdm(valid_dataloader)):
        if shape0 is None:
            shape0 = clean.shape
            print(shape0)
        shape = clean.shape
        assert shape == shape0

        if step < 10:
            sf.write(f"{tmp_dir}/{info['id'][0]}_clean.wav", clean[0].numpy().squeeze(), int(info['fs'][0]))
        if step == 10:
            break
        
