# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""
URGENT 2025 dataset for speech enhancement
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
from pathlib import Path
from copy import deepcopy
from omegaconf import OmegaConf
from utils import simulate_utils
from itertools import cycle


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
        noise_scps: List,
        rir_scps: List,
        wav_len=4, 
        num_per_epoch=10000, 
        random_start=False,
        snr_range=[-5, 15],
        default_fs = 16000,
        mode='train',
        seed=0
    ):
        super().__init__()
        assert mode in ['train', 'validation']
        
        self.speech_dict = parse_scp(speech_scps)
        self.noise_dict = parse_scp(noise_scps)
        self.rir_dict = parse_scp(rir_scps)
                           
        self.wav_len = wav_len
        self.num_per_epoch = num_per_epoch
        self.random_start = random_start
        
        self.snr_range = snr_range
        self.default_fs = default_fs
        
        self.mode = mode
        self.seed = seed
        
        speech_uids = list(self.speech_dict.keys())
        noise_uids = list(self.noise_dict.keys())
        rir_uids = list(self.rir_dict.keys()) + ["none"] * len(self.rir_dict)  # 0.5 prob
        
        rng = np.random.default_rng(0)
        rng.shuffle(speech_uids)
        rng.shuffle(noise_uids)
        rng.shuffle(rir_uids)
        
        noise_cycle = cycle(noise_uids)
        rir_cycle = cycle(rir_uids)
        self.meta = [
            {"id": f"utt_{i}", "speech_uid": s_uid, "noise_uid": next(noise_cycle), "rir_uid": next(rir_cycle)}
            for i, s_uid in enumerate(speech_uids)
        ]
        
        print(f"[{mode}] #speech={len(speech_uids)}, #noise={len(noise_uids)}, #rir={len(rir_uids)}")
        
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
        
        snr = rng.integers(*self.snr_range, endpoint=True)

        speech_path = self.speech_dict[info["speech_uid"]]["path"]
        noise_path = self.noise_dict[info["noise_uid"]]["path"]
        try:
            speech_sample = simulate_utils.read_audio(speech_path, force_1ch=True, fs=fs)[0]
        except:
            print(speech_path)
        
        try:
            noise_info = sf.info(noise_path)
            noise_fs = noise_info.samplerate
            noise_length = int(noise_info.duration * noise_fs)
        except:
            print(noise_path)

        if noise_length > noise_fs * 10:
            start = rng.integers(0, noise_length-noise_fs*10)
            stop = start + 10*noise_fs
            noise_sample = simulate_utils.read_audio(noise_path, force_1ch=True, fs=fs, start=start, stop=stop)[0]
        else:
            noise_sample = simulate_utils.read_audio(noise_path, force_1ch=True, fs=fs)[0]
   
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
                
        rir_uid = info["rir_uid"]
        if rir_uid != "none":
            rir = self.rir_dict[rir_uid]["path"]
            rir_sample = simulate_utils.read_audio(rir, force_1ch=True, fs=fs)[0]
            noisy_speech = simulate_utils.add_reverberation(speech_sample, rir_sample)
            # make sure the clean speech is dry, without early reflection
            # early_rir_sample = simulate_utils.estimate_early_rir(rir_sample, fs=fs)
            # speech_sample = simulate_utils.add_reverberation(speech_sample, early_rir_sample)
        else:
            noisy_speech = speech_sample
        
        # simulation with noise mixing
        noisy_speech, noise_sample = simulate_utils.mix_noise(
            noisy_speech, noise_sample, snr=snr, rng=rng
        )
        
        # normalization
        scale = rng.uniform(0.5, 0.95)
        noisy_speech = noisy_speech / (np.max(np.abs(noisy_speech)) + 1e-9) * scale
        speech_sample = speech_sample / (np.max(np.abs(speech_sample)) + 1e-9) * scale
        
        info = {'id': uid, 'fs': fs, 'length': orig_len}

        return noisy_speech.astype(np.float32).squeeze(), speech_sample.astype(np.float32).squeeze(), info
    
    
    def __len__(self):
        return len(self.meta_selected)

   
 
if __name__ == "__main__":
    import os
    from tqdm import tqdm
    from omegaconf import OmegaConf
    import soundfile as sf
    
    config = OmegaConf.load('configs/cfg_train_dewavlm.yaml')

    train_dataset = URGENT2Dataset(**config['train_dataset'])
    train_dataloader = data.DataLoader(train_dataset, **config['train_dataloader'])

    shape0 = None

    tmp_dir = "/work/user_data/xiaobin/Datasets/dataloader_samples/train_samples"
    os.makedirs(tmp_dir, exist_ok=True)
    os.system(f"rm {tmp_dir}/*")
    
    train_dataloader.dataset.sample_data_per_epoch()
    for step, (noisy, clean, info) in enumerate(tqdm(train_dataloader)):
        if shape0 is None:
            shape0 = noisy.shape
            print(shape0)
        shape = noisy.shape
        assert shape == shape0
        
        if step < 10:
            sf.write(f"{tmp_dir}/{info['id'][0]}_noisy.wav", noisy[0].numpy().squeeze(), int(info['fs'][0]))
            sf.write(f"{tmp_dir}/{info['id'][0]}_clean.wav", clean[0].numpy().squeeze(), int(info['fs'][0]))
        if step == 10:
            break
    
    valid_dataset = URGENT2Dataset(**config['validation_dataset'])
    valid_dataloader = data.DataLoader(valid_dataset, **config['validation_dataloader'])

    tmp_dir = '/work/user_data/xiaobin/Datasets/dataloader_samples/valid_samples'
    os.makedirs(tmp_dir, exist_ok=True)
    os.system(f"rm {tmp_dir}/*")

    shape0 = None
    for step, (noisy, clean, info) in enumerate(tqdm(valid_dataloader)):
        if shape0 is None:
            shape0 = noisy.shape
            print(shape0)
        shape = noisy.shape
        assert shape == shape0

        if step < 10:
            sf.write(f"{tmp_dir}/{info['id'][0]}_noisy.wav", noisy[0].numpy().squeeze(), int(info['fs'][0]))
            sf.write(f"{tmp_dir}/{info['id'][0]}_clean.wav", clean[0].numpy().squeeze(), int(info['fs'][0]))
        if step == 10:
            break
        
