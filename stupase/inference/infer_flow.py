# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from librosa.util import find_files
from omegaconf import OmegaConf
from models.wavlm.feature_extractor import WavLM_feat
from models.vocoder.vocos.vocoder import VocosVocoder as Vocoder
from models.flow.cfm import CFM as Model

steps = 8
cfg_strength = 0.5
sway_sampling_coef = -1.0

@torch.inference_mode()
def infer(args):
    cfg_infer = OmegaConf.load(args.config)
    cfg_network = OmegaConf.load(cfg_infer.network.config)
    
    noisy_folder = cfg_infer.test_dataset.noisy_dir
    clean_folder = cfg_infer.test_dataset.clean_dir
    save_folder = f"{cfg_infer.network.enh_folder}_steps={steps}_strength={cfg_strength}_sway={sway_sampling_coef}"
    os.makedirs(save_folder, exist_ok=True)
    
    ext = cfg_infer.test_dataset.extension
    
    wavs = sorted(find_files(noisy_folder, ext=ext))
    print(f"Inference on folder: {noisy_folder}, {len(wavs)} files")
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    encoder = WavLM_feat(**cfg_network['encoder_config']).to(device).eval()
    vocoder = Vocoder.from_pretrained(**cfg_network['vocoder_config']).to(device)
    model = Model(**cfg_network['cfm_config']).to(device).eval()
    
    model.load_state_dict(
        torch.load(cfg_infer['network']['checkpoint'], map_location=device)['model']
    )

    wavs = sorted(find_files(noisy_folder, ext='wav'))
    print(f"Inference on folder: {noisy_folder}, {len(wavs)} files")

    inf_scp_list = []
    ref_scp_list = []
    
    for wav_path in tqdm(wavs):
        noisy, fs = sf.read(wav_path, dtype='float32')
            
        input = torch.FloatTensor(noisy)[None].to(device)
        
        feat = encoder(input, sr=fs)
        output, _  = model.sample(
            cond=input, cond_noisy=input, text=feat, no_ref_audio=True, vocoder=vocoder,
            steps=steps, cfg_strength=cfg_strength, sway_sampling_coef=sway_sampling_coef
        )
        
        esti_wav = output.cpu().detach().numpy().squeeze()
        
        esti_wav = esti_wav / (np.abs(esti_wav).max() + 1e-8) * 0.5
        
        if esti_wav.shape[-1] < noisy.shape[-1]:
            esti_wav = np.pad(esti_wav, (0, noisy.shape[-1]-esti_wav.shape[-1]))
        else:
            esti_wav = esti_wav[..., :noisy.shape[-1]]
        
        uid = os.path.basename(wav_path).split(f'.{ext}')[0]
        
        true_path = os.path.join(clean_folder, f'{uid}.{ext}')
        esti_path = os.path.join(save_folder, f'{uid}.{ext}')
    
        sf.write(esti_path, esti_wav, fs)
        
        inf_scp_list.append([uid, esti_path])
        ref_scp_list.append([uid, true_path])
        
    # Save paths into scp file for evaluation
    with open(os.path.join(save_folder, "inf.scp"), "w") as f:
        for uid, audio_path in inf_scp_list:
            f.write(f"{uid} {audio_path}\n")

    with open(os.path.join(save_folder, "ref.scp"), "w") as f:
        for uid, audio_path in ref_scp_list:
            f.write(f"{uid} {audio_path}\n")

            

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='configs/cfg_infer.yaml')
    parser.add_argument('-D', '--device', default='0', help='Index of the gpu device')

    args = parser.parse_args()
    infer(args)
