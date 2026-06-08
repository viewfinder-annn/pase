# PASE: Phonologically Anchored Speech Enhancer
[![arxiv](https://img.shields.io/badge/arXiv-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2511.13300)
[![AAAI](https://img.shields.io/badge/AAAI-blue?logo=google-scholar&logoColor=white)](https://ojs.aaai.org/index.php/AAAI/article/view/40562)
[![demo](https://img.shields.io/badge/Demo-orange?logo=audacity&logoColor=white)](https://xiaobin-rong.github.io/pase_demo/)
[![models](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/cisco-ai/pase)

🎉 This is the official implementation of our AAAI 2026 paper: 

[PASE: Leveraging the Phonological Prior of WavLM for Low-Hallucination Generative Speech Enhancement](https://ojs.aaai.org/index.php/AAAI/article/view/40562).

## Pretrained Checkpoints 
Three checkpoints are provided:
- `DeWavLM.tar`
- `Vocoder-L24.tar`
- `Vocoder-Dual.tar`

`DeWavLM.tar` and `Vcooder-Dual.tar` together form the PASE model.

Note that the released checkpoint is trained on a relatively small dataset, including:
- **Speech**: DNS5, LibriTTS, VCTK
- **Noise**: DNS5
- **RIRs**: OpenSLR26, OpenSLR28

The performance of the retrained version compared to the original one:
| Model | DNSMOS | UTMOS | SBS | LPS | SpkSim | WER (%) |
|:-----:|:------:|:-----:|:---:|:---:|:------:|:-------:|
| Vocoder-L24 (orig.) | 3.23 | 3.40 | 0.94 | 0.97 | 0.65 | 2.86 |
| **Vocoder-L24 (retrained)** | 3.29 | 3.30 | 0.94 | 0.96 | 0.59 | 3.46 |
| DeWavLM (orig.) | 3.26 | 3.42 | 0.88 | 0.93 | 0.57 | 7.62 |
| **DeWavLM (retrained)** | 3.31 | 3.39 | 0.88 | 0.93 | 0.52 | 7.25
| PASE (orig.) | 3.12   | 3.09  |0.90 |0.93 |0.80    | 7.49    |
| **PASE (retrained)** | 3.08 | 3.21 | 0.91 | 0.94 | 0.80 | 6.76 |

It can be seen that the retrained version achieves performance very close to that of the original version on our simulated test set.

**Note**: The Vocoder-L24 (retrained) was trained for only 60 epochs (30k iterations), as we found that it tends to overfit on such a small training set.

## Inference
To run inference on audio files, make sure you are in the `pase` directory and use:

```bash
python -m inference.inference -I <input_dir> -O <output_dir> [options]
```

| Argument       | Requirement / Default | Description                                                                 |
|----------------|-----------------------|-----------------------------------------------------------------------------|
| `-I` (`--input_dir`)  | **required**          | Path to the input directory containing audio files.                   |
| `-O` (`--output_dir`) | **required**          | Path to the output directory where enhanced files will be saved.      |
| `-D` (`--device`)     | default: `cuda:0`     | Torch device to run inference on, e.g., `cuda:0`, `cuda:1`, or `cpu`. |
| `-E` (`--extension`)  | default: `.wav`       | Audio file extension to process.                                      |

Audio examples are provided in `../test/audio_enh`.

## Training
### Step 1: Training a single-stream vocoder
- training script: `train/train_vocoder.py`
- training configuration: `configs/cfg_train_vocoder.yaml`

    ```bash
    python -m train.train_vocoder -C configs/cfg_train_vocoder.yaml -D 0,1,2,3
    ```
- inference script: `inference/infer_vocoder.py`
    ```bash
    python -m inference.infer_vocoder -C configs/cfg_infer.yaml -D 0
    ```

This step aims to pre-train a vocoder using the 24th-layer WavLM representations. The pre-trained single-stream vocoder is then used in Step 2 to reconstruct waveforms, enabling the evaluation of DeWavLM’s performance.

### Step 2: Finetuning WavLM
- training script: `train/train_dewavlm.py`
- training configuration: `configs/cfg_train_dewavlm.yaml`
- inference script: `inference/infer_dewavlm.py`

(The usage is the same as in Step 1.)

This step aims to obtain a denoised WavLM (DeWavLM) via knowledge distillation, referred to in the paper as denoising representation distillation (DRD).

### Step 3: Training a dual-stream vocoder
- training script: `train/train_vocoder_dual.py`
- training configuration: `configs/cfg_train_vocoder_dual.yaml`
- inference script: `inference/infer_vocoder_dual.py`

(The usage is the same as in Step 1.)

This step trains the final dual-stream vocoder, which takes the acoustic (1st-layer) and phonetic (24th-layer) DeWavLM representations as inputs and produces the final enhanced waveform.

## Creating Checkpoints
Once all training steps are completed, the corresponding checkpoints can be prepared for inference:
- `utils/create_ckpt_wavlm.py` is used to create a DeWavLM checkpoint.
- `utils/create_ckpt.py` is used to create a Vocoder-L24 or Vocoder-Dual checkpoint.

## Citation
If you find this work useful, please cite our paper:
```bibtex
@article{PASE, 
    title={{PASE: Leveraging the Phonological Prior of WavLM for Low-Hallucination Generative Speech Enhancement}},
    volume={40},
    DOI={10.1609/aaai.v40i39.40562}, 
    number={39}, 
    journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
    author={Rong, Xiaobin and Hu, Qinwen and Yesilbursa, Mansur and Wojcicki, Kamil and Lu, Jing}, 
    year={2026},
    month={Mar.}, 
    pages={32826-32834} }
```

## Contact
Xiaobin Rong: [xiaobin.rong@smail.nju.edu.cn](mailto:xiaobin.rong@smail.nju.edu.cn)

Mansur Yesilbursa: [myesilbu@cisco.com](myesilbu@cisco.com)