# Explainable Fake News Detection under Distribution Shift

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code accompanying the article:

> Abiola, K.; Enamamu, T.S.; Ajao, O. **Explainable Fake News Detection under Distribution Shift: A Comparison of Fully Fine-Tuned Encoder and QLoRA-Adapted Decoder Configurations.** *Analytics* (under review), 2026.

The study trains four transformer classifiers on one fake-news corpus and asks how far their performance carries over to other corpora, and what they are actually learning. It compares:

| Configuration | Models | Adaptation |
|---|---|---|
| Encoder | BERT-base, RoBERTa-base | Full fine-tuning |
| Decoder | Llama 3.2-3B, Qwen2.5-3B | QLoRA (4-bit NF4, LoRA r = 16, α = 32) |

All models are trained on **GonzaloA** and evaluated zero-shot on **Pulk17**, **LIAR** and **WELFake**. LIME and SHAP are used to inspect RoBERTa's predictions, which revealed a Reuters-style dateline shortcut (`"reuters -"`) that survived preprocessing and pushed predictions towards *Real*.

---

## Repository contents

| File | Purpose |
|---|---|
| `FakeNewsDetection.ipynb` | Original end-to-end notebook: preprocessing, training of all four models, in-domain testing, cross-dataset evaluation, and LIME/SHAP analysis. Outputs are kept so logs, training times and metrics can be checked without re-running. |
| `requirements.txt` | Python dependencies. |

### Notebook structure

1. Setup: dependencies, imports, Hugging Face authentication
2. GonzaloA preprocessing and word-cloud exploration
3. RoBERTa: tokenisation, training/validation, testing, prediction
4. BERT: same pipeline as RoBERTa
5. Llama 3.2-3B: prompt formatting, QLoRA/LoRA configuration, training, label extraction, testing
6. Qwen2.5-3B: same pipeline as Llama
7. Training-curve comparison
8. Cross-dataset evaluation on Pulk17, LIAR and WELFake, with confusion matrices
9. LIME and SHAP explanations and LIME-SHAP agreement analysis

---

## Datasets

All datasets are public and are downloaded automatically from Hugging Face; none are redistributed here.

| Dataset | Role | Source |
|---|---|---|
| GonzaloA | Training, validation, in-domain test | [GonzaloA/fake_news](https://huggingface.co/datasets/GonzaloA/fake_news) |
| Pulk17 | Cross-dataset (long-form articles) | [Pulk17/Fake-News-Detection-dataset](https://huggingface.co/datasets/Pulk17/Fake-News-Detection-dataset) |
| LIAR | Cross-dataset (short political claims) | [UKPLab/liar](https://huggingface.co/datasets/UKPLab/liar) |
| WELFake | Cross-dataset (multi-source articles) | [davanstrien/WELFake](https://huggingface.co/datasets/davanstrien/WELFake) |

Labels are harmonised to **Fake = 0, Real = 1**. LIAR uses the binary labels supplied with the UKPLab release (inverted to this convention), and each statement is joined with its `context` field.

---

## Setup

The experiments were run on **Google Colab with an NVIDIA A100 (80 GB)**. The encoders will train on smaller GPUs; the 3B-parameter decoders need roughly 24 GB or more even with 4-bit quantisation.

```bash
git clone https://github.com/<username>/clarinews-fake-news-detection.git
cd clarinews-fake-news-detection
pip install -r requirements.txt
```

**Hugging Face access.** Llama 3.2 is a gated model: accept Meta's licence on its [model page](https://huggingface.co/meta-llama/Llama-3.2-3B), then create an access token. In Colab, store it as a secret named `HF_TOKEN` (the notebook reads it with `userdata.get('HF_TOKEN')`). Elsewhere, run `huggingface-cli login`.

**Paths.** The notebook mounts Google Drive and saves checkpoints under `/content/drive/MyDrive/...`. Change these paths if you run it elsewhere.

---

## Training settings

| | Encoders (BERT, RoBERTa) | Decoders (Llama, Qwen) |
|---|---|---|
| Optimiser | AdamW, weight decay 0.01, linear schedule, 10% warm-up | AdamW via TRL `SFTTrainer` |
| Learning rate | 2e-5 | 2e-4 |
| Epochs | Up to 5, early stopping (patience 2) on validation macro F1 | Up to 5, early stopping (patience 2) on validation loss |
| Batch size | 16 | 8 per device × 4 gradient accumulation = 32 |
| Input | Truncated to 512 tokens | Instruction prompt, article truncated to 1,500 characters, max 512 tokens |
| Quantisation / adapters | — | 4-bit NF4, double quantisation, bfloat16 compute; LoRA r = 16, α = 32, dropout 0.05 on all attention and MLP layers |
| Training time (A100) | ~44 min each | Llama 138.8 min, Qwen 180.4 min |

---

## Running

**Original pipeline.** Open `FakeNewsDetection.ipynb` in Colab and run the cells in order. Each model section can also be run on its own after the setup and GonzaloA preprocessing cells.

---

## Citation

If you use this code, please cite the article (details will be updated on publication) and the archived software:

```bibtex
@article{abiola2026explainable,
  title   = {Explainable Fake News Detection under Distribution Shift: A Comparison of Fully Fine-Tuned Encoder and QLoRA-Adapted Decoder Configurations},
  author  = {Abiola, Kofoworola and Enamamu, Timiboudi S. and Ajao, Oluwaseun},
  journal = {Analytics},
  year    = {2026},
  note    = {Under review}
}

@software{abiola2026code,
  author    = {Abiola, Kofoworola and Enamamu, Timiboudi S. and Ajao, Oluwaseun},
  title     = {Code for: Explainable Fake News Detection under Distribution Shift},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX}
}
```

## Authors

- **Kofoworola Abiola**, School of Computing & Mathematics, Manchester Metropolitan University
- **Timiboudi S. Enamamu**, School of Engineering & Computing, University of Lancashire
- **Oluwaseun Ajao** (corresponding author), School of Computing & Mathematics, Manchester Metropolitan University. s.ajao@mmu.ac.uk

## License

Code is released under the [MIT License](LICENSE). The datasets and pretrained models remain under their original licences, including the Llama 3.2 Community License for Llama.
