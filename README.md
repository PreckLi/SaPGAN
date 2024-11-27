# SaPGAN
Official implementation of SaPGAN, a privacy-preserving model in the paper "Beyond Rigid Perturbation: Adaptive Semantic-Aware Privacy-Preserving for LLMs"
## Preparation
The Datasets path should be set by your path and download the dataset files.

Pretrained models should download from [huggingface](https://huggingface.co/) or [modelscope](https://www.modelscope.cn/home) and move them to your own model path.
## Requirements
- datasets==2.21.0
- evaluate==0.4.3
- huggingface-hub==0.24.6
- numpy==1.26.4
- pandas==2.2.2
- peft==0.12.0
- scikit-learn==1.5.1
- scipy==1.14.1
- tokenizers==0.19.1
- torch==2.4.0+cu124
- transformers==4.44.2
## Run
```
python main.py
```
