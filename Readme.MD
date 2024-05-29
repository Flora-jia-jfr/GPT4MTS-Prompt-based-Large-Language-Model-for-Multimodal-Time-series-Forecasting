# GPT4MTS: Prompt-based Large Language Model for Multimodal Time-series Forecasting

### Preparing environment:

```
conda create -n GPT4MTS python=3.8
conda activate GPT4MTS
pip3 install -r requirements.txt
pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Run code
A demo bash script has been prepared. The `model_name` could be any one between: `DLinear`, `NLinear`, `GPT4TS`, `GPT4MTS`
```
bash run_demo.sh
```