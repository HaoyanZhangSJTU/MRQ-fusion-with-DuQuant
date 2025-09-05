## Installation
```bash
conda create -n duquant python=3.10 -y
conda activate duquant
git clone https://github.com/Hsu1023/DuQuant.git
pip install --upgrade pip 
pip install -r requirements.txt
```

## Usage
### 1. Preprocessing
```bash
python get_rot.py # need to be run only once for all models
python generate_act_scale_shift.py --model PATH_OF_MODEL # need to be run only once for each model (path can be hugging-face hub path or relative path)
```

### 2. Quantization
The bash script can be found in `run.sh`. You can choose the model to be quantized by providing model path after `--model` order. 