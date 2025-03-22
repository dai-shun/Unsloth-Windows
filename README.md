# Unsloth-Windows
- asa

```
conda create -n unsloth python=3.13.2
conda activate unsloth

pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install --no-deps trl peft accelerate bitsandbytes
pip install deepspeed-0.13.1+unknown-py3-none-any.whl
pip install  triton-2.1.0-cp311-cp311-win_amd64.whl
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install xformers==0.0.25.post1
```