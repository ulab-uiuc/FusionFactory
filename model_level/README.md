# 🔧 Model-Level Fusion: Training Pipeline with LLaMA-Factory

To enable model-level fusion, we build upon the LLaMA-Factory framework. Please refer to their official documentation for more details. Below is a walkthrough for preparing data, fine-tuning, and inference.


## Installation

First, install LLaMA-Factory:

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
llamafactory-cli version  # check installation success
```


## Data Preparation

Generate training data with LLM judge annotations:

```bash
python model_level/sft_data_gen.py \
  --setting perf \
  --k 5 \
  --save_path [YOUR_PATH] \
  --csv_path_with_judge [YOUR_PATH]
```

By default, the generated data will be saved under:

```bash
./LLaMA-Factory/data/
```

You then need to modify the following file to point to your new dataset:


```bash
./LLaMA-Factory/data/dataset_info.json
```


## Customize Fine-Tuning Configuration


Refer to the YAML templates in:


```bash
./LLaMA-Factory/examples/
```
Choose or adapt a configuration file to define your SFT settings.


## Start Fine-Tuning

```bash
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=2,3,4,5 \
llamafactory-cli train examples/train_lora/[YOUR_YAML].yaml
```


## Inference Pipeline

First, generate test samples:

```bash
python model_level/sft_test_gen.py \
  --save_path [YOUR_PATH] \
  --csv_path [YOUR_PATH]
```

Update your dataset path just like in training.

Then run inference using your fine-tuned LoRA adapter:


```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 \
python scripts/vllm_infer.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --adapter_name_or_path saves/llama3.1-8b/lora/[YOUR_PATH] \
  --dataset router_test \
  --cutoff_len 2048
```



## Argument Descriptions

`sft_data_gen.py` and `sft_test_gen.py` arguments:


| Argument                | Type   | Default                                  | Description                                                                                                                            |
| ----------------------- | ------ | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `--setting`             | `str`  | `"perf"`                                 | Task type. Options: `"perf"` (performance-based), `"judge"` (LLM-judged labels), `"hybrid"` (combined), `"baseline"`. |
| `--small`               | `flag` | `False`                                  | If set, generate a subset only containing data from small models.                                                                           |
| `--k`                   | `int`  | `5`                                      | Number of candidate responses per question to include.                                                                                 |
| `--save_path`           | `str`  | `"./LLaMA-Factory/data"`                 | Path where the generated training or test data will be saved.                                                                          |
| `--csv_path_with_judge` | `str`  | `"./dataset/router_data_with_judge.csv"` | Path to the CSV file containing LLM-judged data.                                                                                       |

