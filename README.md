# FusionFactory: Fusing LLM Capabilities with Multi-LLM Log Data

<p align="center">
    <a href="https://ulab-uiuc.github.io/FusionFactory/">
        <img alt="Project Page" src="https://img.shields.io/badge/Project-Page-blue">
    </a>
    <a href="http://arxiv.org/abs/2507.10540">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2507.10540-red?logo=arxiv">
    </a>
    <!-- <a href="xxx">
        <img alt="Twitter" src="https://img.shields.io/badge/Twitter-black?logo=X">
    </a> -->
    <a href="https://github.com/ulab-uiuc/FusionFactory/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
    <br>
    <a href="https://github.com/ulab-uiuc/FusionFactory">
        <img alt="Stars" src="https://img.shields.io/github/stars/ulab-uiuc/FusionFactory">
    </a>
    <a href="https://github.com/ulab-uiuc/FusionFactory">
        <img alt="Forks" src="https://img.shields.io/github/forks/ulab-uiuc/FusionFactory">
    </a>
    <a href="https://github.com/ulab-uiuc/FusionFactory">
        <img alt="Issues" src="https://img.shields.io/github/issues/ulab-uiuc/FusionFactory">
    </a>
</p>

<p align="center">
    <a href="https://ulab-uiuc.github.io/FusionFactory/">🌐 Project Page</a> |
    <a href="http://arxiv.org/abs/2507.10540">📜 arXiv</a> |
    <a href="https://huggingface.co/datasets/ulab-ai/FusionBench">📂 Dataset</a> |
    <a href="https://huggingface.co/ulab-ai/FusionFactory">🤖 Model</a> |
    <a href="https://huggingface.co/spaces/ulab-ai/RoutePilot">🖥️ Demo</a>
</p>




<div align="center">
  <img src="./figures/fusion.jpg" width="700" alt="FusionBench">
  <p><b>Overview of LLM capability fusion via FusionFactory with three representative levels: Query-level, Thought-level, and Model-level.</b></p>
</div>


## News

**[2025.06]** 🌟 **FusionFactory** was released.



## 🛠️Environment Setup

```bash
conda create -n fusionfactory python=3.9
conda activate fusionfactory
pip install pandas
pip install datasets
pip install tqdm
pip install transformers
pip install sentence_transformers
pip install torch
pip install numpy
```



## 🎯Data Process

Run the following command to start data collection.

```bash
# split: train OR test
# case num: 500 for train & 50 for partial test
# a sample of LLM description: ./data_process/LLM_Descriptions.json
python data_process/data_combine.py \
--split train \
--case_num 500 \
--round 5 \
--llm_description_path [YOUR_LLM_PATH] \
--csv_save_path [YOUR_SAVE_PATH] \
--api_base [YOUR_API_BASE] \
--api_key [YOUR_API_KEY]
```

You may refer to the specific README in the [`data_process`](data_process/README.md) directory for detailed argument descriptions.

To add quality scores to the collected data using an LLM judge:

```bash
python data_process/add_llm_judge.py
```

This will evaluate each response and add quality scores to the dataset, which can be used for training and evaluation purposes. See the [`data_process/README.md`](data_process/README.md) for more details.




## 📊Experiments


### Query-level Fusion

First, run the data preprocessing script to prepare the dataset:

```bash
# Preprocess the dataset and generate training/testing files
python query_level/data_processing.py
```

For more detailed information about the data preprocessing and model training process, please refer to the specific README in the [`query_level`](query_level/README.md) directory.



### Thought-level Fusion
First, run the data preprocessing script to prepare the thought prompts:

```bash
# Preprocess the dataset and generate training/testing files
python query_level/data_processing.py
```

Or run the script to directly use Huggingface datasets to generate thought-enhanced queries

```bash
python thought_level/get_thought_prompt.py
```

For more detailed information about the data preprocessing and model training process, please refer to the specific README in the [`thought_level`](thought_level/README.md) directory.


### Model-level Fusion

You can refer to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for detailed instructions to start fine-tuning on model-level fusion data. Make sure to first clone the LLaMA-Factory repository into the FusionBench directory, and then execute the following commands to generate SFT data for model-level fusion:


```bash
# setting: perf, judge, hybrid, baseline
python model_level/sft_data_gen.py --settin perf --k 5 --save_path [YOUR_PATH] --csv_path_with_judge [YOUR_PATH]

python model_level/sft_test_gen.py --save_path [YOUR_PATH] --csv_path [YOUR_PATH]
```

Then, you can use the following commands to start SFT and Inference after essential configuration described in [LLaMA-Factory Doc](https://llamafactory.readthedocs.io/en/latest/)

```bash
# SFT
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=2,3,4,5 llamafactory-cli train examples/train_lora/[YOUR_YAML].yaml

# Inference
CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/vllm_infer.py --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --adapter_name_or_path saves/llama3.1-8b/lora/[YOUR_PATH] --dataset router_test --cutoff_len 2048
```


You may refer to the specific README in the [`model_level`](model_level/README.md) directory for detailed instructions.


## 📈 Evaluation

FusionBench provides a comprehensive evaluation framework to assess model performance across various tasks. The evaluation framework supports multiple types of tasks including:

- Mathematical Reasoning (GSM8K, MATH)
- Code Generation (MBPP, HumanEval)
- Commonsense Reasoning (CommonsenseQA, OpenBookQA, ARC Challenge, HellaSwag)
- World Knowledge (Natural Questions, TriviaQA)
- Reading Comprehension (SQuAD, BoolQ)
- Popular Benchmarks (MMLU, GPQA)

To evaluate your model's performance:

```bash
python eval/response_eval.py
```

For detailed information about the evaluation framework, supported metrics, and usage instructions, please refer to the [Evaluation Documentation](eval/README.md).


## Citation

```bibtex
@article{feng2025fusing,
  title={Fusing LLM Capabilities with Routing Data},
  author={Feng, Tao and Zhang, Haozhen and Lei, Zijie and Han, Pengrui and Patwary, Mostofa and Shoeybi, Mohammad and Catanzaro, Bryan and You, Jiaxuan},
  journal={arXiv preprint arXiv:2507.10540},
  year={2025}
}
```
