# Fusing LLM Capabilities with Routing Data

<p align="center">
    <a href="https://ulab-uiuc.github.io/FusionBench/">
        <img alt="Build" src="https://img.shields.io/badge/Project-Page-blue">
    </a>
    <a href="http://arxiv.org/abs/xxxx.xxxxx">
        <img alt="Build" src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-red?logo=arxiv">
    </a>
    <!-- <a href="xxx">
        <img alt="Build" src="https://img.shields.io/badge/Twitter-black?logo=X">
    </a> -->
    <a href="https://github.com/ulab-uiuc/FusionBench/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
    <br>
    <a href="https://github.com/ulab-uiuc/FusionBench">
        <img alt="Build" src="https://img.shields.io/github/stars/ulab-uiuc/FusionBench">
    </a>
    <a href="https://github.com/ulab-uiuc/FusionBench">
        <img alt="Build" src="https://img.shields.io/github/forks/ulab-uiuc/FusionBench">
    </a>
    <a href="https://github.com/ulab-uiuc/FusionBench">
        <img alt="Build" src="https://img.shields.io/github/issues/ulab-uiuc/FusionBench">
    </a>
</p>


<p align="center">
    <a href="https://ulab-uiuc.github.io/FusionBench/">🌐 Project Page</a> |
    <a href="http://arxiv.org/abs/xxxx.xxxxx">📜 arXiv</a>
    <!-- <a href="xxx">📮 Twitter Post</a> -->
<p>



<div align="center">
  <img src="./figures/comparison.jpg" width="700" alt="FusionBench">
  <p><b>Comparison of FusionBench with Existing Router-related Work</b></p>
  <img src="./figures/construction.jpg" width="700" alt="FusionBench">
  <p><b>Overview of FusionBench’s Construction Process</b></p>
  <img src="./figures/fusion.jpg" width="700" alt="FusionBench">
  <p><b>Overview of LLM capability fusion explored through FusionBench, focusing on three representative levels: query-level, thought-level, and model-level.</b></p>
</div>


## News

**[2025.06]** 🌟 **FusionBench** was released.



## 🛠️Environment Setup

```bash
conda create -n fusionbench python=3.9
conda activate fusionbench
pip install pandas
pip install datasets
pip install tqdm
pip install transformers
pip install litellm
pip install sentence_transformers
pip install torch
pip install numpy
```



## 🎯Data Process

Run the following command to start data collection.

```bash
# split: train OR test
python data_process/data_combine.py --split train --case_num 500 --round 5
```




## 📊Experiments


### Query-level Fusion




### Thought-level Fusion



### Model-level Fusion


Run the following command to generate SFT data for model-level fusion.

```bash
# setting: perf, judge, hybrid, baseline
python model_level/sft_data_gen.py --settin perf --k 5 --save_path [YOUR_PATH] --csv_path_with_judge [YOUR_PATH]

python model_level/sft_test_gen.py --save_path [YOUR_PATH] --csv_path [YOUR_PATH]
```


You can refer to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for detailed instructions to start fine-tuning on model-level fusion data.




## 🔍 Evaluation

### Running Evaluation

To evaluate a model on a specific dataset, use the following command:

```bash
# dataset: Specifies the dataset to evaluate on
# model_path: Path to the trained model you want to evaluate
python eval/eval.py --dataset <dataset_name> --model_path <path_to_model>
```





## Citation

```bibtex
@article{FusionBench,
  title={Fusing LLM Capabilities with Routing Data},
  author={Tao Feng and Haozhen Zhang and Zijie Lei and Pengrui Han and Mostofa Patwary and Mohammad Shoeybi and Bryan Catanzaro and Jiaxuan You},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```
