# Omni-Mol: Multitask Molecular Model for Any-to-any Modalities
GitHub REPO for paper *Omni-Mol: Multitask Molecular Model for Any-to-any Modalities*

<div align="center">

<a href="https://arxiv.org/abs/2502.01074"><img src="https://img.shields.io/static/v1?label=NeurIPS2025&message=Paper&color=red"></a>
<a href="https://huggingface.co/datasets/CodeMagic/Omni-Mol-Dataset"><img src="https://img.shields.io/static/v1?label=HuggingFace&message=Models+Data&color=yellow"></a>
<img src="https://img.shields.io/badge/Num_Tasks-16-blue">
[![GitHub Repo stars](https://img.shields.io/github/stars/1789336421/Omni-Mol)](https://github.com/1789336421/Omni-Mol/stargazers)


</div>

## 🔉 News
- [2025.11] We release our model and inference code! We're currently refactoring our entire codebase to ensure it's clean and well-organized, which we believe will benefit the community. The training code is on the road!
- [2025.11] We release our dataset.
- [2025.10] We are cleaning our code and peparing the data, they will be released soon.
- [2025.09] The paper is accepted by NeurIPS2025.

## 📖 TL;DR: What is This Paper About?
- **Big Instruction Tuning Data**: We curated a substantial amount of instruction tuning data for small molecules and found that **training on multiple tasks can benefit individual tasks**. Specifically, we observed higher performance compared to using separate LoRA weights for each task.
- **MoGE**: To further improve performance, we propose MoGE, which fine-tunes the model by integrating a modified LoRA adapter with an MoE expansion layer. This approach not only boosts performance across multiple tasks but also leads to more balanced results.
- **A Unified Model**: Together, we build a unified model that can solve 16 tasks using a shared LLM backbone.

## 🤖 What Tasks can Omni-Mol Do?
Omni-Mol is trained on 16 tasks, the detail is summarized below
| Category | Name |
| :--- | :--- |
| `Mo12Mo1` | <code>Forward</code>, <code>Reagent</code>, <code>Retrosynthesis</code>, <code>Solvent</code>, <code>Catalyst</code>, <code>MolEdit</code> |
| `Mo12Num` | <code>Quantum Mechanics Property Prediction Task</code>, <code>Molecular Weight</code>, <code>TPSA</code>, <code>LogP</code>, <code>Yield</code> |
| `Mo12Text` | <code>Experimental Procedure</code>, <code>Description QA</code>, <code>Molcap</code> |
| `Text2Mol` | <code>IUPAC Name to SELFIES</code>, <code>MolDesign</code> |

> Jointly training seemingly unrelated tasks, such as Molcap and the Quantum Mechanics Property Prediction Task, can also lead to performance improvements for both.

## 💿 Install
First clone this project
```bash
git clone https://github.com/1789336421/Omni-Mol.git
cd Omni-Mol
```

Install `uv`
```bash
pip install uv
```

Install dependencies
```bash
uv sync
```

Activate the correct virtual environment
```bash
conda deactivate <Your current conda venv>
source .venv/bin/activate
```

Install PyTorch according to your CUDA version
```bash
# First clean cache
uv cache clean

# Install PyTorch according to your CUDA version (cu118, cu121, cu124)
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

Install packages that need compilation with PyTorch
```bash
uv pip install deepspeed==0.17.4
uv pip install torch-geometric==2.6.1
uv pip install torch-scatter==2.1.2 --no-build-isolation
# NOTE: This may fail if there are network issues (unable to access GitHub)
uv pip install flash_attn==2.5.9.post1 --no-build-isolation
uv pip install ogb==1.3.6
uv pip install accelerate==1.9.0
```

And finally, install our custmoized `PEFT`
```bash
cd peft
uv pip install -e .
```

### Configure NLTK and WordNet
Metric calculation relies on NLTK and its submodule WordNet, if your network is fine, just run
```python
import nltk
nltk.download('wordnet')
```
Then the metric calculation for texts is set. Otherwise, you should manually install wordnet for NLTK.

The NLTK data can be placed in these directories
```bash
'/root/nltk_data'
'/root/miniconda3/nltk_data'
'/root/miniconda3/share/nltk_data'
'/root/miniconda3/lib/nltk_data'
'/usr/share/nltk_data'
'/usr/local/share/nltk_data'
'/usr/lib/nltk_data'
'/usr/local/lib/nltk_data'
```
1. Download NLTK WordNet from https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip
2. Create a directory in any of the directories listed above
3. Create a dirctory structure like this: `nltk_data/corpora/`
4. Move the `wordnet.zip` into `nltk_data/corpora/`
5. Run `unzip wordnet.zip` and `rm wordnet.zip`

### Download Llama from Meta
Download `meta-llama/Llama-3.2-1B-Instruct` from HuggingFace, it is used as the language backbone `LLM_BACKBONE` in `scripts/test.sh`. Also, please put the LLM in a folder like this:
```bash
Omni-Mol/llms/Llama-3.2-1B-Instruct
```

## 📊 Evaluation
Download the dataset, model weights from HuggingFace
```python
from huggingface_hub import snapshot_download

cache_dir = "cache/download"
local_dir = "omnimol_assets/"
repo_id = "CodeMagic/Omni-Mol-Dataset"
while True:
    try:
        snapshot_download(
          cache_dir=cache_dir,
          local_dir=local_dir,
          repo_id=repo_id,
          local_dir_use_symlinks=False,
          resume_download=True,
          repo_type="dataset",
        )
    except Exception as e :
        print(e)
    else:
        print('Assets downloaded')
        break
```

Configure the `scripts/test.sh` script.

| Argument | Type | Description |
| :--- | :---: | :--- |
| `DATA_BASE_DIR` | str | Base directory of the whole dataset, e.g. `omnimol_assets/test`, the folder contains all test JSON files. |
| `SAVE_BASE_DIR` | str | Base directory to save the answer and the metrics, all inference results will be saved into this folder |
| `MODEL_PATH` | str | Path to the Omni-Mol checkpoint | 
| `LLM_BACKBONE` | str | Path to the LLM backbone | 
| `GRAPH_PATH` | str | Path to moleculestm weight | 
| `NUM_GPUS` | int | Set number of GPUs to use in the inference, we support parallel inference on multiple GPUs, set to 1 if you have only one GPU.
| `TOP_P` | float | top_p sampling parameter, default to 1.0 |
| `TEMPERATURE` | float | temperature used in sampling, default to 0.2 | 
| `NUM_BEAMS` | int | searching beams in sampling, default to 1 |
| `MAX_NEW_TOKENS`| int | Maximum new tokens in generation, default to 512 |
| `REPETITION_PENALTY` | float | Repetition penalty factor, default to 1.0 | 

The script iterates through the following defined tasks and corresponding filenames, make sure the filename is correct.

| Task Name (`--task_name`) | Dataset Filename |
| :--- | :--- |
| `forward` | `forward_prediction` |
| `retrosynthesis` | `retrosynthesis` |
| `reagent` | `reagent_prediction` |
| `solvent` | `solvent_pred` |
| `catalyst` | `catalyst_pred` |
| `homolumo` | `property_prediction` |
| `molcap` | `molcap` |
| `dqa` | `DescriptionQA` |
| `weight` | `Molecular_Weight` |
| `logp` | `LogP` |
| `iupac` | `IUPAC2SELFIES` |
| `tpsa` | `TPSA` |
| `textguidedmolgen` | `text_guided_mol_generation` |
| `yield_BH` | `yield_regression_BH` |
| `yield_SM` | `yield_regression_SM` |
| `molediting` | `molecule_editing` |
| `experiment` | `exp_procedure_pred` |

To run the batch inference, ensure your paths are configured in the script variables and run:

```bash
bash scripts/test.sh
```
This will generate two files for each task, one is the model output answer, the filename will be `{task_name}.json`structured as
```json
{
  "prompt": "Prompt used in the geneartion.",
  "gt": "Ground truth of this task.",
  "pred": "Model prediction."
}
```
Another one is the metric related to this task, the filename is `{task_nam}_metric.json`.

> [!NOTE]
> We're releasing two model versions: `Version 1` represents the main results presented in our paper, while `Version 2` delivers superior performance across all tasks except Yield Regression. Both versions were trained under **identical experimental conditions**.

> [!WARNING]
> We are still refactoring the part of the code for molecule editing task.

## 🔭 Future Directions
Here, we provide our insights about this area and the possible future research directions.

> [!NOTE]
> While we recognize these limitations, they are not the central objective that this project aims to tackle.

### 1 Benchmarks
Some tasks currently lack robust metrics or benchmarks. We believe that establishing reliable benchmarks is essential for future research.

> **`Limitation 1`**: In the Molecular Captioning (Molcap) task, the model is required to generate an answer that matches the reference text. However, the model may correctly predict certain molecular properties that are not included in the reference. In such cases, the evaluation metric may underestimate the model’s true performance. A similar issue arises in the Description QA task: the final performance should be determined by the correctness of the answer, rather than the degree of overlap between the model’s output and the reference answer.

**`Possible Solution`**: Developing benchmarks with definitive answers—such as multiple-choice questions, single-choice questions, and true/false questions—where the model must select the one correct answer from a set of correct and incorrect options, can avoid the aforementioned problems. While some existing works have developed multiple-choice benchmarks, we believe the data volume can be further expanded and more diverse, verifiable question formats can be incorporated.

> **`Limitation 2`**: Tasks related to chemical reactions, such as retrosynthesis, actually have multiple solutions; that is, multiple possible sets of reactants can lead to the same product. However, current answer verification methods often only validate one of these solutions, which may also lead to an underestimation of the model's performance.

**`Possible Solution`**: Developing verifiers that can verify the reaction outcome. For example, use a forward prediction model to verify whether the proposed reactants can produce the desired product.

### 2 Training Data
The available community dataset contains only final answers; however, we posit that training data which includes reasoning steps can significantly improve a model's ability to generalize.

> **`Limitation`**: Current community datasets typically provide only questions and final answers, which compels models to generate answers directly without intermediate reasoning steps. We argue that incorporating explicit thinking processes is crucial for developing robust models. Since reasoning is a transferable skill that generalizes across tasks, such data would not only simplify the current task by breaking it down into steps but also cultivate generalizable abilities applicable to a wider range of problems. Furthermore, it introduces the significant advantage of model interpretability, allowing us to trace the reasoning behind each prediction.

**`Possible Solution`**: To create high-quality SFT data, one can use rejection sampling on commercial or open-source LLMs to distill reasoning steps. Researchers should also explore diverse data synthesis techniques. Additionally, mixing in general reasoning data during training is recommended. After the SFT, one can further train the model with **RLVR**, the popular DeepSeek-R1 like training.

>[!NOTE]
> We also release some of the distilled data from Qwen 2.5, check our HuggingFace repository if you are interested!

### 3 Multi-turn Molecule Understanding
If joint training is able to improve the performance of individual tasks, it is highly probable that training the model to solve diverse tasks based on the same molecules will enable the model to develop a deeper understanding of those molecules. One can search our dataset to collect the tasks that involve a certain molecule or set of molecules, and then use commercial LLMs to construct multi-turn conversation data.

## 🌟 Acknowledgement
We thank [MoleculeSTM](https://github.com/chao1224/MoleculeSTM) for their GNN encoder, we also thank [InstructMol](https://github.com/IDEA-XL/InstructMol), [PRESTO](https://github.com/IDEA-XL/PRESTO/tree/main) for their prior explorations.

## ✍️ Citation
```bibtex
@inproceedings{
  hu2025omnimol,
  title={Omni-Mol: Multitask Molecular Model for Any-to-any Modalities},
  author={Chengxin Hu and Hao Li and Yihe Yuan and Zezheng Song and Chenyang Zhao and Haixin Wang},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```
