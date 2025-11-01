# Omni-Mol: Multitask Molecular Model for Any-to-any Modalities
GitHub REPO for paper *Omni-Mol: Multitask Molecular Model for Any-to-any Modalities*, Link: https://arxiv.org/abs/2502.01074

## 🔉 News
- [2025.10] We are cleaning our code and peparing the data, they will be released soon.
- [2025.09] The paper is accepted by NeurIPS2025.

## 📖 TL; DR: What is This Paper About?
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

## 🔭 Future Directions
Here, we provide our insights about this area and the possible future research directions.

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

## ✍️ Citation
```bibtex
@inproceedings{
  hu2025omnimol,
  title={Omni-Mol: Multitask Molecular Model for Any-to-any Modalities},
  author={Chengxin Hu and Hao Li and Yihe Yuan and Zezheng Song and Chenyang Zhao and Haixin Wang},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=dw9H08UxJb}
}
```
