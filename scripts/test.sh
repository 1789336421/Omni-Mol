#!/bin/bash

set -e

# Base directories
DATA_BASE_DIR="/root/WDHDD2TB/download/dataset/Omni-Mol-Dataset/test"
SAVE_BASE_DIR="infer/rewrite_dataset"

# Model and system paths
MODEL_PATH="/root/autodl-tmp/omni-mol-publish/my_ckpt"
LLM_BACKBONE="/root/WDHDD2TB/download/model/Llama-3.2-1B-Instruct/"
GRAPH_PATH="/root/autodl-tmp/omni-mol-publish/omni-mol-legacy/assets/moleculestm.pth"

# Hardware and Inference parameters
NUM_GPUS=4
TOP_P=1.0
TEMPERATURE=0.2
NUM_BEAMS=1
MAX_NEW_TOKENS=512
REPETITION_PENALTY=1.0

TASK_NAMES=(
    "forward"
    "retrosynthesis"
    "reagent"
    "solvent"
    "catalyst"
    "homolumo"
    "molcap"
    "dqa"
    "weight"
    "logp"
    "iupac"
    "tpsa"
    "textguidedmolgen"
    "yield_BH"
    "yield_SM"
    "molediting"
    "experiment"
)

FILENAMES=(
    "forward_prediction"
    "retrosynthesis"
    "reagent_prediction"
    "solvent_pred"
    "catalyst_pred"
    "property_prediction"
    "molcap"
    "DescriptionQA"
    "Molecular_Weight"
    "LogP"
    "IUPAC2SELFIES"
    "TPSA"
    "text_guided_mol_generation"
    "yield_regression_BH"
    "yield_regression_SM"
    "molecule_editing"
    "exp_procedure_pred"
)

if [ ${#TASK_NAMES[@]} -ne ${#FILENAMES[@]} ]; then
    echo "Error: TASK_NAMES and FILENAMES arrays have different lengths."
    echo "Please check the script configuration."
    exit 1
fi

NUM_TASKS=${#TASK_NAMES[@]}
echo "Starting inference for ${NUM_TASKS} tasks..."
echo "Saving results to: ${SAVE_BASE_DIR}"

for (( i=0; i<${NUM_TASKS}; i++ )); do

    TASK_NAME="${TASK_NAMES[i]}"
    FILE_NAME="${FILENAMES[i]}"

    DATA_PATH="${DATA_BASE_DIR}/${FILE_NAME}.json"
    SAVE_PATH="${SAVE_BASE_DIR}/${FILE_NAME}.json"

    echo ""
    echo "=================================================="
    echo "Starting task (Index ${i}): ${TASK_NAME}"
    echo "Data source:   ${DATA_PATH}"
    echo "Saving to:     ${SAVE_PATH}"
    echo "=================================================="

    python inference.py \
        --num_gpus ${NUM_GPUS} \
        --task_name "${TASK_NAME}" \
        --data_path "${DATA_PATH}" \
        --model_path "${MODEL_PATH}" \
        --language_backbone "${LLM_BACKBONE}" \
        --graph_path "${GRAPH_PATH}" \
        --top_p ${TOP_P} \
        --temperature ${TEMPERATURE} \
        --num_beams ${NUM_BEAMS} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --repetition_penalty ${REPETITION_PENALTY} \
        --answer_save_path "${SAVE_PATH}" \
        --save_metric \
        --seed 0 

    echo "Finished task: ${TASK_NAME}"

done

echo ""
echo "=================================================="
echo "All tasks completed."
echo "=================================================="
