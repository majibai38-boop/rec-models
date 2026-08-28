#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_TORCH_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
export PRE_GRAPH_OPTIMIZER=1
DEVICE=${DEVICE:-npu} # cpu/npu/cuda
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12347}
NODE_RANK=${NODE_RANK:-0}
DEVICE_ID=${DEVICE_ID:-0}
IFS=',' read -r -a DEVICE_IDS <<< "${DEVICE_ID}"
NPROC_PER_NODE=${NPROC_PER_NODE:-${#DEVICE_IDS[@]}}
NNODES=${NNODES:-1}
OPEN_PROFILING=${OPEN_PROFILING:-true}
PREPROCESSED_DATASET=${PREPROCESSED_DATASET:-"${SCRIPT_DIR}/data_process/aliccp_out"}
MODEL_DIR=${MODEL_DIR:-"${SCRIPT_DIR}/checkpoint"}
LOAD_CHECKPOINT=${LOAD_CHECKPOINT:-false}
SAVE_CHECKPOINT=${SAVE_CHECKPOINT:-false}
REPORT_DIR=${REPORT_DIR:-"${SCRIPT_DIR}/reports"}
PROFILING_PATH=${PROFILING_PATH:-"${SCRIPT_DIR}/profiling"}
MODE=${MODE:-train}
NUM_EPOCHS=${NUM_EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-4096}
LEARNING_RATE=${LEARNING_RATE:-0.001}
EMBEDDING_SIZE=${EMBEDDING_SIZE:-32}
GRU_TYPE=${GRU_TYPE:-GRU}
DIEN_DROPOUT=${DIEN_DROPOUT:-0.0}
DIEN_ATTENTION_WEIGHT_NORMALIZATION=${DIEN_ATTENTION_WEIGHT_NORMALIZATION:-true}
DIEN_POSITIVE_CLASS_WEIGHT=${DIEN_POSITIVE_CLASS_WEIGHT:-6.142857142857142}
DNN_HIDDEN_SIZE=${DNN_HIDDEN_SIZE:-}
ATT_HIDDEN_SIZE=${ATT_HIDDEN_SIZE:-}
TRAIN_STOP_STEP=${TRAIN_STOP_STEP:--1}
VAL_STOP_STEP=${VAL_STOP_STEP:--1}
export JOB_ID="${JOB_ID:-10087}"

if [ "$DEVICE" = "npu" ]; then
    export ASCEND_RT_VISIBLE_DEVICES="${DEVICE_ID}"
    echo "set npu"
elif [ "$DEVICE" = "cuda" ]; then
    export CUDA_VISIBLE_DEVICES="${DEVICE_ID}"
    echo "set gpu"
fi

echo "use ${DEVICE}:${DEVICE_ID}"

DIEN_ARGS=(
  "--gru_type=${GRU_TYPE}"
  "--dien_dropout=${DIEN_DROPOUT}"
  "--dien_attention_weight_normalization=${DIEN_ATTENTION_WEIGHT_NORMALIZATION}"
  "--dien_positive_class_weight=${DIEN_POSITIVE_CLASS_WEIGHT}"
)
if [ -n "${DNN_HIDDEN_SIZE}" ]; then
    DIEN_ARGS+=("--dnn_hidden_size=${DNN_HIDDEN_SIZE}")
fi
if [ -n "${ATT_HIDDEN_SIZE}" ]; then
    DIEN_ARGS+=("--att_hidden_size=${ATT_HIDDEN_SIZE}")
fi

PYTHONPATH="${TRAINING_TORCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
torchrun \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --module \
  behavior_and_multi_task.main \
  --master_port="${MASTER_PORT}" \
  --device="${DEVICE}" \
  --device_id="${DEVICE_ID}" \
  --mode="${MODE}" \
  --data_dir="${PREPROCESSED_DATASET}" \
  --model_dir="${MODEL_DIR}" \
  --load_checkpoint="${LOAD_CHECKPOINT}" \
  --save_checkpoint="${SAVE_CHECKPOINT}" \
  --report_dir="${REPORT_DIR}" \
  --profiling_mode="${OPEN_PROFILING}" \
  --profiling_path="${PROFILING_PATH}" \
  --num_epochs="${NUM_EPOCHS}" \
  --batch_size="${BATCH_SIZE}" \
  --learning_rate="${LEARNING_RATE}" \
  --embedding_size="${EMBEDDING_SIZE}" \
  --hf32=true \
  --graph=false \
  --compile=false \
  --dynamic_batch=false \
  --train_stop_step="${TRAIN_STOP_STEP}" \
  --val_stop_step="${VAL_STOP_STEP}" \
  "${DIEN_ARGS[@]}" \
  "$@" \
  --model=dien

echo "finished!"
