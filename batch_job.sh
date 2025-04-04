#!/bin/bash

#SBATCH -J tail-temp-array
#SBATCH --time=12:0:0
# Number of GPU
#SBATCH --gres=gpu:2
#SBATCH --array=0-5
# Start your application

models=("qwen_ct", "qwen_ct", "phi_ct", "phi_ct", "llama_ct", "llama_ct")
prompts=("zero_shot_cot" "few_shot_cot" "zero_shot_cot" "few_shot_cot" "zero_shot_cot" "few_shot_cot")

# Get the combination for this task
model=${models[$SLURM_ARRAY_TASK_ID]}
prompt=${prompts[$SLURM_ARRAY_TASK_ID]}


# Activate python venv if on Uni HPC
source /mnt/nas_home/aa2613/temporal-error-analysis/.venv/bin/activate

echo "Running model: $model, prompt: $prompt"

#! Full path to application executable: 
application="python"
SCRIPT="/mnt/nas_home/aa2613/temporal-error-analysis/clear_temptabqa/script.py"

#! Run options for the application:
options="$SCRIPT --model "$model" --prompt "$prompt" --split tail-temp"


CMD="$application $options"

eval $CMD


