#!/bin/bash

set -e

# Enrollment phase 
python3 ./enroll.py \
    --file_list ./data/enroll.list \
    --model_dir model/ \
    --speaker_id spk1 \
    --sample_rate 8000 \
    --gmm_number 16

# verification phase
python3 ./eval.py \
    --file_list ./data/eval.list \
    --model_dir model/ \
    --speaker_id spk1 \
    --sample_rate 8000 \
    --threshold -10.0 \
