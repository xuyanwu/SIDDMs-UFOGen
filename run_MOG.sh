#!/bin/bash

source /ocean/projects/asc170022p/yanwuxu/miniconda/etc/profile.d/conda.sh
conda activate ldm

python Two_Dimensional_MOG_ddgan_continuous.py --num_timesteps 4  --ac_w 1.0

#python Two_Dimensional_MOG_ddgan.py --num_timesteps 4 --use_AC --ac_w 4.0