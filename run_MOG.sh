#!/bin/bash

conda activate ldm

python Two_Dimensional_MOG_ddgan_continuous_siddms.py --num_timesteps 4  --ac_w 1.0
