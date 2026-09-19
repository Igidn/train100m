#!/bin/bash
# Launch the full checkpoint-final verification on the Pi (runs in tmux: cktest)
cd ~/ai/ckpttest/toktest
export VAL_BS=${VAL_BS:-1}
exec ~/ai/ckpt-venv/bin/python verify_final.py \
    ~/ai/ckpttest/final-model.pt pi 2>&1 | tee verify_final_pi.log
