#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


set -x
set -o pipefail

args=$(cat ../configs/TDETR-dance.args)
python ../submit_dance.py ${args} --exp_name tracker --resume $1
