#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:52:45 2019
@author: akshita
"""
import os
os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4  python train_main.py --gammaD 10 --gammaG 10 \
--gzsl  --manualSeed 3483 --encoded_noise --preprocessing --cuda --point_embedding PointNet --class_embedding ModelNet40_w2v \
--nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataroot datasets --dataset ModelNet40 \
--nclass_all 44 --nclass_seen 30 --batch_size 64 --nz 300 --latent_size 300 --attSize 300 --resSize 1024 --syn_num 300 \
--recons_weight 0.01 --feedback_loop 2 --a1 1 --a2 0.01  --dec_lr 0.0001 ''')

