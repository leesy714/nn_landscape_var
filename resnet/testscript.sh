#!/bin/bash
for i in {1..5};
do
     CUDA_VISIBLE_DEVICES=0 python train.py --arch ResNetNoShort56 --seed $i --optim sgd --lr 1e-1 --batch-size 128; 
done

