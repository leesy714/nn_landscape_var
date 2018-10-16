#!/bin/bash
for i in {1..1};
do
     CUDA_VISIBLE_DEVICES=0 python train.py --epoch 300 --arch ResNet56 --seed $i --optim sgd --lr 1e-1 --batch-size 128; 
done

