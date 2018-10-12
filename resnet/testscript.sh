#!/bin/bash
for i in {1..5};
do
     python train.py --instance Resnet18_optim_sgd_lr_1e-2_wd_5e-4_seed_${i}_batch_128 --seed $i --optim sgd --lr 1e-1 --batch-size 128;
     python train.py --instance Resnet18_optim_sgd_lr_1e-2_wd_5e-4_seed_${i}_batch_1024 --seed $i --optim sgd --lr 1e-1 --batch-size 1024;
done
for i in {1..5};
do
    python train.py --instance Resnet18_optim_adam_lr_1e-4_wd_5e-4_seed_${i}_batch_128 --seed $i --optim adam --lr 1e-4 --batch-size 128;
    python train.py --instance Resnet18_optim_adam_lr_1e-4_wd_5e-4_seed_${i}_batch_1024 --seed $i --optim adam --lr 1e-4 --batch-size 1024;
done

