# # !/bin/bash
# for i in {4,8,12,16}
# do
#     for j in {0.2,0.4,0.6,0.8}
#     do
#         echo $i $j
#         python main0227.py --model hamida --dataset PaviaU    --target_dataset PaviaC    --patch_size 7 \
#         --epoch 50 --training_times 200 --cuda 0 --mmd 0 --ratio_ord $j --na 0.5 --group $i \
#         --mcc 1 --dann 0 --cdan 0 --lr 0.02 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname na_y0$i\2y1$j
#     done
# done
for i in {4,8,12,16}
do
    for j in {0.2,0.4,0.6,0.8}
    do
        echo $i $j
        python main0227.py --model hamida --dataset Houston13 --target_dataset Houston18 --patch_size 7 --seed 0 \
        --epoch 50 --training_times 200 --cuda 0 --mmd 0 --ratio_ord $j --na 0.5 --group $i \
        --mcc 1 --dann 0 --cdan 0 --lr 0.02 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname na_y0$i\2y1$j
    done
done
for i in {4,8,12,16}
do
    for j in {0.2,0.4,0.6,0.8}
    do
        echo $i $j
        python main0313.py --model hamida --dataset Hangzhou  --target_dataset Shanghai  --patch_size 7 \
        --epoch 50 --training_times 200 --cuda 0 --mmd 0 --ratio_ord $j --na 0.5 --group $i \
        --mcc 1 --dann 0 --cdan 0 --lr 0.02 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname na_y0$i\2y1$j
    done
done
 