for i in {0.2,0.02,0.002,0.0002}
do
    echo $i 
    python main0313.py --model hamida --dataset PaviaU    --target_dataset PaviaC    --patch_size 7 \
    --epoch 50 --training_times 200 --cuda 0 --mmd 0 --ratio_ord 0.4 --na 0.5 --group 12  --sample_ratio 1\
    --mcc 1 --dann 0 --cdan 0 --lr $i --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname lr_$i
    python main0313.py --model hamida --dataset Houston13 --target_dataset Houston18 --patch_size 7 \
    --epoch 50 --training_times 200 --cuda 0 --mmd 0 --ratio_ord 0.4 --na 0.5  --group 12 --sample_ratio 1\
    --mcc 1 --dann 0 --cdan 0 --lr $i --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname lr_$i
    done
 
for i in {0.2,0.02,0.002,0.0002}
do
    echo $i 
    python main0313.py --model hamida --dataset Hangzhou  --target_dataset Shanghai  --patch_size 7 \
    --epoch 50 --training_times 200 --cuda 0 --mmd 0 --na 0.5 --ratio_ord 0.4 --group 4 --mcc 1 --dann 0 \
    --cdan 0 --lr $i --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation  --visname lr_$i
    done 



# python main0227.py --model hamida --dataset PaviaU    --target_dataset PaviaC    --patch_size 7 --epoch 50 \
# --training_times 200 --cuda 0 --mmd 0 --na 0.035 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 12 --mcc 1 --dann 0 \
# --cdan 0 --lr 0.002 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname na0.002

# python main0227.py --model hamida --dataset PaviaU    --target_dataset PaviaC    --patch_size 7 --epoch 50 \
# --training_times 200 --cuda 0 --mmd 0 --na 0.035 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 12 --mcc 1 --dann 0 \
# --cdan 0 --lr 0.001 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname na0.001

# python main0227.py --model hamida --dataset PaviaU    --target_dataset PaviaC    --patch_size 7 --epoch 50 \
# --training_times 200 --cuda 0 --mmd 0 --na 0.035 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 12 --mcc 1 --dann 0 \
# --cdan 0 --lr 0.02 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname na0.02

# python main0227.py --model hamida --dataset PaviaU    --target_dataset PaviaC    --patch_size 7 --epoch 50 \
# --training_times 200 --cuda 0 --mmd 0 --na 0.035 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 12 --mcc 1 --dann 0 \
# --cdan 0 --lr 0.2 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname na0.2

# python main0227.py --model hamida --dataset PaviaU    --target_dataset PaviaC    --patch_size 7 --epoch 50 \
# --training_times 200 --cuda 0 --mmd 0 --na 0.035 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 12 --mcc 1 --dann 0 \
# --cdan 0 --lr 0.1 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname na0.1


# python main0227.py --model hamida --dataset Houston13 --target_dataset Houston18 --patch_size 7 --seed 0 \
# --epoch 50 --training_times 200 --cuda 0 --mmd 0 --na 0.7 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 12 --mcc 1 \
# --dann 0 --cdan 0 --lr 0.002 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname na0.002

# python main0227.py --model hamida --dataset Houston13 --target_dataset Houston18 --patch_size 7 --seed 0 \
# --epoch 50 --training_times 200 --cuda 0 --mmd 0 --na 0.7 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 12 --mcc 1 \
# --dann 0 --cdan 0 --lr 0.001 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname na0.001

# python main0227.py --model hamida --dataset Houston13 --target_dataset Houston18 --patch_size 7 --seed 0 \
# --epoch 50 --training_times 200 --cuda 0 --mmd 0 --na 0.7 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 12 --mcc 1 \
# --dann 0 --cdan 0 --lr 0.01 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname na0.01

# python main0227.py --model hamida --dataset Houston13 --target_dataset Houston18 --patch_size 7 --seed 0 \
# --epoch 50 --training_times 200 --cuda 0 --mmd 0 --na 0.7 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 12 --mcc 1 \
# --dann 0 --cdan 0 --lr 0.2 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname na0.2

# python main0227.py --model hamida --dataset Houston13 --target_dataset Houston18 --patch_size 7 --seed 0 \
# --epoch 50 --training_times 200 --cuda 0 --mmd 0 --na 0.7 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 12 --mcc 1 \
# --dann 0 --cdan 0 --lr 0.1 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname na0.1


# python main0227.py --model hamida --dataset Hangzhou  --target_dataset Shanghai  --patch_size 7 --epoch 50 \
# --training_times 200 --cuda 0 --mmd 0 --na 0.1 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 4 --mcc 1 --dann 0 \
# --cdan 0 --lr 0.002 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation  --visname na0.002

# python main0227.py --model hamida --dataset Hangzhou  --target_dataset Shanghai  --patch_size 7 --epoch 50 \
# --training_times 200 --cuda 0 --mmd 0 --na 0.1 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 4 --mcc 1 --dann 0 \
# --cdan 0 --lr 0.001 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation  --visname na0.001

# python main0227.py --model hamida --dataset Hangzhou  --target_dataset Shanghai  --patch_size 7 --epoch 50 \
# --training_times 200 --cuda 0 --mmd 0 --na 0.1 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 4 --mcc 1 --dann 0 \
# --cdan 0 --lr 0.02 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation  --visname na0.02

# python main0227.py --model hamida --dataset Hangzhou  --target_dataset Shanghai  --patch_size 7 --epoch 50 \
# --training_times 200 --cuda 0 --mmd 0 --na 0.1 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 4 --mcc 1 --dann 0 \
# --cdan 0 --lr 0.2 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation  --visname na0.2

# python main0227.py --model hamida --dataset Hangzhou  --target_dataset Shanghai  --patch_size 7 --epoch 50 \
# --training_times 200 --cuda 0 --mmd 0 --na 0.1 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 4 --mcc 1 --dann 0 \
# --cdan 0 --lr 0.1 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation  --visname na0.1






# python main0227.py --model hamida --dataset PaviaU    --target_dataset PaviaC    --patch_size 7 \
# --epoch 50 --training_times 200 --cuda 0 --mmd 0 --na 0 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 12 --mcc 1 --dann 0 --cdan 0 --lr 0.01 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation
# python main0227.py --model hamida --dataset Houston13 --target_dataset Houston18 --patch_size 7 --seed 0 \
# --epoch 50 --training_times 200 --cuda 0 --mmd 0 --na 0 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 12 --mcc 1 --dann 0 --cdan 0 --lr 0.02 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation
# python main0227.py --model hamida --dataset Hangzhou  --target_dataset Shanghai  --patch_size 7 \
# --epoch 50 --training_times 200 --cuda 0 --mmd 0 --na 0 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 4 --mcc 1 \
# --dann 0 --cdan 0 --lr 0.01 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation  --visname raw
# python main0227.py --model hamida --dataset Shanghai  --target_dataset Hangzhou  --patch_size 7 \
# --epoch 50 --training_times 200 --cuda 0 --mmd 0 --na 0 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 4 --mcc 1 \
# --dann 0 --cdan 0 --lr 0.01 --batch_size 100 --runs 1 --flip_augmentation --radiation_augmentation  --visname raw
# python main0227.py --model hamida --dataset Shanghai  --target_dataset Hangzhou  --patch_size 7 \
# --epoch 50 --training_times 200 --cuda 0 --mmd 0 --na 0.1 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 4 --mcc 1 \
# --dann 0 --cdan 0 --lr 0.01 --batch_size 100 --runs 1 --flip_augmentation --radiation_augmentation  --visname na










#save
# python main0227.py --model hamida --dataset PaviaU    --target_dataset PaviaC    --patch_size 7 \
# --epoch 50 --training_times 200 --cuda 0 --mmd 0 --na 0.035 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 12 \
# --mcc 1 --dann 0 --cdan 0 --lr 0.01 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation
# python main0227.py --model hamida --dataset Houston13 --target_dataset Houston18 --patch_size 7 --seed 0 \
# --epoch 50 --training_times 200 --cuda 0 --mmd 0 --na 0.7 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 12 \
# --mcc 1 --dann 0 --cdan 0 --lr 0.02 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation
# python main0227.py --model hamida --dataset Hangzhou  --target_dataset Shanghai  --patch_size 7 \
# --epoch 50 --training_times 200 --cuda 0 --mmd 0 --na 0.1 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 4 \
# --mcc 1 --dann 0 --cdan 0 --lr 0.01 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation  