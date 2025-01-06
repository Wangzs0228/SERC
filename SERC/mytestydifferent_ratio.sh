for i in {0.1,0.3,0.5,0.7,}
do
    echo $i 
    python main0313.py --model hamida --dataset PaviaU    --target_dataset PaviaC    --patch_size 7 \
    --epoch 50 --training_times 200 --cuda 0 --mmd 0 --ratio_ord 0.4 --na 0.5 --group 12  --sample_ratio $i\
    --mcc 1 --dann 0 --cdan 0 --lr 0.01 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname sample_ratio$i
    python main0313.py --model hamida --dataset Houston13 --target_dataset Houston18 --patch_size 7 \
    --epoch 100 --training_times 200 --cuda 0 --mmd 0 --ratio_ord 0.4 --na 0.5  --group 12 --sample_ratio $i\
    --mcc 1 --dann 0 --cdan 0 --lr 0.02 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname sample_ratio_$i
    done
 
 