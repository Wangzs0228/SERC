# for i in {0.2,0.4,0.6,0.8,1.0,}
for i in {0,0.1,0.3,0.5,0.7,0.9,1.1}
do
    echo $i 
    python main0313.py --model hamida --dataset PaviaU    --target_dataset PaviaC    --patch_size 7 --seed 0 \
    --epoch 100 --training_times 200 --cuda 0 --mmd 0 --ratio_ord 0.3 --na $i --group 12 \
    --mcc 1 --dann 0 --cdan 0 --lr 0.02 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname na_y2$i
    python main0313.py --model hamida --dataset Houston13 --target_dataset Houston18 --patch_size 7 --seed 0 \
    --epoch 100 --training_times 200 --cuda 0 --mmd 0 --ratio_ord 0.3 --na $i --group 12 \
    --mcc 1 --dann 0 --cdan 0 --lr 0.02 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname na_y2$i
    # python main0313.py --model hamida --dataset Hangzhou  --target_dataset Shanghai  --patch_size 7 --seed 0 \
    # --epoch 50 --training_times 200 --cuda 0 --mmd 0 --ratio_ord 0.4 --na $i --group 4 \
    # --mcc 1 --dann 0 --cdan 0 --lr 0.01 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname na_y2$i
done
 