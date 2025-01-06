# python main0227.py --model hamida --dataset PaviaU    --target_dataset PaviaC    --patch_size 7 \
# --epoch 50 --training_times 200 --cuda 0 --mmd 0 --na 0.035 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 12 \
# --mcc 1 --dann 0 --cdan 0 --lr 0.01 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname na
python main0227.py --model hamida --dataset Houston13 --target_dataset Houston18 --patch_size 7 --seed 0 \
--epoch 50 --training_times 200 --cuda 0 --mmd 0 --na 0.7 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 12 \
--mcc 1 --dann 0 --cdan 0 --lr 0.02 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname na
# python main0227.py --model hamida --dataset Hangzhou  --target_dataset Shanghai  --patch_size 7 \
# --epoch 50 --training_times 200 --cuda 0 --mmd 0 --na 0.1 --sat 1 --saf 0.001 --ratio_ord 0.3 --group 4 \
# --mcc 1 --dann 0 --cdan 0 --lr 0.01 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation  --visname na