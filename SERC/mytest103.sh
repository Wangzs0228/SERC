python main0103.py --model hamida --dataset PaviaU    --target_dataset PaviaC    --patch_size 7 \
--epoch 100 --training_times 200 --cuda 0 --mmd 0 --na 0.035 --sat 0.005 --saf 0.001 --ratio_ord 0.3 --group 12 --mcc 1 --dann 0 --cdan 0 --lr 0.01 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation
python main0103.py --model hamida --dataset Houston13 --target_dataset Houston18 --patch_size 7 --seed 0 \
--epoch 100 --training_times 200 --cuda 0 --mmd 0 --na 0.7 --sat 0.005 --saf 0.001 --ratio_ord 0.3 --group 12 --mcc 1 --dann 0 --cdan 0 --lr 0.01 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation
python main0103.py --model hamida --dataset Dioni     --target_dataset Loukia    --patch_size 7 \
--epoch 100 --training_times 200 --cuda 0 --mmd 0 --na 0.2 --sat 0.005 --saf 0.001 --ratio_ord 0.3 --group 4 --mcc 1 --dann 0 --cdan 0 --lr 0.01 --batch_size 100 --runs 1 --flip_augmentation --radiation_augmentation 
python main0103.py --model hamida --dataset Hangzhou  --target_dataset Shanghai  --patch_size 7 \
--epoch 100 --training_times 200 --cuda 0 --mmd 0 --na 0.1 --sat 0.000 --saf 0.000 --ratio_ord 0.3 --group 4 --mcc 1 --dann 0 --cdan 0 --lr 0.001 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation  