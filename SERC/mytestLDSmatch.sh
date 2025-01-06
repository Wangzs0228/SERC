# python main0313.py --model hamida --dataset PaviaU    --target_dataset PaviaC    --patch_size 7  --class_match 0  --seed 0 \
# --epoch 100 --training_times 200 --cuda 0 --mmd 0 --na 0.5  --ratio_ord 0.4 --group 12 \
# --mcc 1 --dann 0 --cdan 0 --lr 0.02 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname falseLDS_

# python main0313.py --model hamida --dataset PaviaU    --target_dataset PaviaC    --patch_size 7  --class_match 1  --seed 0 \
# --epoch 100 --training_times 200 --cuda 0 --mmd 0 --na 0.5  --ratio_ord 0.4 --group 12 \
# --mcc 1 --dann 0 --cdan 0 --lr 0.02 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname falseLDS_
python main0313.py --model hamida --dataset Houston13 --target_dataset Houston18 --patch_size 7 --class_match 0  --seed 0 \
--epoch 100 --training_times 200 --cuda 0 --mmd 0 --na 0.5    --ratio_ord 0.4 --group 12 \
--mcc 1 --dann 0 --cdan 0 --lr 0.02 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname falseLDS_
python main0313.py --model hamida --dataset Houston13 --target_dataset Houston18 --patch_size 7 --class_match 1  --seed 0 \
--epoch 100 --training_times 200 --cuda 0 --mmd 0 --na 0.5    --ratio_ord 0.4 --group 12 \
--mcc 1 --dann 0 --cdan 0 --lr 0.02 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname falseLDS_
# python main0313.py --model hamida --dataset Hangzhou  --target_dataset Shanghai  --patch_size 7  --class_match 0 --seed 0 \
# --epoch 100 --training_times 200 --cuda 0 --mmd 0 --na 0.5    --ratio_ord 0.4 --group 4 \
# --mcc 1 --dann 0 --cdan 0 --lr 0.02 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation  --visname falseLDS_

