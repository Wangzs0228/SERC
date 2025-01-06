# python main1230.py --model hamida --dataset PaviaU    --target_dataset PaviaC    --patch_size 7 \
# --epoch 50 --cuda 0 --mmd 0 --mcc 1 --dann 0 --cdan 0 --lr 0.01 --batch_size 100 --runs 3 --seed 0 --flip_augmentation --radiation_augmentation
# python main1230.py --model hamida --dataset Houston13 --target_dataset Houston18 --patch_size 7 --seed 0 \
# --epoch 50 --cuda 0 --mmd 0 --mcc 1 --dann 0 --cdan 0 --lr 0.01 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation 
python main1230.py --model hamida --dataset Hangzhou  --target_dataset Shanghai  --patch_size 7 \
--epoch 50 --cuda 0 --mmd 0 --mcc 1 --dann 0 --cdan 0 --lr 0.01 --batch_size 100 --runs 3  --flip_augmentation --radiation_augmentation 
python main1230.py --model hamida --dataset Dioni     --target_dataset Loukia    --patch_size 7 \
--epoch 50 --cuda 0 --mmd 0 --mcc 1 --dann 0 --cdan 0 --lr 0.01 --batch_size 100 --runs 3 --flip_augmentation --radiation_augmentation 
# python main1230.py --model vit --dataset PaviaU --target_dataset PaviaC --patch_size 15 --epoch 50 --cuda 0
# python main1230.py --model vit --dataset Houston13 --target_dataset Houston18 --patch_size 15 --epoch 50 --cuda 0
# python main1230.py --model vit --dataset Hangzhou --target_dataset Shanghai --patch_size 15 --epoch 50 --cuda 0
# python main1230.py --model vit --dataset Dioni --target_dataset Loukia --patch_size 15 --epoch 50 --cuda 0
