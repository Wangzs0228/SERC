for i in {2,}
do
    python main_revise_model.py --model myhamida --dataset Houston13 --target_dataset Houston18 --patch_size 7 --seed 0 \
    --epoch 50 --training_times 200 --cuda 0 --mmd 0 --na 0    --ratio_ord 0.3 --group 12 \
    --mcc 1 --dann 0 --cdan 0 --lr 0.02 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname na --layernum $i
    python main_revise_model.py --model myhamida --dataset Houston13 --target_dataset Houston18 --patch_size 7 --seed 0 \
    --epoch 50 --training_times 200 --cuda 0 --mmd 0 --na 0.7    --ratio_ord 0.3 --group 12 \
    --mcc 1 --dann 0 --cdan 0 --lr 0.02 --batch_size 100 --runs 1  --flip_augmentation --radiation_augmentation --visname na --layernum $i
    done


