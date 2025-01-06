for j in pavia;
do
    for i in dan dann dsan mcd;
    do
        echo $j $i 
        python train/$i/train.py --config configs/$j/$i/$i.yaml --path ./runs/$j/$i --nodes 1 --gpus 1 --rank-node 0 --backend gloo \
        --master-ip localhost --master-port 8888 --seed 0
    done
done

# for j in houston pavia shanghang;
# do
#     for i in dan dann dsan mcd;
#     do
#         echo $j $i 
#         python train/$i/train.py --config configs/$j/$i/$i.yaml --path ./runs/$j/$i --nodes 1 --gpus 1 --rank-node 0 --backend gloo \
#         --master-ip localhost --master-port 8888 --seed 0
#     done
# done

# python train/ddc/train.py configs/houston/dan/dan.yaml --path ./runs/houston/dan --nodes 1 --gpus 1 --rank-node 0 --backend gloo \
#         --master-ip localhost --master-port 8886 --seed 0 --opt-level O2

# python train/ddc/train.py configs/pavia/dan/dan.yaml --path ./runs/pavia/dan --nodes 1 --gpus 1 --rank-node 0 --backend gloo \
#         --master-ip localhost --master-port 8890 --seed 0 --opt-level O2

# python train/ddc/train.py configs/shanghang/dan/dan.yaml --path ./runs/shanghang/dan --nodes 1 --gpus 1 --rank-node 0 \
#         --backend gloo --master-ip localhost --master-port 8890 --seed 0 --opt-level O2


# python train/dann/train.py configs/houston/dann/dann.yaml --path ./runs/houston/dann --nodes 1 --gpus 1 --rank-node 0 --backend gloo \
#         --master-ip localhost --master-port 8890 --seed 0 --opt-level O2

# python train/dann/train.py configs/pavia/dann/dann.yaml --path ./runs/pavia/dann-train --nodes 1 --gpus 1 ^
#         --rank-node 0 ^
#         --backend gloo ^
#         --master-ip localhost ^
#         --master-port 8890 ^
#         --seed 0 ^
#         --opt-level O2