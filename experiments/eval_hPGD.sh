if [ "$#" -ne 3 ]; then
    echo "There are 3 parameters: run $0 GPU OUTPUT_PATH EPS"
    exit
fi

for LEVEL in 1 2 3 4 5 6
do
# evaluation
    taskset -c $(( 9*$1 ))-$(( 8+9*$1 )) python3 ../scripts/start_training.py \
        --arch resnet18 \
        --loss cross-entropy \
        --lr 1e-5 \
        --dropout 0.5 \
        --data inaturalist19-224 \
        --workers 8 \
        --data-paths-config ../data_paths.yml \
        --output $2 \
        --num_training_steps 200000 \
        --gpu $1 \
        --val_freq 1 \
        --attack-eps $3 \
        --attack-iter 50 \
        --attack-step 1 \
        --evaluate hPGD-u \
        --hPGD intra\
        --hPGD-level $LEVEL \
        --save-adversaries
done

for LEVEL in 1 2 3 4 5 6
do
# evaluation
    taskset -c $(( 9*$1 ))-$(( 8+9*$1 )) python3 ../scripts/start_training.py \
        --arch resnet18 \
        --loss cross-entropy \
        --lr 1e-5 \
        --dropout 0.5 \
        --data inaturalist19-224 \
        --workers 8 \
        --data-paths-config ../data_paths.yml \
        --output $2 \
        --num_training_steps 200000 \
        --gpu $1 \
        --val_freq 1 \
        --attack-eps $3 \
        --attack-iter 50 \
        --attack-step 1 \
        --evaluate hPGD-u \
        --hPGD extra_wo_h\
        --hPGD-level $LEVEL \
        --save-adversaries
done


for LEVEL in 1 2 3 4 5 6
do
# evaluation
    taskset -c $(( 9*$1 ))-$(( 8+9*$1 )) python3 ../scripts/start_training.py \
        --arch resnet18 \
        --loss cross-entropy \
        --lr 1e-5 \
        --dropout 0.5 \
        --data inaturalist19-224 \
        --workers 8 \
        --data-paths-config ../data_paths.yml \
        --output $2 \
        --num_training_steps 200000 \
        --gpu $1 \
        --val_freq 1 \
        --attack-eps $3 \
        --attack-iter 50 \
        --attack-step 1 \
        --evaluate hPGD-u \
        --hPGD extra_max\
        --hPGD-level $LEVEL \
        --save-adversaries
done