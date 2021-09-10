if [ "$#" -ne 3 ]; then
    echo "There are 3 parameters: run $0 GPU OUTPUT_PATH EPS"
    exit
fi

ITER=50

# evaluation
taskset -c $(( 7*$1 ))-$(( 6+7*$1 ))  python3 ../scripts/start_training.py \
    --arch resnet18 \
    --loss cross-entropy \
    --lr 1e-5 \
    --dropout 0.5 \
    --data inaturalist19-224 \
    --workers 6 \
    --data-paths-config ../data_paths.yml \
    --output $2 \
    --num_training_steps 200000 \
    --gpu $1 \
    --val_freq 1 \
    --attack-eps $3 \
    --attack-iter $ITER \
    --attack-step 1 \
    --evaluate PGD-u \
    --test-set
