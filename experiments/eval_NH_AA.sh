if [ "$#" -ne 4 ]; then
    echo "There are 4 parameters: run $0 GPU OUTPUT_PATH EPS LEVEL"
    exit
fi

# evaluation
taskset -c $(( 9*$1 ))-$(( 8+9*$1 ))   python3 ../scripts/validation_warmup.py \
    --arch resnet18 \
    --loss cross-entropy \
    --lr 1e-5 \
    --dropout 0.5 \
    --data inaturalist19-224 \
    --workers 4 \
    --data-paths-config ../data_paths.yml \
    --output $2 \
    --num_training_steps 200000 \
    --gpu $1 \
    --val_freq 1 \
    --attack-eps $3 \
    --attack-step 1 \
    --evaluate NHAA \
    --hPGD extra_max \
    --hPGD-level $4
