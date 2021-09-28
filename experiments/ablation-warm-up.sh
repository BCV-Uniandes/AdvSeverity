EPS=4
ITER=8
STEP=2

echo "Parameters used:"
echo "  GPU: $1"

OUTPUT_PATH=/data/RH-ablation/warm-up

echo "output path: ${OUTPUT_PATH}"

# training
taskset -c $(( 8*$1 ))-$(( 6+8*$1 )) python3 ../scripts/validation_warmup.py \
    --arch resnet18 \
    --loss cross-entropy \
    --lr 1e-5 \
    --dropout 0.5 \
    --data inaturalist19-224 \
    --workers 5 \
    --data-paths-config ../data_paths.yml \
    --output $OUTPUT_PATH \
    --num_training_steps 200000 \
    --gpu $1 \
    --val_freq 1 \
    --attack-eps $EPS \
    --attack-iter $ITER \
    --attack-step $STEP \
    --attack free \
    --curriculum-training \
    --seed 1


taskset -c $(( 8*$1 ))-$(( 6+8*$1 )) python3 ../scripts/start_training.py \
    --arch resnet18 \
    --loss cross-entropy \
    --lr 1e-5 \
    --dropout 0.5 \
    --data inaturalist19-224 \
    --workers 5 \
    --data-paths-config ../data_paths.yml \
    --output $OUTPUT_PATH \
    --num_training_steps 200000 \
    --gpu $1 \
    --val_freq 1 \
    --evaluate none

./eval_PGD.sh $1 $OUTPUT_PATH $EPS
./eval_hPGD.sh $1 $OUTPUT_PATH $EPS
