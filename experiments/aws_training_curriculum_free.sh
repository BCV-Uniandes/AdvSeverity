if [ "$#" -ne 4 ]; then
    echo "There are 4 parameters: run $0 GPU EPSILON ITERATIONS STEP"
    exit
fi

echo "Parameters used:"
echo "  GPU: $1"
echo "  Epsilon: $2"
echo "  Iterations: $3"
echo "  Step: $4"

OUTPUT_PATH=/data/RH/curr/eps$2_iter$3_step$4/

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
    --attack-eps $2 \
    --attack-iter $3 \
    --attack-step $4 \
    --attack free \
    --curriculum-training
