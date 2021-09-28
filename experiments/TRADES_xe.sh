if [ "$#" -ne 5 ]; then
    echo "There are 5 parameters: run $0 GPU EPSILON ITERATIONS STEP TRADESBETA"
    exit
fi

echo "Parameters used:"
echo "  GPU: $1"
echo "  Epsilon: $2"
echo "  Iterations: $3"
echo "  Step: $4"
echo "  Beta: $5"

OUTPUT_PATH=/home/gjeanneret/RESULTS/RH/TRADES/eps$2_iter$3_step$4_beta$5/

echo "output path: ${OUTPUT_PATH}"

# training
taskset -c $(( 9*$1 ))-$(( 8+9*$1 )) python3 ../scripts/validation_warmup.py \
    --arch resnet18 \
    --loss cross-entropy \
    --lr 1e-5 \
    --dropout 0.5 \
    --data inaturalist19-224 \
    --workers 9 \
    --data-paths-config ../data_paths.yml \
    --output $OUTPUT_PATH \
    --num_training_steps 200000 \
    --gpu $1 \
    --val_freq 1 \
    --attack-eps $2 \
    --attack-iter $3 \
    --attack-step $4 \
    --trades-beta $5 \
    --attack trades
