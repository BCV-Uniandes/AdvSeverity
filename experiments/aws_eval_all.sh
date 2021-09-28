if [ "$#" -ne 5 ]; then
    echo "There are 5 parameters: run $0 GPU EPSILON ITERATIONS STEP TYPE"
    exit
fi

echo "Parameters used:"
echo "  GPU: $1"
echo "  Epsilon: $2"
echo "  Iterations: $3"
echo "  Step: $4"
echo "  Type: $5 (free or curr)"

OUTPUT_PATH=/data/RH/$5/eps$2_iter$3_step$4/

# evaluation
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

./eval_PGD.sh $1 $OUTPUT_PATH $2
./eval_hPGD.sh $1 $OUTPUT_PATH $2