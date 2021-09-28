if [ "$#" -ne 5 ]; then
    echo "There are 5 parameters: run $0 GPU EPSILON ITERATIONS STEP TYPE"
    exit
fi

echo "Parameters used:"
echo "  GPU: $1"
echo "  Epsilon: $2"
echo "  Iterations: $3"
echo "  Step: $4"
echo "  Type: $5 (CURR or XE)"

OUTPUT_PATH=/home/gjeanneret/RESULTS/AdvSeverity/$5/eps$2_iter$3_step$4/

# evaluation
taskset -c $(( 7*$1 ))-$(( 6+7*$1 )) python3 ../scripts/start_training.py \
    --arch resnet18 \
    --loss cross-entropy \
    --lr 1e-5 \
    --dropout 0.5 \
    --data inaturalist19-224 \
    --workers 6 \
    --data-paths-config ../data_paths.yml \
    --output $OUTPUT_PATH \
    --num_training_steps 200000 \
    --gpu $1 \
    --val_freq 1 \
    --evaluate none \
    --test-set

./test_PGD.sh $1 $OUTPUT_PATH $2
./test_hPGD.sh $1 $OUTPUT_PATH $2