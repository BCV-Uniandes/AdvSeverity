if [ "$#" -ne 3 ]; then
    echo "There are 3 parameters: run $0 GPU OUTPUT_PATH EPS"
    exit
fi

echo "Parameters used:"
echo "  GPU: $1"
echo "  OUTPUT_PATH: $2"
echo "  EPS: $3"

# training
taskset -c $(( 9*$1 ))-$(( 8+9*$1 )) python3 ../scripts/start_training.py \
    --arch resnet18 \
    --loss cross-entropy \
    --lr 1e-5 \
    --dropout 0.5 \
    --data inaturalist19-224 \
    --workers 9 \
    --data-paths-config ../data_paths.yml \
    --output $2 \
    --num_training_steps 200000 \
    --gpu $1 \
    --val_freq 1 \
    --evaluate none

./eval_PGD.sh $1 $2 $3
./eval_hPGD.sh $1 $2 $3