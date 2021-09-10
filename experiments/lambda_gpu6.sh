OUTPUT_PATH=/home/gjeanneret/RESULTS/newRH/curr/eps8_iter8_step6/
# OUTPUT_PATH=/home/gjeanneret/RESULTS/AdvSeverity/CURR/eps8_iter8_step6/

# evaluation
taskset -c 70-75 python3 ../scripts/run_h_attacks.py \
    --arch resnet18 \
    --loss cross-entropy \
    --lr 1e-5 \
    --dropout 0.5 \
    --data inaturalist19-224 \
    --workers 5 \
    --data-paths-config ../data_paths.yml \
    --output $OUTPUT_PATH \
    --num_training_steps 200000 \
    --gpu 6 \
    --val_freq 1 \
    --attack-iter 50 \
    --attack-step 1 \
    --attack-eps 8 \
    --hPGD-level 3 \
    --chunks 5 \
    --chunk 0
