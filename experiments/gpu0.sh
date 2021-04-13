# ./training_curriculum_free.sh 0 4 6 1
# ./training_curriculum_free.sh 0 4 6 2
# ./training_curriculum_free.sh 0 4 6 4
# ./training_curriculum_free.sh 0 4 6 6

# ./training_curriculum_free.sh 0 4 2 1
# ./training_curriculum_free.sh 0 4 2 2
# ./training_curriculum_free.sh 0 4 2 4
# ./training_curriculum_free.sh 0 4 2 6

# ./training_curriculum_free.sh 0 2 8 1
# ./training_curriculum_free.sh 0 2 8 2

# ./training_curriculum_free.sh 0 2 2 1


# ./eval_all.sh 0 4 6 1 curr
# ./eval_all.sh 0 4 6 2 curr
# ./eval_all.sh 0 4 6 4 curr
# ./eval_all.sh 0 4 6 6 curr

# ./eval_all.sh 0 4 2 1 curr
# ./eval_all.sh 0 4 2 2 curr
# ./eval_all.sh 0 4 2 4 curr
# ./eval_all.sh 0 4 2 6 curr

# ./eval_all.sh 0 2 8 1 curr
# ./eval_all.sh 0 2 8 2 curr

# ./eval_all.sh 0 2 2 1 curr


# ./training_xe_free.sh 0 4 6 1
# ./training_xe_free.sh 0 4 6 4
# ./training_xe_free.sh 0 4 2 1
# ./training_xe_free.sh 0 4 2 4
# ./training_xe_free.sh 0 2 8 1
# ./training_xe_free.sh 0 2 2 1

# ./eval_all.sh 0 4 6 1 free
# ./eval_all.sh 0 4 6 4 free
# ./eval_all.sh 0 4 2 1 free
# ./eval_all.sh 0 4 2 4 free
# ./eval_all.sh 0 2 8 1 free
# ./eval_all.sh 0 2 2 1 free


# ./training_xe_free.sh 0 4 6 2
# ./eval_all.sh 0 4 6 2 free

# OUTPUT_PATH=/data/RH-ablation/xe
# GPU=0

# for ITERS in 10 15 20 25 30 35 40 45
# do
#     for LEVEL in 1 2 3 4 5 6
#     do
#     # evaluation
#         taskset -c $(( 8*$GPU ))-$(( 6+8*$GPU )) python3 ../scripts/start_training.py \
#             --arch resnet18 \
#             --loss cross-entropy \
#             --lr 1e-5 \
#             --dropout 0.5 \
#             --data inaturalist19-224 \
#             --workers 4 \
#             --data-paths-config ../data_paths.yml \
#             --output $OUTPUT_PATH \
#             --num_training_steps 200000 \
#             --gpu $GPU \
#             --val_freq 1 \
#             --attack-eps 4 \
#             --attack-iter $ITERS \
#             --attack-step 1 \
#             --evaluate hPGD-u \
#             --hPGD extra_max\
#             --hPGD-level $LEVEL
#     done
# done

# GPU EPSILON ITERATIONS STEP TYPE
