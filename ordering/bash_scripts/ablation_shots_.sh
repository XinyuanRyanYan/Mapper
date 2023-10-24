DEVICE=3
SAMPLING_STRATEGY=uniform
DATASET=trec
RUNNING_MODE=kl_prompting
MODEL_NAME="/home/sci/zhichao.xu/models/llama-7b"

for NUM_SHOTS in 9 10 11 12
do
    CUDA_VISIBLE_DEVICES=$DEVICE python run_calibration.py \
    --model_name_or_path $MODEL_NAME \
    --dataset $DATASET \
    --loading_mode int4 \
    --train_samples 1024 \
    --logging ./loggings/ablation_shots_.log \
    --num_shots $NUM_SHOTS \
    --sampling_strategy $SAMPLING_STRATEGY \
    --running_mode kl_prompting
done
