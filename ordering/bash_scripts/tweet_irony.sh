DEVICE=2
NUM_SHOTS=8
SAMPLING_STRATEGY=uniform
DATASET=tweet_irony
RUNNING_MODE=baseline

for MODEL_NAME in "/home/sci/zhichao.xu/models/llama-7b" "facebook/opt-13b" "/home/sci/zhichao.xu/models/llama-13b";
do
    CUDA_VISIBLE_DEVICES=$DEVICE python run_calibration.py \
    --model_name_or_path $MODEL_NAME \
    --dataset $DATASET \
    --loading_mode int4 \
    --train_samples 1024 \
    --logging ./loggings/${DATASET}_experiment.log \
    --num_shots $NUM_SHOTS \
    --sampling_strategy $SAMPLING_STRATEGY \
    --running_mode $RUNNING_MODE
done;


# for MODEL_NAME in "gpt2-large" "gpt2-xl" "facebook/opt-1.3b" "facebook/opt-2.7b";
# do
#     CUDA_VISIBLE_DEVICES=$DEVICE python run_calibration.py \
#     --model_name_or_path $MODEL_NAME \
#     --dataset $DATASET \
#     --loading_mode fp16 \
#     --train_samples 1024 \
#     --logging ./loggings/${DATASET}_experiment.log \
#     --num_shots $NUM_SHOTS \
#     --sampling_strategy $SAMPLING_STRATEGY \
#     --running_mode kl_prompting
# done;
