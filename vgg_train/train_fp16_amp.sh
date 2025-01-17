export CUDA_VISIBLE_DEVICES=3
export WANDB_API_KEY=9c912b26625867fd8f46d2d9cda32aaa4bd392ae
MODEL=VGG19
DTYPE=fp16
EPOCHS=200
LEARNING_RATE=0.01
AMP=1
LOSS_SCALE=0

# 定义BATCH_SIZE数组
BATCH_SIZES=(256 128 64 32 16 8)

# 遍历不同的BATCH_SIZE值
for BATCH_SIZE in "${BATCH_SIZES[@]}"
do
    # 执行每个命令，等待前一个命令执行完毕
    nohup python main.py \
        --dtype $DTYPE \
        --model $MODEL \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LEARNING_RATE \
        --amp > results/output_log/Cifar10-$MODEL-e$EPOCHS-b$BATCH_SIZE-$DTYPE-amp$AMP-scale$LOSS_SCALE.log 2>&1
    # 等待命令执行完毕后继续下一个BATCH_SIZE
    wait
done