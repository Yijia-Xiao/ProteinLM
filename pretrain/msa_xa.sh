#!/bin/bash

set -x

#echo HIDDENSIZE LAYERNUM HEAD BATCHSIZE
HEAD=16
if [ $model == 1 ]; then
	HIDDENSIZE=1024
	LAYERNUM=8
elif [ $model == 2 ]; then
	HIDDENSIZE=768
	LAYERNUM=12
elif [ $model == -1 ]; then
	echo "default model"
	HIDDENSIZE=32
	LAYERNUM=1
fi

if [ $msa == 1 ]; then
        DEPTH=64
        LENGTH=1024
elif [ $msa == 2 ]; then
        DEPTH=128
        LENGTH=512
elif [ $msa == 3 ]; then
        DEPTH=128
        LENGTH=768
elif [ $msa == 4 ]; then
        DEPTH=256
        LENGTH=256
elif [ $msa == 5 ]; then
        DEPTH=128
        LENGTH=256
elif [ $msa == 6 ]; then
        DEPTH=64
        LENGTH=384
elif [ $msa == -1 ]; then
	echo "default msa"
        DEPTH=16
        LENGTH=256
fi

#if [ $model == 1 ]; then
#	HIDDENSIZE=1024
#	LAYERNUM=16
#elif [ $model == 2 ]; then
#	HIDDENSIZE=512
#	LAYERNUM=24
#elif [ $model == 3 ]; then
#	HIDDENSIZE=1024
#	LAYERNUM=8
#elif [ $model == 4 ]; then
#	HIDDENSIZE=768
#	LAYERNUM=8
#elif [ $model == 5 ]; then
#	HIDDENSIZE=512
#	LAYERNUM=8
#	HEAD=8
#elif [ $model == -1 ]; then
#	echo "default model"
#	HIDDENSIZE=32
#	LAYERNUM=1
#fi
#
#
#
#if [ $msa == 1 ]; then
#        DEPTH=16
#        LENGTH=1536
#elif [ $msa == 2 ]; then
#        DEPTH=32
#        LENGTH=1024
#elif [ $msa == 3 ]; then
#        DEPTH=64
#        LENGTH=512
#elif [ $msa == 4 ]; then
#        DEPTH=128
#        LENGTH=256
#elif [ $msa == 5 ]; then
#        DEPTH=128
#        LENGTH=1024
#elif [ $msa == 6 ]; then
#        DEPTH=256
#        LENGTH=1024
#elif [ $msa == 7 ]; then
#        DEPTH=384
#        LENGTH=768
#elif [ $msa == 8 ]; then
#        DEPTH=384
#        LENGTH=1024
#elif [ $msa == 9 ]; then
#        DEPTH=512
#        LENGTH=768
#elif [ $msa == 10 ]; then
#        DEPTH=16
#        LENGTH=256
#elif [ $msa == 11 ]; then
#        DEPTH=128
#        LENGTH=512
#elif [ $msa == -1 ]; then
#	echo "default msa"
#        DEPTH=16
#        LENGTH=256
#fi



# HIDDENSIZE=1024
# LAYERNUM=16

HEAD=16
BATCHSIZE=1
MSA=UniRef50-xa-a2m-2017

# DEPTH=64
# LENGTH=1024

if [ -z $MP ]
then
	MP=8
fi

NAME=trash-hidden-$HIDDENSIZE-layer-$LAYERNUM-head-$HEAD-bs-$BATCHSIZE-mp-$MP-depth-$DEPTH-length-$LENGTH-xa-shuffle-expand-changemask-cls
MYPATH=$PWD

GPUS_PER_NODE=8
NNODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=7008
:<<!
case $BATCHSIZE in
	16) ITER=4;;
	8) ITER=6;;
	4) ITER=8;;
	*) echo -e "not found";;
esac
!
ITER=100000

DATA_PATH=/workspace/uniref/UniRef50-xa-a2m-2017_text_document

CHECKPOINT_PATH=/workspace/ckpt/$NAME
mkdir -p $CHECKPOINT_PATH
mkdir -p $MYPATH/tb/$NAME

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

ATTN=/workspace/attn/$NAME
mkdir -p $ATTN
#echo 0 > $ATTN/idx

(python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       $MYPATH/pretrain_msa.py \
       --num-layers $LAYERNUM \
       --hidden-size $HIDDENSIZE \
       --num-attention-heads $HEAD \
       --micro-batch-size $BATCHSIZE \
       --global-batch-size $(($BATCHSIZE*$WORLD_SIZE)) \
       --seq-length $(($DEPTH*$LENGTH)) \
       --max-position-embeddings $(($DEPTH*$LENGTH)) \
       --train-iters $ITER \
       --save $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file /dataset/f0a0efb9/protein/msa/protein-msa/pretrain/msa_tools/iupac_vocab.txt \
       --data-impl mmap \
       --distributed-backend nccl \
       --lr 0.00005 \
       --lr-decay-style linear \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 5 \
       --save-interval 10000 \
       --eval-interval 100 \
       --eval-iters 10 \
       --tensor-model-parallel-size $MP \
       --msa-depth $DEPTH \
       --msa-length $LENGTH \
       --msa-shuffle 1 \
       --fp16 \
       --tensorboard-dir $MYPATH/tb/0814/$NAME/ \
       --compute-attn-weights \
       --attention-save $ATTN \
       --attn-save-freq 2000 \
) |& tee -a $MYPATH/logs/0814/log-$NAME
#       --checkpoint-activations \
#       --fp16 \
#       --compute-attn-weights \

#       --load $CHECKPOINT_PATH \
#       $(($DEPTH*$LENGTH))
#       --checkpoint-activations \
# --distribute-checkpointed-activations
