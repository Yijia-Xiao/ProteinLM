# rm -r ckpt/hidden-32-layer-2-head-2-bs-4-mp-1/*
rm -r ckpt/debug/*
MSA=msa4
DEPTH=32
LENGTH=128
python /dataset/f0a0efb9/protein/msa/protein-msa/pretrain/pretrain_msa.py \
  --num-layers 2 \
  --hidden-size 128 \
  --num-attention-heads 2 \
  --micro-batch-size 1 \
  --global-batch-size 2 \
  --seq-length $(($DEPTH*$LENGTH)) \
  --max-position-embeddings $(($DEPTH*$LENGTH)) \
  --train-iters 100 \
  --load /dataset/f0a0efb9/protein/msa/protein-msa/pretrain/ckpt/debug \
  --save /dataset/f0a0efb9/protein/msa/protein-msa/pretrain/ckpt/debug \
  --data-path /workspace/msa/${MSA}/${MSA}_text_document \
  --vocab-file /dataset/f0a0efb9/protein/msa/protein-msa/pretrain/msa_tools/iupac_vocab.txt \
  --data-impl mmap \
  --distributed-backend nccl \
  --lr 0.00005 \
  --lr-decay-style linear \
  --weight-decay 1e-2 \
  --clip-grad 1.0 \
  --lr-warmup-fraction .01 \
  --log-interval 1 \
  --save-interval 10 \
  --eval-interval 5 \
  --eval-iters 10 \
  --tensor-model-parallel-size 1 \
  --fp16 \
  --msa-depth 32 \
  --msa-length 512
