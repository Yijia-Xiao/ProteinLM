NAME=UniRef50-xa-a2m-2017-large

mkdir -p /workspace/msa/$NAME
python prepare_msa.py /dataset/f0a0efb9/protein/data/UniRef50-xa-a2m-2017 > /workspace/msa/$NAME/$NAME.json
# python prepare_msa.py /dataset/f0a0efb9/protein/data/$NAME > $NAME.json

#PFAM=./
#VOCAB=./

python ../tools/preprocess_data.py --input /workspace/msa/$NAME/$NAME.json \
	--tokenizer-type BertWordPieceCase --vocab-file ./iupac_vocab.txt \
	--output-prefix /workspace/msa/$NAME/$NAME --dataset-impl mmap --workers 96

