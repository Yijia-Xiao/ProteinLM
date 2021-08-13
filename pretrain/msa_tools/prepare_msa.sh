NAME=UniRef50-xa-a2m-2017
python prepare_msa.py /dataset/f0a0efb9/protein/data/$NAME > $NAME.json

cp -rp  $NAME.json /workspace/

exit

PFAM=./
VOCAB=./

mkdir -p /workspace/msa/$NAME
cp ${NAME}.json /workspace/msa/$NAME/

python ../tools/preprocess_data.py --input /workspace/msa/$NAME/$NAME.json \
	--tokenizer-type BertWordPieceCase --vocab-file $VOCAB/iupac_vocab.txt \
	--output-prefix /workspace/msa/$NAME/$NAME --dataset-impl mmap --workers 96



# NAME=tiny
# python prepare_msa.py /dataset/f0a0efb9/protein/msa/data/tiny > $NAME.json


# PFAM=./
# VOCAB=./

# python ../tools/preprocess_data.py --input $PFAM/$NAME.json \
# 	--tokenizer-type BertWordPieceCase --vocab-file $VOCAB/iupac_vocab.txt \
# 	--output-prefix $PFAM/$NAME --dataset-impl mmap --workers 128

# mkdir -p /workspace/msa/$NAME
# cp ${NAME}_text_document.bin /workspace/msa/$NAME/
# cp ${NAME}_text_document.idx /workspace/msa/$NAME/
