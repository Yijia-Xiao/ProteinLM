NAME=msa4
python gen_msa.py > $NAME.json


PFAM=./
VOCAB=./

python ./preprocess_data.py --input $PFAM/$NAME.json \
	--tokenizer-type BertWordPieceCase --vocab-file $VOCAB/iupac_vocab.txt \
	--output-prefix $PFAM/$NAME --dataset-impl mmap --workers 128

mkdir -p /workspace/msa/$NAME
cp ${NAME}_text_document.bin /workspace/msa/$NAME/
cp ${NAME}_text_document.idx /workspace/msa/$NAME/
