set -xe

# rm msa_text_document.*
NAME=tiny
PFAM=./
VOCAB=./

python ../tools/preprocess_data.py --input $PFAM/$NAME.json \
	--tokenizer-type BertWordPieceCase --vocab-file $VOCAB/iupac_vocab.txt \
	--output-prefix $PFAM/$NAME --dataset-impl mmap --workers 64
