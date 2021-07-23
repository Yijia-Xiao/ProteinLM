set -xe

rm msa_text_document.*

PFAM=./
VOCAB=./

python ../tools/preprocess_data.py --input $PFAM/data.json \
	--tokenizer-type BertWordPieceCase --vocab-file $VOCAB/iupac_vocab.txt \
	--output-prefix $PFAM/msa --dataset-impl mmap --workers 64
