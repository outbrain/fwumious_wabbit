#!/bin/bash
SCRIPT=$(readlink -f "$0")
DIR=$(dirname "$SCRIPT")

MODELS_DIR=$DIR/models
PREDICTIONS_DIR=$DIR/predictions
DATASETS_DIR=$DIR/datasets
PROJECT_ROOT=$DIR/../../
FW=$PROJECT_ROOT/target/release/fw
echo "Generating input datasets"
(cd $DIR
python3 generate.py)
# keep namespaces A and B as regular logistic regression features
# have two fields, one with feature A, and the second with feature B
# we only need k=1, but we use k=10 for test here
rm -f $PREDICTIONS_DIR/*.fw.out;
namespaces="--keep A --keep B --interactions AB --ffm_k 2 --ffm_field A --ffm_field B" 

initializations=("he" "constant" "default" "xavier_normalized" "xavier")
for i in "${initializations[@]}"
do
	echo ">>>>>>>>>>>>>>> Considering initialization $i";
	rest="-l 0.1 -b 25 -c --adaptive --sgd --loss_function logistic --link logistic --power_t 0.0 --l2 0.0 --hash all --noconstant --initialization_type $i"
	mkdir -p $MODELS_DIR
	mkdir -p $PREDICTIONS_DIR
	rm -f $MODELS_DIR/*.fw.model
	rm -f $DATASETS_DIR/*.fwcache
	echo "Building FW"
	(cd $PROJECT_ROOT
	 cargo build --release) && \
		echo "Running training"
	$FW $namespaces $rest --data $DATASETS_DIR/train.vw -p $PREDICTIONS_DIR/train.fw.out -f $MODELS_DIR/trained.fw.model --save_resume
	echo "Running prediction on \"hard\" data set (that needs factorization to be succeffully predicted)"
	$FW $namespaces $rest --data $DATASETS_DIR/test-hard.vw -p $PREDICTIONS_DIR/test-initialization_$i.fw.out -i $MODELS_DIR/trained.fw.model -t
	echo "DONE"
	echo "You can find output datasets in directory $PREDICTIONS"

done

# Requires pip install scikit-learn if not present
python compare_initializations.py
