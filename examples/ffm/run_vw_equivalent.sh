#/bin/sh
SCRIPT=$(readlink -f "$0")
DIR=$(dirname "$SCRIPT")

MODELS_DIR=$DIR/models
PREDICTIONS_DIR=$DIR/predictions
DATASETS_DIR=$DIR/datasets
PROJECT_ROOT=$DIR/../../
VW=vw
echo "Generating input datasets"
(cd $DIR
python3 generate.py)
# keep namespaces A and B as regular logistic regression features
# have two fields, one with feature A, and the second with feature B
# we only need k=1, but we use k=10 for test here
namespaces="--keep A --keep B --interactions AB --lrqfa AB10"

rest="-l 0.1 -b 25 -c --adaptive --sgd --loss_function logistic --link logistic --power_t 0.0 --l2 0.0 --hash all --noconstant"
mkdir -p $MODELS_DIR
mkdir -p $PREDICTIONS_DIR
rm -f $MODELS_DIR/*.vw.model
rm -f $DATASETS_DIR/*.cache
rm -f $PREDICTIONS_DIR/*.vw.out



echo "Running training"
$VW $namespaces $rest --data $DATASETS_DIR/train.vw -p $PREDICTIONS_DIR/train.vw.out -f $MODELS_DIR/trained.vw.model --save_resume
echo "Running prediction on \"easy\" data set"
$VW $namespaces $rest --data $DATASETS_DIR/test-easy.vw -p $PREDICTIONS_DIR/test-easy.vw.out -i $MODELS_DIR/trained.vw.model -t
echo "Running prediction on \"hard\" data set (that needs factorization to be succeffully predicted)"
$VW $namespaces $rest --data $DATASETS_DIR/test-hard.vw -p $PREDICTIONS_DIR/test-hard.vw.out -i $MODELS_DIR/trained.vw.model -t

echo "DONE"
echo "You can find output datasets in directory $PREDICTIONS"
