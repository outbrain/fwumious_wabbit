#!/bin/sh
SCRIPT=$(readlink -f "$0")
DIR=$(dirname "$SCRIPT")

MODELS_DIR=$DIR/models
PREDICTIONS_DIR=$DIR/predictions
DATASETS_DIR=$DIR/datasets
PROJECT_ROOT=$DIR/../../
FW=$PROJECT_ROOT/target/release/fw

echo $FW
namespaces="--interactions 4G --interactions 4GHX --interactions 4GUW --interactions 4K --interactions 4c --interactions 4go --interactions 4v --interactions BC --interactions BD --interactions BGO --interactions BX --interactions CO --interactions DG --interactions DW --interactions GU --interactions Gx --interactions KR --interactions MN --interactions UW --interactions Ug --interactions eg --keep B --keep C --keep D --keep F --keep G --keep H --keep L --keep O --keep S --keep U --keep W --keep e --keep f --keep g --keep h --keep i --keep o --keep p --keep q --keep r --keep v --keep x "
rest="-l 0.025 -b 25 --adaptive --sgd --link=logistic --loss_function logistic --power_t 0.39 --l2 0.0 --hash all"

mkdir -p $MODELS_DIR
mkdir -p $PREDICTIONS_DIR
rm -f $MODELS_DIR/*.fw.model
rm -f $DATASETS_DIR/*.fwcache
rm -f $PREDICTIONS_DIR/*.fw.out
echo "Building FW"
(cd $PROJECT_ROOT
cargo build --release)
CMDLINE="$FW $namespaces $rest --data $DATASETS_DIR/train.vw -p $PREDICTIONS_DIR/train.fw.out -f $MODELS_DIR/trained.fw.model --save_resume"
echo "We will run $CMDLINE"
$CMDLINE
echo "DONE"
