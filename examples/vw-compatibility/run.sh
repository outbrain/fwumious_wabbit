#!/bin/sh
SCRIPT=$(readlink -f "$0")
DIR=$(dirname "$SCRIPT")

MODELS_DIR=$DIR/models
PREDICTIONS_DIR=$DIR/predictions
DATASETS_DIR=$DIR/datasets
PROJECT_ROOT=$DIR/../../
FW=$PROJECT_ROOT/target/release/fw
VW=vw

# the incompatibility problem: Vowpal automatically takes all interaction features also as single features. fw does not.
#namespaces="--interactions 4G --interactions 4GHX --interactions 4GUW --interactions 4K --interactions 4c --interactions 4go --interactions 4v --interactions BC --interactions BD --interactions BGO --interactions BX --interactions CO --interactions DG --interactions DW --interactions GU --interactions Gx --interactions KR --interactions MN --interactions UW --interactions Ug --interactions eg --keep B --keep C --keep D --keep F --keep G --keep H --keep L --keep O --keep S --keep U --keep W --keep e --keep f --keep g --keep h --keep i --keep o --keep p --keep q --keep r --keep v --keep x "
namespaces="--keep B --keep C --keep D --keep F --keep G --keep H --keep L --keep O --keep S --keep U --keep W --keep e --keep f --keep g --keep h --keep i --keep o --keep p --keep q --keep r --keep v --keep x "
rest="-l 0.025 -b 25 --adaptive --sgd --link=logistic --loss_function logistic --power_t 0.35 --l2 0.0 --hash all"

mkdir -p $PREDICTIONS_DIR
rm -f $DATASETS_DIR/*.fwcache
rm -f $PREDICTIONS_DIR/*.fw.out
rm -f $DATASETS_DIR/*.cache
rm -f $PREDICTIONS_DIR/*.vw.out
echo "Building FW"
(cd $PROJECT_ROOT
cargo build --release)
VW_CMDLINE="$VW $namespaces $rest --data $DATASETS_DIR/train.vw -p $PREDICTIONS_DIR/train.vw.out"
FW_CMDLINE="$FW $namespaces $rest --data $DATASETS_DIR/train.vw -p $PREDICTIONS_DIR/train.fw.out --vwcompat"
$VW_CMDLINE && $FW_CMDLINE
echo "DONE, now running diff"

diff -s $PREDICTIONS_DIR/train.vw.out $PREDICTIONS_DIR/train.fw.out


