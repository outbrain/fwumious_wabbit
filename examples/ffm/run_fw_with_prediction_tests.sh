#!/bin/bash
set -e

SCRIPT=$(readlink -f "$0")
DIR=$(dirname "$SCRIPT")
echo "Generating input datasets"
rm -rf datasets;
(cd $DIR
 python3 generate.py --num_animals 10 --num_foods 5)

# Probability threshold considered
THRESHOLD=0.5
MARGIN_OF_PERFORMANCE_BA=0.2
PROJECT_ROOT=$DIR/../../
FW=$PROJECT_ROOT/target/release/fw
DATASET_FOLDER=$DIR/datasets
TRAIN_DATA=$DATASET_FOLDER/data.vw;
PRELOG="$DATASET_FOLDER/prelog.csv";
TEST_DATA=$TRAIN_DATA;
INFO_STRING="==============>"

rm -rf models;
rm -rf predictions;

mkdir -p models;
mkdir -p predictions;

echo "Building FW"
(cd $PROJECT_ROOT
cargo build --release);

# Change this to your preference if required!
namespaces="--keep A --keep B --interactions AB --ffm_k 10 --ffm_field A --ffm_field B" 
rest="-l 0.1 -b 25 -c --sgd --loss_function logistic --link logistic --power_t 0.0 --l2 0.0 --hash all --noconstant"

# Train on a given data set
$FW $namespaces $rest --data $DATASET_FOLDER/train.vw -p $DIR/predictions/training.txt -f $DIR/models/full_weights.fw.model --save_resume

# Create inference weights
$FW $namespaces $rest -i models/full_weights.fw.model --convert_inference_regressor ./models/inference_weights.fw.model

# Test full weights on a given data set
$FW $namespaces $rest -i models/full_weights.fw.model --data $DATASET_FOLDER/train.vw -p ./predictions/eval_full_weight_space.txt -t

# Test inference weights on a given data set
$FW $namespaces $rest -i models/inference_weights.fw.model -d $DATASET_FOLDER/train.vw -t -p ./predictions/eval_inference_only.txt

###########################################
# Test the predictions and their validity #
###########################################

# Create ground truth labels first
cat datasets/train.vw |mawk '{print $1}' > predictions/ground_truth.txt;

# get last n predictions of training
cat ./predictions/training.txt | tail -n $(cat predictions/ground_truth.txt | wc -l) > ./predictions/training_eval_part_only.txt

# check line counts first (same amount of eval instances)
if [ $(cat predictions/ground_truth.txt | wc -l) = $(cat predictions/training_eval_part_only.txt | wc -l) ]
then
	echo "$INFO_STRING Matching prediction counts! The test can proceed .."
else
	echo "$INFO_STRING Ground truth number different to eval number of training predictions, exiting .."
	exit 1;
fi

######################################################################################################################
# Create a single file for subsequent prediction analysis; columns are:												 #
# training's predictions -- predictions only inference -- predictions using full weight space -- ground truth labels #
######################################################################################################################
paste predictions/training_eval_part_only.txt predictions/eval_inference_only.txt predictions/eval_full_weight_space.txt predictions/ground_truth.txt  > ./predictions/joint_prediction_space.txt

yes "0.0" | head -n $(cat predictions/joint_prediction_space.txt | wc -l) > ./predictions/all_negative.txt

paste predictions/joint_prediction_space.txt predictions/all_negative.txt > ./tmp.txt;
mv ./tmp.txt ./predictions/joint_prediction_space.txt;

# All instances
ALL_INSTANCES=$(cat predictions/joint_prediction_space.txt | wc -l);

# TEST - are all inference predictions the same?
NUM_UNIQUE_INFERENCE_ONLY_EVAL=$(cat predictions/eval_inference_only.txt | sort -u | wc -l)
NUM_UNIQUE_FULL_WEIGHTS_EVAL=$(cat predictions/eval_full_weight_space.txt | sort -u | wc -l)
NUM_UNIQUE_TRAINING_RUN_EVAL=$(cat predictions/training.txt | sort -u | wc -l)


if [ $NUM_UNIQUE_FULL_WEIGHTS_EVAL = 1 ]
then
	echo "$INFO_STRING WARNING: all predictions are the same if using full weights file for inference only.";
	exit 1;
fi

if [ $NUM_UNIQUE_INFERENCE_ONLY_EVAL = 1 ]
then
	echo "$INFO_STRING WARNING: all predictions are the same if using inference weights file for inference only.";
	exit 1;
fi

if [ $NUM_UNIQUE_TRAINING_RUN_EVAL = 1 ]
then
	echo "$INFO_STRING WARNING: all predictions are the same during training.";
	exit 1;
	
fi

echo -e "OUTPUT_TAG\tTHRESHOLD\tPRECISION\tRECALL\tF1\tACCURACY\tBALANCED_ACCURACY"
ALL_INSTANCES=$(cat predictions/joint_prediction_space.txt | wc -l)
ALL_INSTANCES_POSITIVE=$(cat predictions/joint_prediction_space.txt| awk '{print $4}'| grep '1' | wc -l)
ALL_INSTANCES_NEGATIVE=$(cat predictions/joint_prediction_space.txt| awk '{print $4}'| grep '\-1' | wc -l)

TP=$(cat predictions/joint_prediction_space.txt | awk -v THRESHOLD="$THRESHOLD" '($4=="1") &&  ($1>=THRESHOLD) {positiveMatch++} END {print positiveMatch}');

TN=$(cat predictions/joint_prediction_space.txt | awk -v THRESHOLD="$THRESHOLD" '($4=="-1") &&  ($1<THRESHOLD) {positiveMatch++} END {print positiveMatch}');

FP=$(cat predictions/joint_prediction_space.txt | awk -v THRESHOLD="$THRESHOLD" '($4=="-1") &&  ($1>=THRESHOLD) {positiveMatch++} END {print positiveMatch}');

FN=$(cat predictions/joint_prediction_space.txt | awk -v THRESHOLD="$THRESHOLD" '($4=="1") &&  ($1<THRESHOLD) {positiveMatch++} END {print positiveMatch}');

PRECISION_FW=$(bc <<<"scale=5 ; $TP / ($TP + $FP)");
RECALL_FW=$(bc <<< "scale=5 ; $TP / ($TP + $FN)");
F1_FW=$(bc <<< "scale=5 ; $TP / ($TP + 0.5 * ($FP + $FN))");
ACCURACY_FW=$(bc <<< "scale=5 ; (TP + TN) / ($TP + $TN + $FP + $FN)");
SENSITIVITY_FW=$(bc <<< "scale=5 ; $TP / $ALL_INSTANCES_POSITIVE");
SPECIFICITY_FW=$(bc <<< "scale=5 ; $TN / $ALL_INSTANCES_NEGATIVE");
BALANCED_ACCURACY_FW=$(bc <<< "scale=5 ; ($SENSITIVITY_FW + $SPECIFICITY_FW) / 2");
echo -e "FW\t$THRESHOLD\t$PRECISION_FW\t$RECALL_FW\t$F1_FW\t$ACCURACY_FW\t$BALANCED_ACCURACY_FW";

# Random baseline
TP=$(cat predictions/joint_prediction_space.txt | awk -v THRESHOLD="$THRESHOLD" '($4=="1") &&  (rand()>=THRESHOLD) {positiveMatch++} END {print positiveMatch}');

TN=$(cat predictions/joint_prediction_space.txt | awk -v THRESHOLD="$THRESHOLD" '($4=="-1") &&  (rand()<THRESHOLD) {positiveMatch++} END {print positiveMatch}');

FP=$(cat predictions/joint_prediction_space.txt | awk -v THRESHOLD="$THRESHOLD" '($4=="-1") &&  (rand()>=THRESHOLD) {positiveMatch++} END {print positiveMatch}');

FN=$(cat predictions/joint_prediction_space.txt | awk -v THRESHOLD="$THRESHOLD" '($4=="1") &&  (rand()<THRESHOLD) {positiveMatch++} END {print positiveMatch}');

PRECISION=$(bc <<<"scale=5 ; $TP / ($TP + $FP)");
RECALL=$(bc <<< "scale=5 ; $TP / ($TP + $FN)");
F1=$(bc <<< "scale=5 ; $TP / ($TP + 0.5 * ($FP + $FN))");
ACCURACY=$(bc <<< "scale=5 ; (TP + TN) / ($TP + $TN + $FP + $FN)");
SENSITIVITY=$(bc <<< "scale=5 ; $TP / $ALL_INSTANCES_POSITIVE");
SPECIFICITY=$(bc <<< "scale=5 ; $TN / $ALL_INSTANCES_NEGATIVE");
BALANCED_ACCURACY=$(bc <<< "scale=5 ; ($SENSITIVITY + $SPECIFICITY) / 2");
echo -e "RANDOM\t$THRESHOLD\t$PRECISION\t$RECALL\t$F1\t$ACCURACY\t$BALANCED_ACCURACY";

# Is the difference substantial (in BA)
BA_DIFF=$(bc <<< "scale=5 ; $BALANCED_ACCURACY_FW - $BALANCED_ACCURACY");
ZERO_VAR="0.0"
if [ 1 -eq "$(echo "$BA_DIFF > $ZERO_VAR" | bc)" ];
then
	echo "$INFO_STRING FW learned much better than random (on training), exiting gracefully.";
fi
