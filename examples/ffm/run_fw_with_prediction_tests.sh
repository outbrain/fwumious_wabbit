#!/bin/bash
set -e

function compute_main_metrics {
	# A method that takes err1/2 counts from context and overwrites main statistics
	
	PRECISION=$(bc <<<"scale=5 ; $TP / ($TP + $FP)");
	RECALL=$(bc <<< "scale=5 ; $TP / ($TP + $FN)");
	F1=$(bc <<< "scale=5 ; $TP / ($TP + 0.5 * ($FP + $FN))");
	SENSITIVITY=$(bc <<< "scale=5 ; $TP / $ALL_INSTANCES_POSITIVE");
	SPECIFICITY=$(bc <<< "scale=5 ; $TN / $ALL_INSTANCES_NEGATIVE");
	BALANCED_ACCURACY=$(bc <<< "scale=5 ; ($SENSITIVITY + $SPECIFICITY) / 2");
}

SCRIPT=$(readlink -f "$0")
DIR=$(dirname "$SCRIPT")
echo "Generating input datasets"
rm -rf datasets;
(cd $DIR
 python3 generate.py --num_animals 300 --num_foods 200 --num_train_examples 40000)

# Probability threshold considered
THRESHOLD=0.5

# Training performance margin required to pass (Balanced acc.)
MARGIN_OF_PERFORMANCE_BA=0.45
MARGIN_OF_PERFORMANCE_HARD_TEST_BA=0.80

# Project structure
PROJECT_ROOT=$DIR/../../
FW=$PROJECT_ROOT/target/release/fw
DATASET_FOLDER=$DIR/datasets
TRAIN_DATA=$DATASET_FOLDER/data.vw;

# Nicer printing
INFO_STRING="==============>"

# Cleanup
rm -rf models;
rm -rf predictions;
mkdir -p models;
mkdir -p predictions;

echo "Building FW"
(cd $PROJECT_ROOT
cargo build --release);

# Change this to your preference if required - this is tailored for the toy example
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
cat datasets/train.vw |mmawk '{print $1}' > predictions/ground_truth.txt;

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

# Generate a "dummy" prediction space
yes "0.0" | head -n $(cat predictions/joint_prediction_space.txt | wc -l) > ./predictions/all_negative.txt

# Form the final dataframe
paste predictions/joint_prediction_space.txt predictions/all_negative.txt > ./tmp.txt;
mv ./tmp.txt ./predictions/joint_prediction_space.txt;

# All instances
ALL_INSTANCES=$(cat predictions/joint_prediction_space.txt | wc -l);

# Are inference weights' predictions the same?
INFERENCE_SAME_COUNT=$(cat ./predictions/joint_prediction_space.txt | mawk '$2==$3' | wc -l)

if [ $ALL_INSTANCES = $INFERENCE_SAME_COUNT ];
then
	echo "$INFO_STRING All inferences' weights' predictions are the same .."
else
	echo "$INFO_STRING inference weights produce different predictions to full weights!"
	exit 1;
fi

NUM_UNIQUE_INFERENCE_ONLY_EVAL=$(cat predictions/eval_inference_only.txt | sort -u | wc -l)
NUM_UNIQUE_FULL_WEIGHTS_EVAL=$(cat predictions/eval_full_weight_space.txt | sort -u | wc -l)
NUM_UNIQUE_TRAINING_RUN_EVAL=$(cat predictions/training.txt | sort -u | wc -l)

# Are all predictions for full weights the same?
if [ $NUM_UNIQUE_FULL_WEIGHTS_EVAL = 1 ]
then
	echo "$INFO_STRING WARNING: all predictions are the same if using full weights file for inference only.";
	exit 1;
fi

# Are all inference weights-based predictions fine?
if [ $NUM_UNIQUE_INFERENCE_ONLY_EVAL = 1 ]
then
	echo "$INFO_STRING WARNING: all predictions are the same if using inference weights file for inference only.";
	exit 1;
fi

# Are all training predictions same?
if [ $NUM_UNIQUE_TRAINING_RUN_EVAL = 1 ]
then
	echo "$INFO_STRING WARNING: all predictions are the same during training.";
	exit 1;
	
fi

# Create a benchmark against random classifier
echo -e "OUTPUT_TAG\tTHRESHOLD\tPRECISION\tRECALL\tF1\tBALANCED_ACCURACY"
ALL_INSTANCES=$(cat predictions/joint_prediction_space.txt | wc -l)
ALL_INSTANCES_POSITIVE=$(cat predictions/joint_prediction_space.txt| mawk '{print $4}'| grep -v '\-1' | wc -l)
ALL_INSTANCES_NEGATIVE=$(cat predictions/joint_prediction_space.txt| mawk '{print $4}'| grep '\-1' | wc -l)

TP=$(cat predictions/joint_prediction_space.txt | mawk -v THRESHOLD="$THRESHOLD" '($4=="1") &&  ($3>=THRESHOLD) {positiveMatch++} END {print positiveMatch}');

TN=$(cat predictions/joint_prediction_space.txt | mawk -v THRESHOLD="$THRESHOLD" '($4=="-1") &&  ($3<THRESHOLD) {positiveMatch++} END {print positiveMatch}');

FP=$(cat predictions/joint_prediction_space.txt | mawk -v THRESHOLD="$THRESHOLD" '($4=="-1") &&  ($3>=THRESHOLD) {positiveMatch++} END {print positiveMatch}');

FN=$(cat predictions/joint_prediction_space.txt | mawk -v THRESHOLD="$THRESHOLD" '($4=="1") &&  ($3<THRESHOLD) {positiveMatch++} END {print positiveMatch}');

# Account for corner cases
if [ "$FP" = "" ]; then
	FP=0
fi

if [ "$FN" = "" ]; then
	FN=0
fi

compute_main_metrics
echo -e "FW\t$THRESHOLD\t$PRECISION\t$RECALL\t$F1\t$BALANCED_ACCURACY";

# Random baseline
TP=$(cat predictions/joint_prediction_space.txt | mawk -v THRESHOLD="$THRESHOLD" '($4=="1") &&  (rand()>=THRESHOLD) {positiveMatch++} END {print positiveMatch}');

TN=$(cat predictions/joint_prediction_space.txt | mawk -v THRESHOLD="$THRESHOLD" '($4=="-1") &&  (rand()<THRESHOLD) {positiveMatch++} END {print positiveMatch}');

FP=$(cat predictions/joint_prediction_space.txt | mawk -v THRESHOLD="$THRESHOLD" '($4=="-1") &&  (rand()>=THRESHOLD) {positiveMatch++} END {print positiveMatch}');

FN=$(cat predictions/joint_prediction_space.txt | mawk -v THRESHOLD="$THRESHOLD" '($4=="1") &&  (rand()<THRESHOLD) {positiveMatch++} END {print positiveMatch}');

compute_main_metrics

echo -e "RANDOM\t$THRESHOLD\t$PRECISION\t$RECALL\t$F1\t$BALANCED_ACCURACY";

# Is the difference substantial (in BA)
BA_DIFF=$(bc <<< "scale=5 ; $BALANCED_ACCURACY_FW - $BALANCED_ACCURACY");
ZERO_VAR="0.0"

# BA margin must be beyond specified threshold for this to pass
if [ 1 -eq "$(echo "$BA_DIFF > $ZERO_VAR" | bc)" ];
then
	echo "$INFO_STRING FW learned much better than random (on training), exiting gracefully.";
fi


#######################################
# PART 2 - benchmarks on the test set #
#######################################

# Test inference weights on a given data set
$FW $namespaces $rest -i models/inference_weights.fw.model -d $DATASET_FOLDER/test-hard.vw -t -p ./predictions/test_hard_predictions.txt

cat ./datasets/test-hard.vw|mmawk '{print $1}' > ./predictions/hard_ground_truth.txt;
paste predictions/test_hard_predictions.txt predictions/hard_ground_truth.txt > ./predictions/joint_hard_predictions_and_ground.txt;


ALL_INSTANCES_POSITIVE=$(cat predictions/joint_hard_predictions_and_ground.txt | mawk '{print $2}'| grep -v '\-1' | wc -l)
ALL_INSTANCES_NEGATIVE=$(cat predictions/joint_hard_predictions_and_ground.txt | mawk '{print $2}'| grep '\-1' | wc -l)

# Random baseline
TP=$(cat predictions/joint_hard_predictions_and_ground.txt | mawk -v THRESHOLD="$THRESHOLD" '($2=="1") &&  ($1>=THRESHOLD) {positiveMatch++} END {print positiveMatch}');

TN=$(cat predictions/joint_hard_predictions_and_ground.txt | mawk -v THRESHOLD="$THRESHOLD" '($2=="-1") &&  ($1<THRESHOLD) {positiveMatch++} END {print positiveMatch}');

FP=$(cat predictions/joint_hard_predictions_and_ground.txt | mawk -v THRESHOLD="$THRESHOLD" '($2=="-1") &&  ($1>=THRESHOLD) {positiveMatch++} END {print positiveMatch}');

FN=$(cat predictions/joint_hard_predictions_and_ground.txt | mawk -v THRESHOLD="$THRESHOLD" '($2=="1") &&  ($1<THRESHOLD) {positiveMatch++} END {print positiveMatch}');

compute_main_metrics
echo -e "FW-hard-test\t$THRESHOLD\t$PRECISION\t$RECALL\t$F1\t$BALANCED_ACCURACY";

if [ 1 -eq "$(echo "($BALANCED_ACCURACY - $MARGIN_OF_PERFORMANCE_HARD_TEST_BA) > 0" | bc)" ];
then
	echo "$INFO_STRING FW learned much better than random (on hard test), exiting gracefully.";
else
	echo "$INFO_STRING FW did not learn to classify the hard problem well enough!"
	exit 1
fi
