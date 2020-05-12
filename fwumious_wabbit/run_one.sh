#/bin/sh
infile=$1
#namespaces="--interactions 4G --interactions 4GHX --interactions 4GUW --interactions 4K --interactions 4c --interactions 4go --interactions 4v --interactions BC --interactions BD --interactions BGO --interactions BX --interactions CO --interactions DG --interactions DO --interactions DW --interactions GU --interactions Gx --interactions KR --interactions MN --interactions UW --interactions Ug --interactions eg --keep B --keep C --keep D --keep F --keep G --keep H --keep L --keep O --keep S --keep U --keep W --keep e --keep f --keep g --keep h --keep i --keep o --keep p --keep q --keep r --keep v --keep x "
namespaces="--keep A"
rest="--data $infile -l 0.025 -b 25 --adaptive --sgd --loss_function logistic --link logistic --power_t 0.0 --l2 0.0 --hash all --noconstant"

#rest="-b 24 --data $infile -l 0.1 --power_t 0.39 --adaptive --link logistic --sgd --loss_function logistic --noconstant --l2 0.0"
rm v f
#rm $1.cache $1.fwcache
vw=/home/minmax/minmax_old/zgit/vowpal_wabbit/vowpalwabbit/vw
#vw=/home/minmax/obgit/vw/vowpal_wabbit/vowpalwabbit/vw
clear;
#echo "build --release && target/release/fw $namespaces $rest -p f"
echo "cargo build --release && target/release/fw $namespaces $rest -c -p f && time $vw $namespaces $rest -c -p v"

cargo build --release && target/release/fw $namespaces $rest -c -p f && time $vw $namespaces $rest -c -p v
#clear; cargo build && target/debug/fw $namespaces $rest -p f && time vw $namespaces $rest -p v

