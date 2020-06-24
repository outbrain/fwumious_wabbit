#/bin/sh
infile=$1
namespaces="--interactions BC --interactions BD --interactions BGO --interactions BTf --interactions BX --interactions CO --interactions DG --interactions DW --interactions GHXx --interactions GU --interactions GUx --interactions Gw --interactions KR --interactions KTd --interactions Kx --interactions MN --interactions QS --interactions Sx --interactions UW --interactions Ug --interactions cx --interactions eg --interactions gx --interactions rvx --keep B --keep D --keep F --keep G --keep H --keep L --keep O --keep S --keep U --keep W --keep e --keep f --keep g --keep h --keep i --keep o --keep p --keep q --keep r --keep v --keep w"
#namespaces="--keep A"
rest="--data $infile -l 0.025 -b 25 --adaptive --sgd --loss_function logistic --link logistic --power_t 0.38 --l2 0.00 --hash all --noconstant"
#vwonly="--lrqfa AB-64"
#fwonly="--lrqfa AB-64"

#rest="-b 24 --data $infile -l 0.1 --power_t 0.39 --adaptive --link logistic --sgd --loss_function logistic --noconstant --l2 0.0"
#rm v f
#rm $1.cache $1.fwcache
vw=/home/minmax/minmax_old/zgit/vowpal_wabbit/vowpalwabbit/vw
#vw=/home/minmax/obgit/vw/vowpal_wabbit/vowpalwabbit/vw
clear;
#echo "build --release && target/release/fw $namespaces $rest -p f"
#echo "cargo build --release && target/release/fw $namespaces $rest -c -p f $fwonly && time $vw $namespaces $rest -c -p v $vwonly"

echo "target/release/fw $namespaces $rest -c -p f $fwonly "

cargo build --release && \
target/release/fw $namespaces $rest -c -p f $fwonly && time $vw $namespaces $rest -c -p v $vwonly
#clear; cargo build && target/debug/fw $namespaces $rest -p f && time vw $namespaces $rest -p v



