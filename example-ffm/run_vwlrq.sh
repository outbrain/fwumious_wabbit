#/bin/sh
infile=$1
#namespaces="--interactions 4G --interactions 4GHX --interactions 4GUW --interactions 4K --interactions 4c --interactions 4go --interactions 4v --interactions BC --interactions BD --interactions BGO --interactions BX --interactions CO --interactions DG --interactions DO --interactions DW --interactions GU --interactions Gx --interactions KR --interactions MN --interactions UW --interactions Ug --interactions eg --keep B --keep C --keep D --keep F --keep G --keep H --keep L --keep O --keep S --keep U --keep W --keep e --keep f --keep g --keep h --keep i --keep o --keep p --keep q --keep r --keep v --keep x "
namespaces="--keep A --keep B"
rest="-l 0.1 -c --adaptive --sgd --loss_function logistic  --power_t 0.0 --l2 0.0 --hash all --noconstant"
rm v1 veasy vhard
rm fm/*.fwcache
vw=/home/minmax/minmax_old/zgit/vowpal_wabbit/vowpalwabbit/vw
#vw=/home/minmax/obgit/vw/vowpal_wabbit/vowpalwabbit/vw
fw=../target/debug/fw
clear;
#echo "build --release && target/release/fw $namespaces $rest -p f"
(cd ..
cargo build) && \
$vw $namespaces $rest --lrqfa AB10 --link logistic -b 25 --data train.vw -p v1 -f v1.reg --save_resume && \
$vw $namespaces $rest --data easy.vw -p veasy -i v1.reg -t && \
$vw $namespaces $rest --data hard.vw -p vhard -i v1.reg -t

#clear; cargo build && target/debug/fw $namespaces $rest -p f && time vw $namespaces $rest -p v

