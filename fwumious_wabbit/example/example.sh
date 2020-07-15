#!/bin/sh
$infile 
namespaces="--interactions 4G --interactions 4GHX --interactions 4GUW --interactions 4K --interactions 4c --interactions 4go --interactions 4v --interactions BC --interactions BD --interactions BGO --interactions BX --interactions CO --interactions DG --interactions DW --interactions GU --interactions Gx --interactions KR --interactions MN --interactions UW --interactions Ug --interactions eg --keep B --keep C --keep D --keep F --keep G --keep H --keep L --keep O --keep S --keep U --keep W --keep e --keep f --keep g --keep h --keep i --keep o --keep p --keep q --keep r --keep v --keep x "
rest="--data 100.vw -l 0.025 -b 25 --adaptive --sgd --link=logistic --loss_function logistic --power_t 0.39 --l2 0.0 --hash all"

.. cargo build --release
../target/release/fw $namespaces $rest -c -p fw.out
echo "DONE, results are in fw.out"
