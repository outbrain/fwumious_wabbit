
## Scenarios
1. train a new model from a dataset and an output model file - *typical scenario for one-off training on the dataset*
1. train a new model from a cached dataset, and generate an output model - *this is also a typical scenario - we usually run many concurrent model evaluations as part of the model search*
1. use a generated model to make predictions over a dataset read from a text file, and print them to an output predictions file - *this is to illustrate potential serving performance, we don't usually predict from file input as our offline flows always apply online learning. note that when running as daemon we use half as much memory since gradients are not loaded - only model weights.*


## Model
We train a logistic regression model, applying online learning one example at a time (no batches), 
using '--adaptive' flag for adaptive learning rates (AdaGrad variant).

## Results
here are the results for 3 runs for each scenario, taking mean values:
![benchmark results](benchmark_results.png)
Scenario|Runtime (seconds)|Memory (MB)|CPU %
----|----:|----:|----:
vw train, no cache|102.11 | 558 | 166.63
fw train, no cache|14.86 | 257 | 102.00
vw train, using cache|102.98 | 566 | 161.00
fw train, using cache|11.63 | 257 | 101.87
vw predict, no cache|86.45 | 139 | 170.10
fw predict, no cache|11.56 | 129 | 101.90

### Model equivalence
loss values for the test set:

```
Vowpal Wabbit predictions loss: 0.6370
Fwumious Wabbit predictions loss: 0.6370
```


for more details on what makes Fwumious Wabbit so fast, see [here](https://github.com/outbrain/fwumious_wabbit/blob/benchmark/SPEED.md)

### Dataset details
we generate a synthetic dataset with 10,000,000 train records ('train.vw'), and 10,000,000 test records ('easy.vw').

the task is 'Eat-Rate prediction' - each record describes the observed result of a single feeding experiment.
each record is made of a type of animal, a type of food (in Vowpal Wabbit jargon these are our namespaces A and B respectively), and a label indicating whether the animal ate the food.
the underlying model is simple - animals are either herbivores or carnivores,
and food is either plant based or meat based.

herbivores always eat plants (and only plants), and carnivores always eat meat (and only meat).

we name animals conveniently using the pattern 'diet-id', for example 'Herbivore-1234' and 'Carnivore-5678'
and the food similarly as 'food_type-id' - for example 'Plant-678' and 'Meat-234' so the expected label for a record is always obvious.
there are 1,000 animal types, and 1,000 food types. we generate additional 10 random features,
to make the dataset dimensions a bit more realistic.

see for example the first 5 lines from the train dataset (after some pretty-printing):

label|animal|food|feat_2|feat_3|feat_4|feat_5|feat_6|feat_7|...
----:|------|----|----|----|----|----|----|----|----
-1 |A Herbivore-65 |B Meat-120 |C C8117 |D D7364 |E E7737 |F F6219 |G G3439 |H H1537 |...
1 |A Carnivore-272 |B Meat-184 |C C3748 |D D9685 |E E1674 |F F5200 |G G501 |H H365 |...
1 |A Carnivore-135 |B Meat-227 |C C7174 |D D8123 |E E9058 |F F3818 |G G5663 |H H3782 |...
-1 |A Herbivore-47 |B Meat-644 |C C4856 |D D1980 |E E5450 |F F8205 |G G6915 |H H8318 |...
-1 |A Carnivore-603 |B Plant-218 |C C565 |D D7868 |E E3977 |F F6623 |G G6788 |H H2834 |...


### Feature engineering
if we train using separate 'animal type' and 'food type' features, the model won't learn well, 
since knowing the animal identity alone isn't enough to predict if it will eat or not - and the same 
goes for knowing the food type alone.
so we apply an interaction between the animal type and food type fields.
            
## Prerequisites and running
you should have Vowpal Wabbit installed, as the benchmark invokes it via the 'vw' command.

additionally the rust toolchain (particularly cargo and rustc) is required in order to build Fwumious Wabbit (the benchmark invokes '../target/release/fw') 
in order to build and run the benchmark use one of these bash scripts:
```
./run_with_plots.sh
```
in order to run the benchmark and plot the results (requires matplotlib, last used with version 2.1.2)
or, if you just want the numbers with less dependencies run:
```
./run_without_plots.sh
```
## Latest run setup

### versions:
```
vowpal wabbit 8.9.0 (git commit: c24836b36)
fwumious wabbit 0.7 (git commit: 0ca89a6)
```

### CPU Info
```
Intel(R) Xeon(R) CPU E5-2630 v2 @ 2.60GHz
```
### Operating System
```
System: Linux
Version: #186-Ubuntu SMP Mon Dec 4 19:09:19 UTC 2017
```
