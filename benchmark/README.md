## Prerequisites and running
you should have Vowpal Wabbit installed, as the benchmark invokes it via the 'vw' command.
additionally the rust compiler is required in order to build Fwumious Wabbit (using './target/release/fw') 
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

### CPU Info
```
Physical cores: 4
Total cores: 8
Current Frequency: 2900.00Mhz
```
### System Information
```
System: Darwin
Version: Darwin Kernel Version 19.6.0: Mon Aug 31 22:12:52 PDT 2020; root:xnu-6153.141.2~1/RELEASE_X86_64
Machine: x86_64
Processor: i386
```
### Dataset details
we generate a synthetic dataset with 10,000,000 train records ('train.vw'), and 10,000,000 test records ('easy.vw').

the task is 'Eat-Rate prediction' - each record describes the observed result of a single feeding experiment.

each record is made of a type of animal, a type of food, and a label indicating whether the animal ate the food.

the underlying model is simple - animals are either herbivores or carnivores,
and food is either plant based or meat based.
herbivores always eat plants (and only plants), and carnivores always eat meat (and only meat).

we name animals conveniently using the pattern 'diet-id', for example 'Herbivore-1234' and 'Carnivore-5678',
and the food similarly as 'food_type-id' - for example 'Plant-678'
 and 'Meat-234' so the expected label for a record is always obvious.

there are 1,000 animal types, and 1,000 food types.


see for example the first 5 lines from the train dataset (after some pretty-printing):
label|animal|food
-----|------|----
1 |A Herbivore-708 |B Plant-43
1 |A Herbivore-588 |B Plant-107
-1 |A Herbivore-134 |B Meat-33
1 |A Carnivore-567 |B Meat-161
1 |A Carnivore-206 |B Meat-607


## Results

We train a logistic regression model, applying online learning one example at a time (no batches), 

using '--adaptive' learning rates (AdaGrad variant).

if we train using separate 'animal type' and 'food type' features, the model won't learn well, 
since knowing the animal identity alone isn't enough to predict if it will eat or not - and the same 
goes for knowing the food type alone.

That's why we use an interaction between the animal type and food type.

**we measure 3 scenarios:**
1. train a new model from a gzipped dataset, generating a gzipped cache file for future runs, and an output model file - *this is a typical scenario in our AutoML system - we start by generating the cache file for the next runs.*
1. train a new model over the dataset in the gzipped cache, and generate an output model - *this is also a typical scenario - we usually run many concurrent model evaluations as part of the model search*
1. use a generated model to make predictions over a dataset read from a text file, and print them to an output predictions file - *this is to illustrate potential serving performance, we don't usually predict from file input as our offline flows always apply online learning. note that when running as daemon we use half as much memory since gradients are not loaded - only model weights.*


### Summary
here are the results for 1 runs for each scenario, taking mean values:

![benchmark results](benchmark_results.png)
Scenario|Runtime (seconds)|Memory (MB)|CPU %
----|----|----|----
vw train, no cache|67.35 | 618 | 159.70
fw train, no cache|9.13 | 258 | 99.70
vw train, using cache|65.98 | 618 | 152.40
fw train, using cache|4.46 | 258 | 99.50
vw predict, no cache|69.75 | 138 | 155.10
fw predict, no cache|5.35 | 257 | 99.50


### Model equivalence
see here the loss value calculated over the test predictions for the tested models:
```
Vowpal Wabbit predictions loss: 0.5745
Fwumious Wabbit predictions loss: 0.5646
```


for more details on what makes Fwumious Wabbit so fast, see [here](https://github.com/outbrain/fwumious_wabbit/blob/benchmark/SPEED.md)
## Field aware factorization machines
in this experiment we demonstrate how field aware factorization machines (FFMs) can better capture 
feature interactions, resulting in better model accuracy.

### Dataset
In the train set we generated, the animals and foods are each divided to two groups - we'll mark them A1 and A2 for the animals,
and F1 and F2 for the foods.

the train set and the test set named 'easy.vw' (used in the previous section) are both drawn from the same distribution, 
with records which belong to {(A1 U A2, F1)} U {(A1, F1 U F2}).

the test set we use here, 'hard.vw', is different: it contains exclusively records from {(A2, F2)} - combinations unseen in the train set.

In order for a model to make correct predictions on this dataset after training on the train dataset, 
it must be able to generalize for unseen combinations.
we skip the latency, memory and CPU comparison as for this synthetic dataset the difference is negligible.
### Loss on the test set
```
Fwumious Wabbit Logistic Regression predictions loss: 0.6948
Fwumious Wabbit FFM predictions loss: 0.2027
```
