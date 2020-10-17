
## Scenarios
1. train a new model from a dataset and an output model file - *typical scenario for one-off training on the dataset*
1. train a new model from a cached dataset, and generate an output model - *this is also a typical scenario - we usually run many concurrent model evaluations as part of the model search*
1. use a generated model to make predictions over a dataset read from a text file, and print them to an output predictions file - *this is to illustrate potential serving performance, we don't usually predict from file input as our offline flows always apply online learning. note that when running as daemon we use half as much memory since gradients are not loaded - only model weights.*


## Model
We train a logistic regression model, applying online learning one example at a time (no batches), 
using '--adaptive' flag for adaptive learning rates (AdaGrad variant).

## Results
here are the results for {times} runs for each scenario, taking mean values:
![benchmark results](benchmark_results.png)
Scenario|Runtime (seconds)|Memory (MB)|CPU %
----|----:|----:|----:
vw train, no cache|6.32 | 542 | 170.47
fw train, no cache|1.10 | 258 | 105.87
vw train, using cache|6.25 | 542 | 164.17
fw train, using cache|1.08 | 258 | 105.40
vw predict, no cache|4.59 | 139 | 177.07
fw predict, no cache|0.91 | 258 | 105.87

        ### Model equivalence
        loss values for the test set:
```
Vowpal Wabbit predictions loss: 0.7804
Fwumious Wabbit predictions loss: 0.7804
```


for more details on what makes Fwumious Wabbit so fast, see [here](https://github.com/outbrain/fwumious_wabbit/blob/benchmark/SPEED.md)

### Dataset details
we generate a synthetic dataset with 1,000,000 train records ('train.vw'), and 1,000,000 test records ('easy.vw').
the task is 'Eat-Rate prediction' - each record describes the observed result of a single feeding experiment.
each record is made of a type of animal, a type of food (in Vowpal Wabbit jargon these are our namespaces A and B respectively), and a label indicating whether the animal ate the food.
the underlying model is simple - animals are either herbivores or carnivores,
and food is either plant based or meat based.
herbivores always eat plants (and only plants), and carnivores always eat meat (and only meat).
we name animals conveniently using the pattern 'diet-id', for example 'Herbivore-1234' and 'Carnivore-5678'
and the food similarly as 'food_type-id' - for example 'Plant-678' and 'Meat-234' so the expected label for a record is always obvious.
there are 1,000 animal types, and 1,000 food types.

see for example the first 5 lines from the train dataset (after some pretty-printing):
                label|animal|food
                ----:|------|----
                
-1 |A Herbivore-65 |B Meat-120 |C C8117 |D D7364 |E E7737 |F F6219 |G G3439 |H H1537 |I I7993 |J J464 |K K6386 |L L7090
1 |A Carnivore-272 |B Meat-184 |C C3748 |D D9685 |E E1674 |F F5200 |G G501 |H H365 |I I416 |J J8870 |K K150 |L L6245
1 |A Carnivore-135 |B Meat-227 |C C7174 |D D8123 |E E9058 |F F3818 |G G5663 |H H3782 |I I3584 |J J7530 |K K4747 |L L352
-1 |A Herbivore-47 |B Meat-644 |C C4856 |D D1980 |E E5450 |F F8205 |G G6915 |H H8318 |I I3110 |J J4970 |K K4655 |L L9626
-1 |A Carnivore-603 |B Plant-218 |C C565 |D D7868 |E E3977 |F F6623 |G G6788 |H H2834 |I I6014 |J J8991 |K K6139 |L L1416


### Feature engineering
if we train using separate 'animal type' and 'food type' features, the model won't learn well, 
since knowing the animal identity alone isn't enough to predict if it will eat or not - and the same 
goes for knowing the food type alone.
so we apply an interaction between the animal type and food type fields.
            
## Prerequisites and running
you should have Vowpal Wabbit installed, as the benchmark invokes it via the 'vw' command.
additionally the rust compiler is required in order to build Fwumious Wabbit (the benchmark invokes '../target/release/fw') 
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
Current Frequency: 4108.18Mhz
```
### System Information
```
System: Linux
Version: #52-Ubuntu SMP Thu Sep 10 10:58:49 UTC 2020
Machine: x86_64
Processor: x86_64
```
