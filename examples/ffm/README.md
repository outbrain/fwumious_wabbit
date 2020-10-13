# Tests for Factorization Machines functionality

We're trying to predict which animal likes to eat what.

Our dataset will be a simple combination of animal and food and 
outcome - either one likes it (1) or dislikes it (-1)

Namespace A will be Animal, and B will be Food. Each animal has its latent
type - Herbivore or Carnivore. And each food has its latent type - Plant or
Meat.

Example of features will be "Herbivore-13" and "Plant-55". However these are
just strings for the algo. These names are handy when we look at the data as 
it makes it easy for us to check for correctness.

We will split each feature set into two sets - let's call them A1 & A2 set 
and B1 & B2 set.

Our training data will only have interactions between 
- A1 and (B1 U B2) 
- B1 and (A1 U A2)

Importantly there are no interactions between A2 and B2 in the training set.
Predicting resutls of these interactions correctly requires discovery of 
latent variable - which is what FFM can do.

Datasets created by generate.py:
train.vw - In the training set there are no interactions between A2 and B2.
easy.vw - The distribution here is the same as in the training dataset
hard.vw - The distribution here consists entirely of interactions between A2
and B2.

# Notes:
- if we don't use feature combinations - only namespaces A and B in
isolation, then Logistic Regression will not be able to provide any
predictive power
- if we use plain feature combinations - AB, then LR will easily have
correct predictions on easy.vw, but not on hard.vw
- Only if the algo is able to capture the existence of latent variable
it can perform better than random on hard.vw

# Usage:
Demonstrate factorization machines:
sh run.sh 

Run equivalent vowpal setup:
sh run_vw_equivalent.sh
