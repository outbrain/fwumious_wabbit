Fwumious Wabbit is:
- super-fast 
- Rust implementation of 
- a subset of Vowpal Wabbit functionality

Fwumious Wabbit is actively used in Outbrain for off-line research. It 
enables "high bandwidth research" when doing feature engineering, feature 
selection, hyperparameter tuning and similar. Data scientists can train 
hundreds of models over the data of hundreds of millions of examples in 
the matter of hours on a single machine.

For what it supports it is two orders of magnitude faster than the fastest 
Tensorflow implementation we could come up with (including using GPUs). 
It is an order of magnitude faster than Vowpal for some specific use-cases.

Fwumious Wabbit name is a homage to Vowpal Wabbit which was original
inspiration for developing this software.

Fwumious Wabbit main properties are:
- logloss is the only loss function supported
- namespaces used in input examples have to be listed ahead of time
- very fast implementation of Field-aware Factorization Machines (FFM)
- single core only

Why is it faster?
- tighter encoding format for examples cache
- namespaces have to be declared up-front, so tight encoding is possible
- using lz4 for examples cache compression instead of gz
- using Look Up Tables for AdaGrad (--fastmath option)
- inner loop is single purpose and super-optimized
- it is written in Rust and it uses specialization tricks (via macros)
- it cuts corners by preallocating buffers
- a lot of profiling and optimizing the bottlenecks

# Vowpal compatibility
WARNING: Fwumious cuts a lot of corners. Beware.

Input file compatibility:
- Namespaces can only be single letters
- In each example each namespace can only be delcared once (and can have multiple features)
- there has to be a map file available with all namespaces declared (vw_namespace_map.csv)

Restrictions of Vowpal compatibility as regard to command line parameters:
MUST:
 --hash all	This treats all features as categorical. Otherwise Vowpal
treats some as already prehashed features
 --adaptive	Adagrad  mode
 --sgd		Disable vowpal defaults of "--invariant" and "--normalize"

OPTIONAL:
 --link logistic           Use logistic function for prediction printouts (always on)
 --loss_function logistic  Use logloss (always on)
 --power_t 0.5		  Value for Adagrad's exponent (default 0.5 = square root)
 --l2 0.0		  L2 regularization, not supported. Only 0.0 allowed
 --keep X		  Include namespace into the feature set
 --interactions XYZ       Include namesapce interactions into the feature set
 --noconstant		  Do not add intercept
 --testonly		  Do not do learning, only prediction

Other known incompatibilities:
 - when not specifying either --keep or --interactions, vowpal will use all
input features. Fwumious will use none.
 - Example weight and individual feature weights passing via input examples is not
supported. Passing namespace weight is supported (|A:2 fature_value)

# vw_namspace_map.csv
It maps single letter namespaces to their full names. Its purpose is:
 - to disclose namespaces ahead of time
 - to map from namespace letters to their full names

