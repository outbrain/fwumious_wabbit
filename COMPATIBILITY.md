# Vowpal Wabbit compatibility

WARNING: Fwumious Wabbit cuts a lot of corners. Beware.

### Input file format
- [Vowpal Wabbit input format](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format) is supported
- Namespaces can only be single letters
- In each example each namespace can only be delcared once (and can have multiple features)
- there has to be a map file ("vw_namespace_map.csv") available with all the namespaces declared


### Command line arguments

#### Vowpal Wabbit compatibility
    --vwcompat   It causes Fwumious to complain if the arguments used for LR part would not
                 produce exactly the same results in vowpal wabbit.

#### Required when using "--vwcompat" to force
    --hash all   This treats all features as categorical,
                 Otherwise Vowpal Wabbit treats some as pre-hashed.

    --adaptive   Adagrad  mode
 
    --sgd        disable vowpal defaults of "--invariant" and "--normalize"
 

#### Optional
    --link logistic             Use logistic function for prediction printouts (always on)
 
    --loss_function logistic    Use logloss (always on)
 
    --power_t 0.5               Value for Adagrad's exponent (default 0.5 = square root)
 
    --l2 0.0                    L2 regularization, not supported. Only 0.0 allowed
 
    --keep X                    Include namespace into the feature set
 
    --interactions XYZ          Include namesapce interactions into the feature set
 
    --noconstant                Don't add intercept
 
    --testonly                  Don't learn, only predict
 

#### Other known incompatibilities and differences:
 - Fwumious Wabbit currently only supports log-loss for loss function
 - when not specifying either --keep or --interactions, Vowpal Wabbit will use all
input features. Fwumious Wabbit will use none.

#### vw_namspace_map.csv
It maps single letter namespaces to their full names. Its purpose is:
 - to disclose namespaces ahead of time
 - to map from namespace letters to their full names
Check out examples directory to see how it is formatted.
