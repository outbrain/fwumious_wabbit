# Vowpal Wabbit compatibility

WARNING: Fwumious Wabbit cuts a lot of corners. Beware.

### Input file format
- Namespaces can only be single letters
- In each example each namespace can only be delcared once (and can have multiple features)
- there has to be a map file ("vw_namespace_map.csv") available with all namespaces declared


### Command line arguments
#### Required
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
 - Fwumious Wabbit is single core only

 - when not specifying either --keep or --interactions, Vowpal Wabbit will use all
input features. Fwumious Wabbit will use none.
 - the passing of example weight and individual feature weights via input examples is not
supported. Passing namespace weight is supported (|A:2 fature_value)

#### vw_namspace_map.csv
It maps single letter namespaces to their full names. Its purpose is:
 - to disclose namespaces ahead of time
 - to map from namespace letters to their full names