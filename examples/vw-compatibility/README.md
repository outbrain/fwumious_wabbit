# Vowpal Wabbit compatibility

Under certain circumstances Fwumious Wabbit makes bit-by-bit the same 
predictions as Vowpal Wabbit. This is achieved because FW uses the same
hashing mechanisms with the same constants as VW.

This mode can be turned on by --vwcompat. This mode is slower as it
does not use Adagrad look up tables. 

Compatibility also isn't perfect. There are multiple edge cases.

