Fwumious Wabbit is
- a very fast machine learning tool
- built with Rust
- inspired by and partially compatible with Vowpal Wabbit (much love! read more about compatibility [here](COMPATIBILITY.md))
- currently supports logistic regression and field-aware factorization machines

Fwumious Wabbit is actively used in Outbrain for off-line research, as well as for some production flows. It 
enables "high bandwidth research" when doing feature engineering, feature 
selection, hyper-parameter tuning and the like. 

Data scientists can train hundreds of models over hundreds of millions of examples in 
a matter of hours on a single machine.

For our tested scenarios it is two orders of magnitude faster than the fastest 
Tensorflow implementation we could come up with (faster even when using GPUs with TensorFlow). 
It is an order of magnitude faster than Vowpal Wabbit for some specific use-cases.

check out our [benchmark](BENCHMARK.md), here's a teaser:

![benchmark results](benchmark_results.png)


**Why is it faster?** (see [here](SPEED.md) for more details)
- tighter encoding format for examples cache
- namespaces have to be declared up-front, so tight encoding is possible
- using lz4 for examples cache compression instead of gz
- using Look Up Tables for AdaGrad
- inner loop is single purpose and super-optimized
- it is written in Rust and it uses specialization tricks (via macros)
- it cuts corners by preallocating buffers
- a lot of profiling and optimizing the bottlenecks

