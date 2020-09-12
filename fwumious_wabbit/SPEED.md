# What makes Fwumious Wabbit (FW) so fast?

a) FW is strictly focused on two machine learning methods: logistic 
regression and field-aware factorization machines. This allows lots of 
shortcuts.

b) FW only supports logistic function. It does not support anything else.
This would be rather easy expansion without hurting speed, but it has not
been done yet.

c) All external libraries that were chosen were benchmarked for speed and
best option for specific workloads was always chosen. For example
Cloudflare's gzip library and lz4 for internal cache compression.

d) Rust is by nature fast. We use a lot of rust features to support
specialization, like macros and statically compiled traits.

e) Careful attention was paid to internal data structures organization,
always choosing the fastest approach (via benchmarking).

f) There are some novel (compared to VW) approaches, like look up table for
Adagrad learning rate and novel approach in how to organize data structures
and algorithm for field-aware factorization machines.

g) There's lots of functionality missing - no L2 regularization, no
multipass support

# Why is it faster?
- tighter encoding format for examples cache
- using lz4 for examples cache compression instead of gz
- using Look Up Tables for AdaGrad (--fastmath option)
- inner loop is single purpose and super-optimized
- it is written in Rust and it uses specialization tricks (via macros)
- it cuts corners by preallocating buffers and not doing bound checking
- a lot of profiling and optimizing the bottlenecks

