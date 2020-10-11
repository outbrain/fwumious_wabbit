# What makes Fwumious Wabbit (FW) fast?

# Strict focus

The biggest advantage that FW gets over VW and other logistic regression 
implementations is much narrower focus on what it does. Everything that 
would cause conditional jumps in inner loops is avoided or specialized 
using macros.

FW does not implement regularization nor multipass. They could be added 
without hurting performance in the fast path by using static traits.
Multipass can be simulated by saving final model and doing another run.

Compared to VW we also do not track prediction performance during the run - 
we do not continously compute logloss. Also we are interested in the 
predictions only on the evaluation part of the dataset, therefore a new 
parameter  --predictions-after allows for skipping outputing all predictions. 
We were surprised to learn that fromatting floating point for human readable 
output can take significant time compared to making the prediction itself. 

# Reduced flexibility in input formats

FW builds on VW's idea that you will likely run a lot of different models
over the same input data and it is worthwhile to parse it and save it in a
"cache". FW packs that cache format even more tightly than VW.

Generally FW supports a subset of VW's input format with more rigidness 
around namespace names. Namespaces can only be single letters and all of them 
used in an input file have to be listed ahead of time in a separate file 
(called vw_namespaces_map.csv).

# Carefully chosen external libraries

Benchmarking was done to pick the fastest gzip library for our use case
(Cloudflare's). For input cache file compression we use an extremely 
efficient LZ4 library (https://github.com/lz4/lz4). Deterministic random 
library is a Rust copy of Vowpal's method (merand48). Fasthash's murmur3 
algorithm is used for hashing to be compatible with Vowpal.

# Using Rust to an extreme

The code uses Rust macro's in order to create specialized code blocks and
thus avoids branching in inner loops. We are waiting for const generics to
hopefully use them in the future.

In core parts of parser, translator and regressor "unsafe" mode is used. 
We do that to avoid the need for bounds checking in inner loops and to 
avoid the need to initialize memory.

Some frequent codepaths were unrolled manually.

# Specialization
- We have specialized inner loops (with macros) for --ffm_k of 2, 4 and 8.
- Optimizer is specialized as part of the code, so inner-loop ifs are
avoided

# Algorithmic optimization

We heavily relied on ideas from VW and built on top of them. VW's buffer
management is fully replicated for logistic regression code.

FFM implementation in VW is not well tested and in our opinion it is 
buggy. We created a novel approach to FFM calculation. 
Traditional quadraple loop (which VW uses) was replaced by double loop. 
Intra-field combinations were allowed due to better prediction performance 
on our datasets and faster execution time (no conditional jumps in the inner 
loop).

We sum all changes to each feature weight in all FFM combinations and do the
final update of each feature weight only once per example. 

# Look up tables

A large speed boost comes in Adagrad from using look up table to 
map accumulated squared gradients to learning rate. A simple bit shift plus
lookup replaces power function (or sqrt) and multiplication. 
This removes two mathematical operations from inner loop and provides a
substantial speed boost.

# Code and data structures optimization

Examples cache file is really tightly packed, inspired by video codecs.
Similarly other data structures were carefully trimmed to avoid cache
trashing.


# Things we have tried which did not work
- Using data oriented programming approach. We've separated weights and 
accumulated gradients into separate vectors. This created 50% slowdown. The
theory is that we are doing lots of random memory accesses (to load weights)
and additional latencies overshadow benefits of (possibly) better
vectorization.

# Ideas for future speed improvements
- We know that using stack instead of heap for temporary buffer in learn() 
Gives us up to 10% speedup on FFMs. However the specialization code is ugly 
and therefore it is not on the main branch.
- Many more structures on stack. However it is hard to make it work without
crashing in extreme cases.
- On-demand specialization. This would requrie a compile-per-run, but could
really bring additional improvements
- Full unrolling of the double for loop for FFM would be possible.
- Compile for architecture of your chips, use vectorization.
export RUSTFLAGS="-C opt-level=3 -C target-cpu=skylake"
No measurable effect on laptop. Need to do further testing on server.
- Use vectorization
export RUSTFLAGS="-C opt-level=3 -C target-cpu=skylake -C llvm-args=--force-vector-width=4"
No measurable effect on laptop. Need to do further testing on server.
- Profile Guided Optimizations
We tried using PGO. The difference was unmeasurable - at most 0.5% speed
up, which is basically at the noise level. Given the complications of 
doing PGO builds it is simply not worth it.












