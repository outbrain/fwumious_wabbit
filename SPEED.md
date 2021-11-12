# What Makes Fwumious Wabbit (FW) Fast?

# Strict Focus

The biggest advantage that FW has over VW and other logistic regression 
implementations is much narrower focus on what it does. Everything that 
would cause conditional jumps in inner loops is avoided or specialized 
using macros.

FW does not implement regularization nor multipass. They could be added 
without hurting performance in the fast path by using static traits.
Multipass can be done with an external tool by saving the final model and 
then doing another pass with that as an initial model.

Unlike VW we also do not track prediction performance during the run - 
we do not continuously compute logloss. Additionally we are interested in 
the predictions only on the evaluation part of the dataset, therefore a 
new parameter  --predictions-after allows for skipping outputting all 
predictions. We were surprised to learn that formatting floating point 
values for human readable output can take significant time compared to 
making the prediction itself. 

# Reduced Flexibility in Input Formats

FW builds on VW's idea that you will likely run a lot of different models
over the same input data, so it is worthwhile to parse it and save it in a
"cache". FW packs that cache format even more tightly than VW.

Generally FW supports a subset of VW's input format with more rigidity 
around namespace names. Namespaces can only be single letters and those
that are used in an input file must be listed ahead of time in a separate 
file (called vw_namespaces_map.csv).

# Carefully Chosen External Libraries

Benchmarking was done to pick the fastest gzip library for our use case
(Cloudflare's). For input cache file compression we use an extremely 
efficient LZ4 library (https://github.com/lz4/lz4). The deterministic random 
library is a Rust copy of Vowpal's method (merand48). Fasthash's murmur3 
algorithm is used for hashing to be compatible with Vowpal.

# Using Rust to an Extreme

The code uses Rust macros in order to create specialized code blocks and
thus avoids branching in inner loops. We are waiting for const generics to
hopefully use them in the future.

Core parts of the parser, translator and regressor use the "unsafe" mode. 
This is done in order avoid the need for bounds checking in inner loops and 
the need to initialize memory.

Some frequent codepaths were unrolled manually.

# Specialization
- We have specialized inner loops (with macros) for --ffm_k of 2, 4 and 8.
- Optimizer is specialized as part of the code, so inner-loop ifs are
avoided
- In FFM we optimize situations where feature values (not weights) are 1.0.
Three different multiplications by 1.0 within the loop that iterates ffm_k 
times are avoided with, 4% speedup.

# Algorithmic Optimization

We relied heavily on ideas from VW and built on top of them. VW's buffer
management is fully replicated for logistic regression code.

The FFM implementation in VW is not well tested and in our opinion it is 
buggy. We created a novel approach to FFM calculation as follows. 
Traditional quadruple loop (which VW uses) was replaced by a double loop. 
Intra-field combinations were allowed due to better prediction performance 
on our datasets and a faster execution time (no conditional jumps in the 
inner loop).

We sum all changes to each feature weight in all FFM combinations and do the
final update of each feature weight only once per example. 

# Look Up Tables

A large speed boost comes in Adagrad from using a look-up table to 
map accumulated squared gradients to the learning rate. A simple bit shift 
plus lookup replaces use of the power function (or sqrt) and multiplication. 
This removes two mathematical operations from inner loop and provides a
substantial speed boost.

# Code and Data Structures Optimization

The examples cache file is very tightly packed, inspired by video codecs.
Similarly other data structures were carefully trimmed to avoid cache
thrashing.


# Prefetching
When sensible, we prefetch the weight ahead of the for-loop. Since weights
basically cause  random memory access and modern machines are effectively
NUMA, this helps a bit.

# Compiling for Your Architecture
We are compiling our code with 
```
export RUSTFLAGS="-C opt-level=3 -C target-cpu=skylake"
```
This provides a speed improvement of about 5%.

# Using Stack for Temporary Buffer
We use a fixed size stack for the FFM temporary buffer, and when the buffer 
is bigger than the fixed sized stack, we use a heap-allocated buffer. Our 
code is entirely specialized on each codepath. Surprisingly we saw a 5%+ 
speedup when using the stack. Our belief is that there are more optimization 
opportunities if we make all memory addresses static, however that is really 
hard to achieve without per-run recompilation of rust code.

# Things we Have Tried
- Using a data oriented programming approach: We've separated weights and 
accumulated gradients into separate vectors. This caused a 50% slowdown. The
theory is that we are doing lots of random memory accesses (to load weights)
and additional latencies overshadow the benefits of (possibly) better
vectorization.
- Manually rolled-out AVX2 code for LR: While on paper instructions take 
less time to execute than LLVM code, in practice there is no difference due
to the floating point operations not being the bottleneck - it looks like
the bottleneck is delivering values from memory.


# Ideas for Future Speed Improvements
- On-demand specialization. This would require a compile-per-run, however
everything we have learned indicates that this would bring an additional 
speed boost.
- Use vectorization
export RUSTFLAGS="-C opt-level=3 -C target-cpu=skylake -C llvm-args=--force-vector-width=4":
We saw no measurable effect on laptop. Need to do further testing on server.
- Profile Guided Optimizations
We tried using PGO. The difference was unmeasurable - at most 0.5% speed
up, which is basically at the noise level. Given the complications of 
doing PGO builds it is simply not worth it.












