# What makes Fwumious Wabbit (FW) so fast?

# Strict focus

The biggest advantage that FW gets over VW is much narrower focus on what it
does. Everything that would cause conditional jumps in inner loops is
avoided or specialized using macros.

As an example FW does not implement regularization nor multipass, as those
aren't top features needed in our use case at Outbrain. However they could
be easily added without hurting performance in the fast path.

Compared to VW we also do not track prediction performance during the run - 
we do not continously compute logloss. Also we are interested in the 
predictions only o n the evaluation part of the dataset, therefore a new 
parameter  --predictions-after allows for skipping outputing all predictions. 
We were surprised to learn that fromatting floating point for human readable 
output can take significant time compared to making the prediction itself. 

# Reduced flexibility in input formats

FW builds on VW's idea that you will likely run a lot of different models
over the same input data and it is worthwhile to parse it and save it in a
"cache". FW packs that cache format even more tightly than VW.

Generally FW supports a subset of VW's input format with special rigidness 
around namespaces. Namespaces can only be single letters and all of them used 
in an input file have to be listed ahead of time in a separate file (called
vw_namespaces_map.csv).

# Carefully chosen external libraries

Benchmarking was done to pick the fastest gzip library for our use case
(Cloudflare's). Examples cache file FW uses an extremely efficient LZ4
library (https://github.com/lz4/lz4). Deterministic random library is a Rust
copy of Vowpal's method (merand48). Fasthash's murmur3 algorithm is used for
hashing to be compatible with Vowpal.

# Using Rust to an extreme

The code uses Rust macro's in order to create specialized code blocks and
thus avoids branching in inner loops. We are waiting for const generics to
hopefully use them in the future.

In core parts of parser, translator and regressor "unsafe" mode
is used, mainly to avoid the need to to avoid bounds checking in
inner loops and to avoid the need to initialize memory at declaration. 

Some frequent codepaths were unrolled manually.

# Algorithmic optimization

We heavily relied on ideas from VW and built on top of them. VW's buffer
management is fully replicated for logistic regression code.

However FFM implementation in VW is not well tested and is in our opinion 
buggy. We created a novel approach to FFM calculation that allows for fields 
that are filled from multiple features and those can be multi valued. 
Traditional quadraple loop to handle such cases (which also VW uses) was 
replaced by double loop with minimal logic to avoid self-combinations.

We sum all changes to each feature weight in all FFM combinations and do the
final update of each feature weight only once per example. 

A large speed boost comes in Adagrad from using look up table to 
map accumulated squared gradients to learning rate. A simple bit shift plus
lookup basically replaces power function and multiplication (or sqrt if 
power_t is 0.5). This look up table had almost no effect on performance 
of our large scale machine learning tasks. And it removes an expensive
function from inner loop.

# Code and data structures optimization

Examples cache file is really tightly packed, inspired by video codecs.
Similarly other data structures were carefully trimmed to avoid cache
trashing.

Buffers are allocated statically on stack. There are some bound checks and
if they fail FW will exit.
