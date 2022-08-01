git clone git@github.com:brendangregg/FlameGraph.git
git clone git@github.com:Yamakaky/rust-unmangle.git
set -x
perf record --call-graph dwarf,16384 -e cpu-clock -F 997 -- "$@" && perf script \
| FlameGraph/stackcollapse-perf.pl | sed rust-unmangle/rust-unmangle | FlameGraph/flamegraph.pl > flame.svg && firefox flame.svg
