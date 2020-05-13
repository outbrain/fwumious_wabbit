perf record --call-graph dwarf,16384 -e cpu-clock -F 997 -- $1 && perf script \
| FlameGraph/stackcollapse-perf.pl | FlameGraph/rust-unmangle | FlameGraph/flamegraph.pl > flame.svg; firefox flame.svg
