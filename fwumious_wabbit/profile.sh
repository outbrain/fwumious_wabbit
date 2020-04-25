perf record --call-graph dwarf,16384 -e cpu-clock -F 997 -- \
target/release/fw -b 24 --data bigdata.vw -l 0.1 --power_t 0.3 --interactions AB --keep A --keep B --keep C -p s --adaptive  --l2 3 -c && perf script \
| FlameGraph/stackcollapse-perf.pl | Flamegraph/rust-unmangle | FlameGraph/flamegraph.pl > flame.svg; firefox flame.svg
