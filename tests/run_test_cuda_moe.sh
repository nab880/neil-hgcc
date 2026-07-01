#!/bin/sh
# Smoke test for the expert-parallel MoE demo (examples/mercury_moe): all-to-all
# dispatch/combine traffic with routing skew. Asserts the step completes and that
# increasing routing skew increases simulated time (more traffic concentrates on
# fewer links -> contention -> slower).
set -eu

SST_BIN="${SST_BIN:-sst}"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
MOE_PY="$TEST_DIR/../examples/mercury_moe/moe.py"
cd "$TEST_DIR"

sim_seconds() {
  grep "simulated time:" "$1" \
    | sed -n 's/.*simulated time: \([0-9.eE+-]*\) \([a-z]*\).*/\1 \2/p' | head -1 \
    | awk '{ u=$2;
        m=(u=="s")?1:(u=="ms")?1e-3:(u=="us")?1e-6:(u=="ns")?1e-9:(u=="ps")?1e-12:1;
        printf "%.12g", $1*m }'
}

# run <log> <NRANKS> <SKEW> <LAYERS> <LINK_BW>
run() {
  NRANKS="$2" SKEW="$3" LAYERS="$4" LINK_BW="$5" \
    "$SST_BIN" "$MOE_PY" > "$1" 2>&1
  grep -q "mercury_moe:.*done" "$1" \
    || { echo "FAIL: step did not complete ($1)"; cat "$1"; exit 1; }
}

run out_moe_uniform.log  4 0.0 2 100GB/s   # uniform routing
run out_moe_skewed.log   4 0.8 2 100GB/s   # heavy routing skew -> link contention

t_uniform=$(sim_seconds out_moe_uniform.log)
t_skewed=$(sim_seconds out_moe_skewed.log)
test -n "$t_uniform" && test -n "$t_skewed"

awk -v uni="$t_uniform" -v skew="$t_skewed" 'BEGIN{
  ok = 1
  # Skewed routing concentrates traffic -> contention -> slower than uniform.
  if (!(skew > uni)) {
    printf "FAIL: skewed (%g s) not > uniform (%g s)\n", skew, uni; ok = 0
  }
  if (ok) {
    printf "PASS: moe smoke -- uniform %g s, skew=0.8 %g s (%.2fx slower)\n",
           uni, skew, skew/uni
  } else { exit 1 }
}'
