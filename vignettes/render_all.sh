#!/usr/bin/env bash
# Render every vignette (.qmd) to html + gfm + pdf, continuing on error.
# Logs a one-line PASS/FAIL per vignette to render_all.log.
set -u
cd "$(dirname "$0")"
log=render_all.log
: > "$log"
# Warm the compile cache BEFORE rendering. After a src/ change the first
# render otherwise recompiles the dependency stack inside the Quarto
# session, and Julia's "Precompiling packages..." progress (stderr) is
# captured into the rendered .md as if it were vignette output — seen on
# 01_getting_started, 18_dalton and 41_kan.
echo "=== warming compile cache ===" | tee -a "$log"
# Lux + FluxKAN are included so the PartiallySpecifiedModelsFluxKANExt
# package extension compiles here rather than inside 41_kan's render.
julia --project=. -e 'using PartiallySpecifiedModels, OrdinaryDiffEq, Plots, Random, Lux, FluxKAN' >>"$log" 2>&1
pass=0; fail=0
for d in [0-9][0-9]_*/ ; do
    qmd="${d}$(basename "$d").qmd"
    [ -f "$qmd" ] || continue
    echo "=== rendering $qmd ===" | tee -a "$log"
    if quarto render "$qmd" >>"$log" 2>&1 ; then
        echo "PASS $qmd" | tee -a "$log"
        pass=$((pass+1))
    else
        echo "FAIL $qmd" | tee -a "$log"
        fail=$((fail+1))
    fi
done
echo "DONE: $pass passed, $fail failed" | tee -a "$log"
