#!/bin/sh
lli=${LLVMINTERP-lli}
exec $lli \
    /scratch/data/vrazavim/hls4ml/example-prjs/graph/gnn_simple/myproject_prj/solution1/.autopilot/db/a.g.bc ${1+"$@"}
