#!/bin/bash
scriptDir=$(dirname $0)
testsDir="$scriptDir/tests"

resultsDir="colorPerfResults/$(date '+%Y-%m-%d.%H%M.%S')"
mkdir -p $resultsDir

colorExec="$scriptDir/../../obj/test/colorPerfTest/color.e"

colorTests=(
  "small.t.in"
  # "perf.t.in"
)

for file in "${colorTests[@]}"; do
  echo "Profiling Color on $file"
  valgrind --callgrind-out-file=$resultsDir/callgrind.tmp.out --tool=callgrind $colorExec <$testsDir/$file >$resultsDir/$file.color.out
  gprof2dot -f callgrind $resultsDir/callgrind.tmp.out >$resultsDir/$file.color.dot
  rm $resultsDir/callgrind.tmp.out
done