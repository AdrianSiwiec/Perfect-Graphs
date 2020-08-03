#!/bin/bash
scriptDir=$(dirname $0)
testsDir="$scriptDir/tests"

resultsDir="perfResults/$(date '+%Y-%m-%d.%H%M.%S')"
mkdir -p $resultsDir

perfectExec="$scriptDir/../../obj/test/perfTest/perfect.e"
naiveExec="$scriptDir/../../obj/test/perfTest/naive.e"

perfectTests=(
  "perf.t.in"
)

naiveTests=(
  "small.t.in"
)

for file in "${perfectTests[@]}"; do
  echo "Profiling Perfect on $file"
  valgrind --callgrind-out-file=$resultsDir/callgrind.tmp.out --tool=callgrind $perfectExec <$testsDir/$file >$resultsDir/$file.perfect.out
  gprof2dot -f callgrind $resultsDir/callgrind.tmp.out >$resultsDir/$file.perfect.dot
  rm $resultsDir/callgrind.tmp.out
done

for file in "${naiveTests[@]}"; do
  echo "Profiling Naive on $file"
  valgrind --callgrind-out-file=$resultsDir/callgrind.tmp.out --tool=callgrind $naiveExec <$testsDir/$file >$resultsDir/$file.naive.out
  gprof2dot -f callgrind $resultsDir/callgrind.tmp.out >$resultsDir/$file.naive.dot
  rm $resultsDir/callgrind.tmp.out
done