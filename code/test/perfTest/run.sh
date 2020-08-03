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
  "naive.t.in"
)

perfectTestsValgrind=(
  "perfVal.t.in"
)

naiveTestsValgrind=(
  "smallVal.t.in"
)

for file in "${perfectTests[@]}"; do
  echo "Running Perfect on $file"
  $perfectExec <$testsDir/$file >$resultsDir/$file.perfect.out
done

for file in "${naiveTests[@]}"; do
  echo "Running Naive on $file"
  $naiveExec <$testsDir/$file >$resultsDir/$file.naive.out
done

for file in "${perfectTestsValgrind[@]}"; do
  echo "Profiling Perfect on $file"
  valgrind --callgrind-out-file=$resultsDir/callgrind.tmp.out --tool=callgrind $perfectExec <$testsDir/$file >$resultsDir/$file.Valgrind.perfect.out
  gprof2dot -f callgrind $resultsDir/callgrind.tmp.out >$resultsDir/$file.perfect.dot
  rm $resultsDir/callgrind.tmp.out
done

for file in "${naiveTestsValgrind[@]}"; do
  echo "Profiling Naive on $file"
  valgrind --callgrind-out-file=$resultsDir/callgrind.tmp.out --tool=callgrind $naiveExec <$testsDir/$file >$resultsDir/$file.Valgrind.naive.out
  gprof2dot -f callgrind $resultsDir/callgrind.tmp.out >$resultsDir/$file.naive.dot
  rm $resultsDir/callgrind.tmp.out
done
