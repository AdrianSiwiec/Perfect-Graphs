#!/bin/bash
scriptDir=$(dirname $0)
testsDir="$scriptDir/tests"

resultsDir="cuPerfResults/$(date '+%Y-%m-%d.%H%M.%S')"
mkdir -p $resultsDir

perfectExec="$scriptDir/../../obj/test/cuPerfTest/perfect.e"
cuPerfectNaiveExec="$scriptDir/../../obj/test/cuPerfTest/cuPerfectNaive.e"
cuPerfectExec="$scriptDir/../../obj/test/cuPerfTest/cuPerfect.e"
naiveExec="$scriptDir/../../obj/test/cuPerfTest/naive.e"

perfectTests=(
  "perf.t.in"
  # "perfSmaller.t.in"
)

naiveTests=(
  "naive.t.in"
)

benchTests=(
  "bench.t.in"
)

for file in "${benchTests[@]}"; do
  echo "Running CuPerfect on $file"
  $cuPerfectExec <$testsDir/$file >$resultsDir/$file.cuPerfect.out.csv
done

# for file in "${perfectTests[@]}"; do
#   echo "Running Perfect on $file"
#   $perfectExec <$testsDir/$file >$resultsDir/$file.perfect.out.csv
# done

# for file in "${perfectTests[@]}"; do
#   echo "Running CuPerfect on $file"
#   $cuPerfectExec <$testsDir/$file >$resultsDir/$file.cuPerfect.out.csv
# done

# for file in "${perfectTests[@]}"; do
#   echo "Running CuPerfectNaive on $file"
#   $cuPerfectNaiveExec <$testsDir/$file >$resultsDir/$file.cuPerfectNaive.out.csv
# done


# for file in "${naiveTests[@]}"; do
#   echo "Running CuPerfect on $file"
#   $cuPerfectNaiveExec <$testsDir/$file >$resultsDir/$file.cuPerfectNaive.out.csv
# done

# for file in "${naiveTests[@]}"; do
#   echo "Running Naive on $file"
#   $naiveExec <$testsDir/$file >$resultsDir/$file.naive.out.csv
# done