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
  # "perf2.t.in"
  # "perf.t.in"
  # "perfSmaller.t.in"
  # "bench.t.in"
  # "fullBinary.t.in"
  # "fullBinaryBig.t.in"
  # "grid6by5to8.t.in"
  # "knightGraph6by5to8.t.in"
  # "hypercubes2to5.t.in"
  # "kingGraph3by3to12.t.in"
  "rookGraph5by4to6.t.in"
)

naiveTests=(
  # "naive.t.in"
  "perfSmaller.t.in"
)

benchTests=(
  "bench.t.in"
)

# for file in "${benchTests[@]}"; do
#   echo "Running CuPerfect on $file"
#   $cuPerfectExec <$testsDir/$file >$resultsDir/$file.cuPerfect.out.csv
# done

for file in "${perfectTests[@]}"; do
  echo "Running Perfect on $file"
  $perfectExec <$testsDir/$file >>$resultsDir/$file.out.csv
done

for file in "${perfectTests[@]}"; do
  echo "Running CuPerfect on $file"
  $cuPerfectExec <$testsDir/$file >>$resultsDir/$file.out.csv
done

for file in "${perfectTests[@]}"; do
  echo "Running Naive on $file"
  $naiveExec <$testsDir/$file >>$resultsDir/$file.out.csv
done

# for file in "${perfectTests[@]}"; do
#   echo "Running CuPerfectNaive on $file"
#   $cuPerfectNaiveExec <$testsDir/$file >>$resultsDir/$file.out.csv
# done


# for file in "${naiveTests[@]}"; do
#   echo "Running CuPerfect on $file"
#   $cuPerfectNaiveExec <$testsDir/$file >$resultsDir/$file.out.csv
# done

