#!/bin/bash
scriptDir=$(dirname $0)
testsDir="$scriptDir/tests"

resultsDir="colorPerfResults/$(date '+%Y-%m-%d.%H%M.%S')"
mkdir -p $resultsDir

colorExec="$scriptDir/../../obj/test/colorPerfTest/color.e"

colorTests=(
  # "perf.t.in"

  # "fullBinary20to45.t.in"
  # "grid5by4to9.t.in"
  # "hypercube20to40.t.in"
  # "knightGraph6by4to8.t.in"
  # "rookGraph6by4to8.t.in"
  # "biparite18to48.t.in"
  # "split.t.in"

  # "perfLin.t.in"
  # "lattice.t.in"
  # "rook.t.in"
  # "knight.t.in"
  # "hypercube.t.in"
  # "split.t.in"
  # "fullBinary.t.in"
  "tmp.t.in"
)

# for file in "${colorTests[@]}"; do
#   echo "Profiling Color on $file"
#   valgrind --callgrind-out-file=$resultsDir/callgrind.tmp.out --tool=callgrind $colorExec <$testsDir/$file >$resultsDir/$file.color.out
#   gprof2dot -f callgrind $resultsDir/callgrind.tmp.out >$resultsDir/$file.color.dot
#   rm $resultsDir/callgrind.tmp.out
# done

for file in "${colorTests[@]}"; do
  echo "Running Color on $file"
  $colorExec <$testsDir/$file >>$resultsDir/$file.out.csv
done