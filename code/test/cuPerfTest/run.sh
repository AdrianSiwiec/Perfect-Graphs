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
  # # "bench.t.in"
  # # "fullBinaryBig.t.in"
  # # "knightGraph6by5to8.t.in"
  # "hypercubes16to50.t.in"
  # # "kingGraph3by3to12.t.in"

  # "perf2.t.in"
  # "perf.t.in"
  # "fullBinary20to100.t.in"
  # "grid6by5to11.t.in"
  # "hypercubes20to55.t.in"
  # "knightGraph8by4to8.t.in"
  # "rookGraph5by4to7.t.in"
  # "split20to50.t.in"
  # "biparite.t.in"
)

pefectSupplement=(
  # "grid6by5to11.t.in"
)

cudaPerfectSupplement=(
  # "grid6by5to11.t.in"
  # "hypercubes20to55.t.in"
  # "rookGraph5by4to7.t.in"
  # "split20to50.t.in"
  # "knightGraph8by4to8.t.in"
  # "perf2.t.in"
  "biparite.t.in"
)

naivePerfectSupplement=(
  # "rookGraph5by4to7.t.in"
  # "split20to50.t.in"
  # "perf2.t.in"
  "biparite.t.in"
)

# naiveTests=(
#   # "naive.t.in"
#   "perfSmaller.t.in"
# )

# benchTests=(
#   "hypercubes30to40.t.in"
# )


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
#   echo "Running CuNaive on $file"
#   $cuPerfectNaiveExec <$testsDir/$file >>$resultsDir/$file.out.csv
# done

# for file in "${pefectSupplement[@]}"; do
#   echo "Running Perfect Sup on $file"
#   $perfectExec <$testsDir/perfectSup/$file >>$resultsDir/$file.out.csv
# done

for file in "${cudaPerfectSupplement[@]}"; do
  echo "Running CuPerfect Sup on $file"
  $cuPerfectExec <$testsDir/cuPerfectSup/$file >>$resultsDir/$file.out.csv
done

for file in "${naivePerfectSupplement[@]}"; do
  echo "Running Naive Sup on $file"
  $naiveExec <$testsDir/naiveSup/$file >>$resultsDir/$file.out.csv
done

# for file in "${naivePerfectSupplement[@]}"; do
#   echo "Running Cu Naive Sup on $file"
#   $cuPerfectNaiveExec <$testsDir/naiveSup/$file >>$resultsDir/$file.out.csv
# done



# for file in "${benchTests[@]}"; do
#   echo "Running Naive on $file"
#   $naiveExec <$testsDir/$file >$resultsDir/$file.out.csv
# done


# for file in "${perfectTests[@]}"; do
#   echo "Running CuPerfectNaive on $file"
#   $cuPerfectNaiveExec <$testsDir/$file >>$resultsDir/$file.out.csv
# done


# for file in "${naiveTests[@]}"; do
#   echo "Running CuPerfect on $file"
#   $cuPerfectNaiveExec <$testsDir/$file >$resultsDir/$file.out.csv
# done

