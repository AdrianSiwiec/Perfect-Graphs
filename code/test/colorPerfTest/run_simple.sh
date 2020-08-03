#!/bin/bash
scriptDir=$(dirname $0)
testsDir="$scriptDir/tests"

# resultsDir="colorPerfResults/$(date '+%Y-%m-%d.%H%M.%S')"
# mkdir -p $resultsDir

colorExec="$scriptDir/../../obj/test/colorPerfTest/color.e"

colorTests=(
  # "small.t.in"
  "generated.t.in"
)

for file in "${colorTests[@]}"; do
  echo "Running Color on $file"
  $colorExec <$testsDir/$file
done