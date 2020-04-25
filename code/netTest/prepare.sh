#!/bin/bash

scriptDir=$(dirname $0)
downloadedDir="$scriptDir/downloaded"

# Showg is a tool to parse g6 coding of graphs
showgAddress="https://users.cecs.anu.edu.au/~bdm/data/showg_versions/showg_linux64"
showgPath="$downloadedDir/show_linux64"

testAddresses=(
  "https://users.cecs.anu.edu.au/~bdm/data/perfect5.g6"
  "https://users.cecs.anu.edu.au/~bdm/data/perfect6.g6"
  "https://users.cecs.anu.edu.au/~bdm/data/perfect7.g6"
  "https://users.cecs.anu.edu.au/~bdm/data/perfect8.g6"
  "https://users.cecs.anu.edu.au/~bdm/data/perfect9.g6"
  "https://users.cecs.anu.edu.au/~bdm/data/perfect10.g6"
  # "https://users.cecs.anu.edu.au/~bdm/data/perfect11.g6.gz"
)

testerExecPath="$scriptDir/../obj/netTest/perfect.e"

mkdir -p $downloadedDir

if [ ! -e $showgPath ]
then
  echo "Downloading showg tool"
  wget $showgAddress -O $showgPath
  chmod +x $showgPath
fi

for test in "${testAddresses[@]}"
do 
  testPath=$downloadedDir/$(basename $test)
  if [ ! -e $testPath ]
  then
    echo "Downloading $testPath"
    wget $test -O $testPath
  fi

  testDecodedPath=$testPath.t.in
  if [ ! -e $testDecodedPath ]
  then 
    echo "Converting $testPath"
    $showgPath -aq $testPath $testDecodedPath
  fi
done 

echo "Preparing net tests done!"
echo "Runing net tests:"

for test in "${testAddresses[@]}"
do
  testPath=$downloadedDir/$(basename $test)
  testDecodedPath=$testPath.t.in
  $testerExecPath <$testDecodedPath
done 

echo "Net tests done!"
