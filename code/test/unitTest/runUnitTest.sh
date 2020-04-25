scriptDir=$(dirname $0)
unitTestDir="$scriptDir/../../obj/test/unitTest"

for test in "$unitTestDir"/*.e; do \
		echo "Running $(basename $test)" && \
		./$test || exit; \
	done &&	echo "ALL OK"