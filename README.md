# Perfect-Graphs

## Building and testing instructions.

All source code is located in ./code

First line of the Makefile enables/disables CUDA. In order to run CUDA tests, a Nvidia graphics card and nvcc is required.

To build and run unit tests type:
```
cd code
make unitTest
```

To build and run performance tests for CCLSV, GPU CCLSV and naive type:
```
make cuPerfTest
```

```test/cuPerfTest/run.sh``` specifies tests that will be run.


To build and run performance tests for CSDP coloring type:
```
cd 3rdparty/Csdp/
make
cd ../..
make colorPerfTest
```

```test/colorPerfTest/run.sh``` specifies tests that will be run.
