# Perfect-Graphs

This is a library that provides a ```O(n^9)``` method for testing if a graph is perfect, a Nvidia CUDA parallel method to do the same and a polynomial method coloring perfect graph using CSDP. See ```paper/main.pdf``` for description of the algorithms.

```isPerfectGraph(const Graph &G)``` in ```code/inc/perfect.h``` provides a method for perfect graph testing. See ```code/test/cuPerfTest/perfect.cpp``` for an example of use.

```color(const Graph &G)``` in ```code/inc/color.h``` provides a method for coloring perfect graphs. See ```code/test/colorPerfTest/color.cpp``` for an example of use.


## Building and testing instructions.

All source code is located in ```./code```

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
