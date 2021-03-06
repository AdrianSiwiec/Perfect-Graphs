# Perfect-Graphs

This is a library that provides a ```O(n^9)``` method for testing if a graph is perfect, a Nvidia CUDA parallel method to do the same and a polynomial method coloring perfect graph using CSDP. See ```paper/main.pdf``` for description of the algorithms.

All of the source code and tests are located in ```./code```. Please run commands mentioned below from this directory.

```isPerfectGraph(const Graph &G)``` in ```code/inc/perfect.h``` provides a method for perfect graph testing. See ```code/test/cuPerfTest/perfect.cpp``` for an example of use.

```color(const Graph &G)``` in ```code/inc/color.h``` provides a method for coloring perfect graphs. See ```code/test/colorPerfTest/color.cpp``` for an example of use.


## Building and testing instructions.

First line of the Makefile enables/disables CUDA. In order to run CUDA tests, a Nvidia graphics card and nvcc is required.

Cloning submodules is required before building. Type ```git submodule init; git submodule update``` to do this.

First, you need to build the CSDP library used for the coloring algorithm. To do this type:
```
cd 3rdparty/Csdp/
make
cd ../..
```

To build and run unit tests type: (if you do not have CUDA installed, disabling it in the Makefile is needed)
```
cd code
make unitTest
```

To build and run performance tests for CCLSV, GPU CCLSV and naive type:
```
make cuPerfTest
```

```test/cuPerfTest/run.sh``` specifies tests that will be run. By default it runs on a dataset similar to the one used in the thesis and takes a couple of hours to complete. By default the results will be in ```cuPerfResults/<datetime of the test>/<test filename>.out```

If you do not have a CUDA gpu and wish to only run CPU tests type:
```
make perfTest
```

```test/perfTest/run.sh``` specifies tests that will be run. By default it runs on a dataset similar to the one used in the thesis and takes a couple of hours to complete. By default the results will be in ```perfResults/<datetime of the test>/<test filename>.out```

To build and run performance tests for CSDP coloring type:
```
make colorPerfTest
```

```test/colorPerfTest/run.sh``` specifies tests that will be run. By default it runs on a dataset similar to the one used in the thesis and takes about an hour to complete. By default the results will be in ```colorPerfResults/<datetime of the test>/<test filename>.out```
