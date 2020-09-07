#pragma once

#include <moderngpu/context.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/memory.hxx>

#include "commons.h"
using namespace mgpu;
using namespace std;

void CudaAssert(cudaError_t error, const char *code, const char *file, int line);
#define CUCHECK(x) CudaAssert(x, #x, __FILE__, __LINE__)

#define CUDA_MAX_N 100

void printArray(int *dev, int n, context_t &context);

struct CuGraph {
  int n;
  context_t &context;

  // of size n*n
  int *devMatrix;
  // of size n
  int *devFirstNeighbor;
  // of size n*n
  int *devNextNeighbor;

  CuGraph(const Graph &G, context_t &context);
  void deleteCuGraph();  // We cannot use destructor, because moderngpu transforms capture by value
};