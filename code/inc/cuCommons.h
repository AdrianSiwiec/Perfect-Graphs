#pragma once

#include <moderngpu/context.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/memory.hxx>

#include "commons.h"
using namespace mgpu;
using namespace std;

void CudaAssert(cudaError_t error, const char *code, const char *file, int line);
#define CUCHECK(x) CudaAssert(x, #x, __FILE__, __LINE__)

void printArray(int *dev, int n, context_t &context);

struct CuGraph {
  int n;
  context_t &context;

  int *devMatrix;

  CuGraph(const Graph &G, context_t &context);
};