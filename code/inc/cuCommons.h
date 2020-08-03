#include "commons.h"

#include <moderngpu/context.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/memory.hxx>
using namespace mgpu;

#define CUCHECK(x) CudaAssert(x, #x, __FILE__, __LINE__)

struct CuGraph {
  int n;
  context_t &context;

  int *devMatrix;

  CuGraph(const Graph &G, context_t &context);
}