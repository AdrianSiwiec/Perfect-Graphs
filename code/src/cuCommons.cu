#include "commons.h"
#include "cuCommons.h"

void CudaAssert(cudaError_t error, const char *code, const char *file, int line) {
  if (error != cudaSuccess) {
    cerr << "Cuda error :" << code << ", " << file << ":" << error << endl;
    exit(1);
  }
}

void printArray(int *dev, int n, context_t &context) {
  transform(
      [=] MGPU_DEVICE(int i) {
        for (int a = 0; a < n; a++) {
          printf("%d", dev[a]);
        }
        printf("\n");
      },
      1, context);
  context.synchronize();
}

CuGraph::CuGraph(const Graph &G, context_t &context) : n(G.n), context(context) {
  CUCHECK(cudaMalloc((void **)&devMatrix, sizeof(int) * n * n));
  for (int i = 0; i < n; i++) {
    CUCHECK(cudaMemcpy(devMatrix + (i * n), G._matrix[i].data(), sizeof(int) * n, cudaMemcpyHostToDevice));
  }
}