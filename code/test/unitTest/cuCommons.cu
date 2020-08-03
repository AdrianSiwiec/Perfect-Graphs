#include "commons.h"
#include "cuCommons.h"
#include "testCommons.h"

void testCuGraph(context_t &context) {
  Graph G = getRandomGraph(10, 0.5);

  CuGraph CG(G, context);
  assert(CG.n == 10);

  int *matrix = new int[G.n * G.n];
  for (int i = 0; i < G.n; i++) {
    for (int j = 0; j < G.n; j++) {
      matrix[i * G.n + j] = G.areNeighbours(i, j);
    }
  }
  int *copiedToDevMatrix;
  CUCHECK(cudaMalloc((void **)&copiedToDevMatrix, sizeof(int) * G.n * G.n));
  CUCHECK(cudaMemcpy(copiedToDevMatrix, matrix, sizeof(int) * G.n * G.n, cudaMemcpyHostToDevice));

  printArray(copiedToDevMatrix, CG.n * CG.n, context);
  printArray(CG.devMatrix, CG.n * CG.n, context);

  transform(
      [=] MGPU_DEVICE(int i) {
        assert(CG.n == 10);

        for (int i = 0; i < CG.n; i++) {
          for (int j = 0; j < CG.n; j++) {
            assert(CG.devMatrix[i * CG.n + j] == copiedToDevMatrix[i * CG.n + j]);
          }
        }
      },
      1, context);
  context.synchronize();
}

int main() {
  init();
  standard_context_t context(0);
  testCuGraph(context);
}