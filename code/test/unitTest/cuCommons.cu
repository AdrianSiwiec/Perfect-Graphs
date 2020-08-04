#include "commons.h"
#include "cuCommons.h"
#include "testCommons.h"

void testCuGraph(context_t &context) {

  Graph G = getRandomGraph(10, 0.5);

  CuGraph CG(G, context);
  assert(CG.n == 10);

  int *matrix = new int[G.n * G.n];
  int *FN = new int[G.n];
  int *NN = new int[G.n * G.n];
  for (int i = 0; i < G.n; i++) {
    FN[i] = G.getFirstNeighbour(i);
    for (int j = 0; j < G.n; j++) {
      matrix[i * G.n + j] = G.areNeighbours(i, j);
      NN[i * G.n + j] = G.areNeighbours(i, j) ? G.getNextNeighbour(i, j) : -2;
    }
  }
  int *copiedToDevMatrix;
  int *copiedToDevFN;
  int *copiedToDevNN;

  CUCHECK(cudaMalloc((void **)&copiedToDevMatrix, sizeof(int) * G.n * G.n));
  CUCHECK(cudaMalloc((void **)&copiedToDevFN, sizeof(int) * G.n));
  CUCHECK(cudaMalloc((void **)&copiedToDevNN, sizeof(int) * G.n * G.n));
  CUCHECK(cudaMemcpy(copiedToDevMatrix, matrix, sizeof(int) * G.n * G.n, cudaMemcpyHostToDevice));
  CUCHECK(cudaMemcpy(copiedToDevFN, FN, sizeof(int) * G.n, cudaMemcpyHostToDevice));
  CUCHECK(cudaMemcpy(copiedToDevNN, NN, sizeof(int) * G.n * G.n, cudaMemcpyHostToDevice));

  delete[] matrix;
  delete[] FN;
  delete[] NN;

  transform(
      [=] MGPU_DEVICE(int i) {
        assert(CG.n == 10);

        for (int i = 0; i < CG.n; i++) {
          assert(CG.devFirstNeighbor[i] == copiedToDevFN[i]);

          for (int j = 0; j < CG.n; j++) {
            assert(CG.devMatrix[i * CG.n + j] == copiedToDevMatrix[i * CG.n + j]);

            assert(CG.devNextNeighbor[i * CG.n + j] == copiedToDevNN[i * CG.n + j]);
          }
        }
      },
      1, context);
  context.synchronize();

  CUCHECK(cudaFree(copiedToDevMatrix));
  CUCHECK(cudaFree(copiedToDevNN));
  CUCHECK(cudaFree(copiedToDevFN));
}

int main() {
  init();
  standard_context_t context(0);
  testCuGraph(context);
}