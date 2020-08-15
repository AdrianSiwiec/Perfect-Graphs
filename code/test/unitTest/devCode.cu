#include "commons.h"
#include "cuCommons.h"
#include "cuOddHoles.h"
#include "testCommons.h"

#include "src/devCode.dev"

void testPreparePathStart(context_t &context) {
  int *dev;
  CUCHECK(cudaMalloc(&dev, sizeof(int) * 5));

  transform(
      [=] MGPU_DEVICE(int id) {
        devPreparePathStart((7 * 11 + 3) * 11 + 5, 3, 11, dev);
        assert(dev[0] == 5);
        assert(dev[1] == 3);
        assert(dev[2] == 7);
        assert(dev[3] == 0);

        devPreparePathStart(((7 * 11 + 3) * 11 + 5) * 11 + 10, 4, 11, dev);
        assert(dev[0] == 10);
        assert(dev[1] == 5);
        assert(dev[2] == 3);
        assert(dev[3] == 7);
        assert(dev[4] == 0);

        devPreparePathStart(7 * 7 * 7 * 7 - 1, 4, 7, dev);
        assert(dev[0] == 6);
        assert(dev[1] == 6);
        assert(dev[2] == 6);
        assert(dev[3] == 6);
        assert(dev[4] == 0);

        devPreparePathStart(7, 4, 7, dev);
        assert(dev[0] == 0);
        assert(dev[1] == 1);
        assert(dev[2] == 0);
        assert(dev[3] == 0);
        assert(dev[4] == 0);
      },
      1, context);

  context.synchronize();
  CUCHECK(cudaFree(dev));
}

void testDevAreNeighbors(context_t &context) {
  Graph G = getRandomGraph(10, 0.5);
  CuGraph CG(G, context);

  int *ans = new int[100];
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      ans[i * 10 + j] = G.areNeighbours(i, j);
    }
  }

  int *devAns;
  CUCHECK(cudaMalloc(&devAns, sizeof(int) * 100));
  CUCHECK(cudaMemcpy(devAns, ans, sizeof(int) * 100, cudaMemcpyHostToDevice));

  free(ans);

  transform(
      [=] MGPU_DEVICE(int id) {
        for (int i = 0; i < 10; i++) {
          for (int j = 0; j < 10; j++) {
            assert(devAns[i * 10 + j] == devAreNeighbors(CG, i, j));
          }
        }
      },
      1, context);

  context.synchronize();

  CUCHECK(cudaFree(devAns));
}

void testDevIsDistinctValues(context_t &context) {
  transform(
      [=] MGPU_DEVICE(int id) {
        int dev[10];
        dev[0] = 0;
        dev[1] = 1;
        dev[2] = 2;
        dev[3] = 2;

        assert(devIsDistinctValues(dev, 3));
        assert(!devIsDistinctValues(dev, 4));
        assert(devIsDistinctValues(dev, 1));
        assert(devIsDistinctValues(dev, 0));
      },
      1, context);
}

void testCuOddHole(context_t &context) {
  Graph G = getRandomGraph(11, 0.5);

  CuGraph CG(G, context);

  cuContainsHoleOfSize(CG, 9, context);

  assert(false);
}

int main() {
  init();
  standard_context_t context(0);
  testPreparePathStart(context);
  testDevAreNeighbors(context);
  testDevIsDistinctValues(context);
  testCuOddHole(context);

  context.synchronize();
}