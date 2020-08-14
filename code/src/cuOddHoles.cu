#include "commons.h"
#include "cuCommons.h"
#include "cuOddHoles.h"

// __device__ int get42() { return 42; }

__device__ void preparePathStart(int code, int len, int n, int *devPath) {
  int tmp = 0;

  while (tmp < len) {
    devPath[tmp] = code % n;
    code /= n;
    tmp++;
  }
}

__device__ void devPrintArray(int *dev, int n) {
  printf("[");
  for (int a = 0; a < n; a++) {
    printf("%d", dev[a]);
    if (a + 1 < n) printf(", ");
  }
  printf("]\n");
}

__device__ bool devAreNeighbors(const CuGraph &G, int a, int b) { return G.devMatrix[a * G.n + b]; }

__device__ bool devIsDistinctValues(const CuGraph &G, int *path, int length) {
  
}

__device__ bool devIsAPath(const CuGraph &G, int *path, int length, bool isCycleOk = false,
                           bool areChordsOk = false) {
  if (length <= 0) return false;
}

bool cuContainsHoleOfSize(const CuGraph &G, int size, context_t &context) {
  static const int max_size = 30;

  assert(max_size > size);

  int maxThreads = 100000;
  int k = G.n;
  int kCounter = 1;
  while (k * G.n <= maxThreads && kCounter < size) {
    k *= G.n;
    kCounter++;
  }

  cout << kCounter << endl;
  cout << k << endl;

  transform(
      [=] MGPU_DEVICE(int id) {
        int path[max_size];
        preparePathStart(id, kCounter, G.n, path);

        for (int i = 1; i < kCounter; i++) {
          if (!devAreNeighbors(G, path[i], path[i - 1])) return;
        }
      },
      k, context);

  context.synchronize();
}