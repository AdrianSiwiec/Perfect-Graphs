#include "commons.h"
#include "cuCommons.h"
#include "cuOddHoles.h"

bool cuContainsHoleOfSize(const CuGraph &G, int size, context_t &context) {
  int maxThreads = 100000;
  int k = G.n;
  int kCounter = 1;
  while (k * G.n < maxThreads && kCounter < G.n) {
    k *= G.n;
    kCounter++;
  }

  transform(
      [=] MGPU_DEVICE(int i) {
        if (i % 1000 == 0) {
          printf("%d\n", i);
        }
      },
      1000000, context);

  context.synchronize();
}