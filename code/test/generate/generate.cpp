#include <ctime>
#include <map>
#include <random>
#include "commons.h"
#include "oddHoles.h"
#include "perfect.h"
#include "testCommons.h"

int main() {
  srand(time(0));

  int minN = 10;
  int maxN = 19;

  for (int i = 0; i < 10; i++) {
    set<int> done;

    while (done.size() < maxN - minN + 1) {
      Graph G = getBipariteGraph(8 + getDistr() * 3, getDistr()).getLineGraph();
      if (rand() % 2) G = G.getComplement();

      if (G.n >= minN && G.n <= maxN && done.count(G.n) == 0) {
        G.printOut();
        done.insert(G.n);
      }
    }
  }
}