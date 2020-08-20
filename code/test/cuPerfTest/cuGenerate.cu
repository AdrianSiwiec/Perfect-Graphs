#include <ctime>
#include <map>
#include <random>
#include "commons.h"
#include "oddHoles.h"
#include "perfect.h"
#include "testCommons.h"

#include "src/devCode.dev"

int main() {
  srand(time(0));

  set<int> done;

  int minN = 15;
  int maxN = 25;

  while (done.size() < maxN - minN + 1) {
    Graph G = getBipariteGraph(9 + getDistr() * 3, getDistr()).getLineGraph();

    if (G.n >= minN && G.n <= maxN && done.count(G.n) == 0) {
      cout << G << endl;
      done.insert(G.n);
    }
  }
}