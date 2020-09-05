#include <ctime>
#include <map>
#include <random>
#include "commons.h"
#include "oddHoles.h"
#include "perfect.h"
#include "testCommons.h"

int main() {
  srand(time(0));

  // int minN = 10;
  // int maxN = 19;

  // for (int i = 0; i < 10; i++) {
  //   set<int> done;

  //   while (done.size() < maxN - minN + 1) {
  //     Graph G = getBipariteGraph(8 + getDistr() * 3, getDistr()).getLineGraph();
  //     if (rand() % 2) G = G.getComplement();

  //     if (G.n >= minN && G.n <= maxN && done.count(G.n) == 0) {
  //       G.printOut();
  //       done.insert(G.n);
  //     }
  //   }
  // }

  // int minN = 30;
  // int maxN = 40;
  // for (int k = minN; k <= maxN; k++) {
  //   Graph G = getFullBinaryTree(k);
  //   G.printOut();
  // }

  // vec<vec<int>> sizes = {{5, 4}, {5, 5}, {5, 6}, {5, 7}};
  // for (auto size : sizes) {
  //   // Graph G = getCityGrid(size[0], size[1]);
  //   // Graph G = getKnightGraph(size[0], size[1]);
  //   Graph G = getRookGraph(size[0], size[1]);
  //   G.printOut();
  // }

  for (int k = 30; k <= 40; k++) {
    Graph G = getHypercube(k);
    G.printOut();
  }

  // for (int k = 3; k <= 12; k++) {
  //   Graph G = getGridWithMoves(3, k, {1, -1, 0, 0, 1, 1, -1, -1}, {0, 0, 1, -1, 1, -1, 1, -1});
  //   G.printOut();
  // }
}