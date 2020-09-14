#include <ctime>
#include <map>
#include <random>
#include "commons.h"
#include "oddHoles.h"
#include "perfect.h"
#include "testCommons.h"

int main() {
  srand(time(0));

  int minN = 50;
  int maxN = 50;
  int step = 5;

  for (int i = 0; i < 10; i++) {
    set<int> done;

    while (done.size() < (maxN - minN + step) / step) {
      Graph G = getBipariteGraph(8 + (rand() % 20), getDistr()).getLineGraph();
      // if (rand() % 2) G = G.getComplement();

      if ((G.n % step == 0) && G.n >= minN && G.n <= maxN && done.count(G.n) == 0) {
        G.printOut();
        done.insert(G.n);
      }
    }
  }

  // int minN = 18;
  // int maxn = 48;
  // int step = 6;
  // for (int i = 0; i < 10; i++) {
  //   for (int i = minN; i <= maxn; i += step) {
  //     auto G = getBipariteGraph(i, 0.5);
  //     G.printOut();
  //   }
  // }

  // int minN = 20;
  // int maxN = 45;
  // for (int k = minN; k <= maxN; k+=5) {
  // // for (int k = minN; k <= maxN; k++) {
  //   Graph G = getFullBinaryTree(k);
  //   G.printOut();
  // }

  // vec<vec<int>> sizes = {{5, 4}, {5, 5}, {5, 6}, {5, 7}, {5, 8}, {5, 9}};
  // vec<vec<int>> sizes = {{6, 4}, {6, 5}, {6, 6}, {6, 7}, {6, 8}};
  // vec<vec<int>> sizes = {{6,14}, {6, 15}, {6, 16}, {6, 17}};
  // vec<vec<int>> sizes = {{5, 4}, {5, 5}, {5, 6}, {5, 7}};
  // vec<vec<int>> sizes = {{5, 10}, {5, 11}, {5, 12} };
  // for (auto size : sizes) {
  //   // Graph G = getCityGrid(size[0], size[1]);
  //   // Graph G = getKnightGraph(size[0], size[1]);
  //   Graph G = getRookGraph(size[0], size[1]);
  //   G.printOut();
  // }

  // for (int k = 60; k <= 75; k += 5) {
  //   Graph G = getHypercube(k);
  //   G.printOut();
  // }

  // for (int k = 3; k <= 12; k++) {
  //   Graph G = getGridWithMoves(3, k, {1, -1, 0, 0, 1, 1, -1, -1}, {0, 0, 1, -1, 1, -1, 1, -1});
  //   G.printOut();
  // }

  // for (int k = 0; k < 10; k++) {
  //   for (int i = 24; i <= 30; i += 3) {
  //     Graph G = getSplitGraph(i, getDistr());
  //     G.printOut();
  //   }
  // }
}