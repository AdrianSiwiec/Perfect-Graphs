#include "perfect.h"
#include <ctime>
#include <map>
#include <random>
#include "commons.h"
#include "oddHoles.h"
#include "testCommons.h"

void testPerfectVsNaive() {
  int r = rand() % 20;

  if (r == 0) {
    Graph G = getRandomGraph(10, getDistr());
    testGraph(G, true);
  } else {
    Graph G = getRandomGraph(9, getDistr());
    testGraph(G, true);
  }
}

void testLineBiparite() {
  Graph G = getBipariteGraph(8 + (getDistr() * 5), getDistr()).getLineGraph();
  testGraph(G, true, true);
}

void testNonPerfect() {
  Graph G = getNonPerfectGraph(5 + (rand() % 15) * 2, 3 + (getDistr() * 60), getDistr());
  testGraph(G, false, true);
}

int main() {
  init(true);
  while (1) {
    testPerfectVsNaive();
    testLineBiparite();
    testNonPerfect();

    printStats();
  }
}
