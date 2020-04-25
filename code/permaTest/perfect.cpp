#include "perfect.h"
#include "commons.h"
#include "oddHoles.h"
#include "testCommons.h"
#include <ctime>
#include <map>
#include <random>

void testPerfectVsNaive() {
  int r = rand() % 100;

  if (r == 0) {
    Graph G = getRandomGraph(9, getDistr());
    testGraph(G);
  } else {
    Graph G = getRandomGraph(8, getDistr());
    testGraph(G);
  }
}

void testLineBiparite() {
  Graph G = getBipariteGraph(6 + (getDistr()*5), getDistr()).getLineGraph();
  testGraph(G, true);
}

void testNonPerfect() {
  Graph G = getNonPerfectGraph(5 + (rand() % 7) * 2, 3 + (getDistr() * 20), getDistr());
  testGraph(G, false);
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