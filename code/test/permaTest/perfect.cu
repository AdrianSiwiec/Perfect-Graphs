#include <ctime>
#include <map>
#include <random>
#include "commons.h"
#include "oddHoles.h"
#include "perfect.h"
#include "testCommons.h"

#include "src/devCode.dev"

void testPerfectVsCuNaiveVsCuda() {
  Graph G = getRandomGraph(14 + getDistr() * 10, getDistrWide());
  testGraph(G, {algoPerfect, algoCudaNaive, algoCudaPerfect, algoNaive},
            {nullptr, cuIsPerfectNaive, cuIsPerfect});
}

void testPerfectVsCuda() {
  Graph G = getRandomGraph(14 + getDistrWide() * 15, getDistrWide());
  testGraph(G, {algoPerfect, algoCudaPerfect, algoNaive}, {nullptr, cuIsPerfect});
}

void testPerfectVsCudaLine() {
  Graph G = getBipariteGraph(11 + getDistrWide() * 3, getDistrWide()).getLineGraph();
  cerr << G.n << endl;
  testGraph(G, {algoPerfect, algoCudaPerfect}, {nullptr, cuIsPerfect});
}

// void testLineBiparite() {
//   Graph G = getBipariteGraph(7 + (getDistr() * 5), getDistr()).getLineGraph();
//   testGraph(G, true, true);
// }

void testNonPerfect() {
  Graph G = getNonPerfectGraph(31 + (rand() % 35) * 2, 10 + (getDistr() * 200), getDistr());
  assert(!testGraph(G, {algoNaive, algoCudaNaive}, {nullptr, cuIsPerfectNaive}));

  G = getBipariteGraph(11 + getDistrWide() * 3, getDistrWide()).getLineGraph();
  assert(testGraph(G, {algoNaive, algoCudaNaive}, {nullptr, cuIsPerfectNaive}));
}

int main() {
  init(true);
  while (1) {
    // testPerfectVsCuNaiveVsCuda();
    // testPerfectVsCuda();
    testPerfectVsCudaLine();
    // testLineBiparite();
    testNonPerfect();
    StatsFactory::printStats2();
    cout << endl;
  }
}
