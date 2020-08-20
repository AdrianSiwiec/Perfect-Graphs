#include <ctime>
#include <map>
#include <random>
#include "commons.h"
#include "oddHoles.h"
#include "perfect.h"
#include "testCommons.h"

#include "src/devCode.dev"

void testPerfectVsCuNaiveVsCuda() {
  Graph G = getRandomGraph(14 + getDistr() * 10, getDistr());
  testGraph(G, {algoPerfect, algoCudaNaive, algoCudaPerfect}, {nullptr, cuIsPerfectNaive, cuIsPerfect});
}

void testPerfectVsCuda() {
  Graph G = getRandomGraph(14 + getDistr() * 15, getDistr());
  testGraph(G, {algoPerfect, algoCudaPerfect}, {nullptr, cuIsPerfect});
}

void testPerfectVsCudaLine() {
  Graph G = getBipariteGraph(11 + getDistr() * 3, getDistr()).getLineGraph();
  testGraph(G, {algoPerfect, algoCudaPerfect}, {nullptr, cuIsPerfect});
}

// void testLineBiparite() {
//   Graph G = getBipariteGraph(7 + (getDistr() * 5), getDistr()).getLineGraph();
//   testGraph(G, true, true);
// }

// void testNonPerfect() {
//   Graph G = getNonPerfectGraph(5 + (rand() % 35) * 2, 10 + (getDistr() * 200), getDistr());
//   testGraph(G, false, true);
// }

int main() {
  init(true);
  while (1) {
    // testPerfectVsCuNaiveVsCuda();
    // testPerfectVsCuda();
    testPerfectVsCudaLine();
    // testLineBiparite();
    // testNonPerfect();
    StatsFactory::printStats2();
    cout << endl;
  }
}
