#include "testCommons.h"
#include <cstdlib>
#include <iostream>
#include "commons.h"
#include "perfect.h"
// Unit test of testCommons

void testRandomGraphs() {
  auto graphs = getRandomGraphs(12, 1, 15);
  assert(graphs.size() == 15);
  for (int i = 0; i < 15; i++) {
    assert(graphs[i].n == 12);
    for (int v = 0; v < 12; v++) {
      for (int u = 0; u < 12; u++)
        if (u != v)
          assert(graphs[i].areNeighbours(u, v) == 1);
        else
          assert(graphs[i].areNeighbours(u, v) == 0);
    }
  }

  graphs = getRandomGraphs(15, 0.5, 15);
  for (int i = 0; i < 15; i++) {
    for (int v = 0; v < 15; v++) {
      for (int u = 0; u < 15; u++) {
        assert(graphs[i].areNeighbours(u, v) == graphs[i].areNeighbours(v, u));
      }
    }
  }
}

void testGetRandomPerfect() {
  for (int i = 0; i < 10; i++) {
    Graph G = getRandomPerfectGraph(6, 0.5);
    assert(isPerfectGraph(G));
  }
}

void testTuples() {
  assert(isAllZeros({0, 0, 0, 0}));
  assert(!isAllZeros({1, 0, 0, 0, 0, 0}));
  assert(!isAllZeros({0, 0, 0, 0, 0, 1}));

  assert(nextTuple({1, 0, 0, 1}, 3) == vec<int>({2, 0, 0, 1}));
  assert(nextTuple({2, 0, 0, 1}, 3) == vec<int>({0, 1, 0, 1}));

  assert(generateTuples(4, 3).size() == 3 * 3 * 3 * 3);
}

void testIsColoringValid() {
  Graph G(5,
          "\
  .XX..\
  X.X..\
  XX.XX\
  ..X.X\
  ..XX.\
  ");

  assert(isColoringValid(G, {2, 1, 0, 2, 1}));
  assert(!isColoringValid(G, {2, 1, 0, 2}));
  assert(!isColoringValid(G, {2, 1, 0, 2, 3}));
  assert(!isColoringValid(G, {1, 1, 0, 2, 3}));
}

int main() {
  init();
  testRandomGraphs();
  testGetRandomPerfect();
  testTuples();
  testIsColoringValid();
}
