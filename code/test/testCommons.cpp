#include "testCommons.h"
#include "commons.h"
#include <cstdlib>
#include <iostream>
// Unit test of testCommons

void testRandomGraphs() {
  auto graphs = getRandomGraphs(12, 1, 15);
  assert(graphs.size() == 15);
  for (int i = 0; i < 15; i++) {
    assert(graphs[i].n == 12);
    for (int v = 0; v < 12; v++) {
      for (int u = 0; u < 12; u++)
        if (u != v)
          assert(graphs[i][u][v] == 1);
        else
          assert(graphs[i][u][v] == 0);
    }
  }

  graphs = getRandomGraphs(15, 0.5, 15);
  for (int i = 0; i < 15; i++) {
    for (int v = 0; v < 15; v++) {
      for (int u = 0; u < 15; u++) {
        assert(graphs[i][u][v] == graphs[i][v][u]);
      }
    }
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

int main() {
  init();
  testRandomGraphs();
  testTuples();
}
