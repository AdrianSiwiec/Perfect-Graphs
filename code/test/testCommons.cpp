#include "testCommons.h"
#include "commons.h"
#include <cstdlib>
#include <iostream>


// Unit test of testCommons
int main() {
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
