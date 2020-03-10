#include "commons.h"
#include <cassert>
#include <iostream>

using namespace std;

int main() {
  Graph g(10);
  assert(g.n == 10);
  for (int i = 0; i < 10; i++) {
    assert(g[i].size() == 10);
  }

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      g[i][j] = i + j;
    }
  }

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      assert(g[i][j] == i + j);
    }
  }
}