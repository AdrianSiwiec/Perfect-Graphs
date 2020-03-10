#include "commons.h"
#include "testCommons.h"
#include <cassert>
#include <iostream>

using namespace std;

void testGraph() {
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

void testGetTriangles() {
  for (int i = 0; i < 15; i++) {
    Graph G = getRandomGraph(10, 0.7);

    int numTriangles = 0;
    for (int i = 0; i < 10; i++) {
      for (int j = i; j < 10; j++) {
        for (int k = j; k < 10; k++) {
          if (G[i][j] && G[i][k] && G[k][j])
            numTriangles++;
        }
      }
    }

    auto triangles = getTriangles(G);
    assert(triangles.size() == numTriangles);
    for(auto t: triangles) {
      assert(G[t[0]][t[1]]);
      assert(G[t[0]][t[2]]);
      assert(G[t[1]][t[0]]);
      assert(G[t[1]][t[2]]);
      assert(G[t[2]][t[0]]);
      assert(G[t[2]][t[1]]);
    }
  }
}

int main() {
  testGraph();
  testGetTriangles();
}