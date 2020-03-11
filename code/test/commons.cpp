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
      g[i][j] = (i + j) % 2;
    }
  }

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      assert(g[i][j] == (i + j) % 2);
    }
  }

  bool caught = false;
  try {
    g = Graph(5, "\
    XXXXX\
    XXXXX\
    XXXXX");
  } catch (invalid_argument &e) {
    caught = true;
  }
  assert(caught);

  caught = false;
  try {
    g = Graph(5, "\
    .....\
    .....\
    .....\
    .....\
    X....\
    ");
  } catch (invalid_argument &e) {
    caught = true;
  }
  assert(caught);

  g = Graph(5, "\
  .XXXX\
  X....\
  X....\
  X....\
  X....\
  ");
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      assert(g[i][j] == ((i == 0) ^ (j == 0)));
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
    for (auto t : triangles) {
      assert(G[t[0]][t[1]]);
      assert(G[t[0]][t[2]]);
      assert(G[t[1]][t[0]]);
      assert(G[t[1]][t[2]]);
      assert(G[t[2]][t[0]]);
      assert(G[t[2]][t[1]]);
    }
  }
}

void testVec() {
  vec<int> v(3);
  v[0] = 0;
  v[1] = 1;
  v[2] = 2;

  bool thrown = 0;
  try {
    v[3] = 3;
  } catch (std::out_of_range e) {
    thrown = 1;
  }

  assert(thrown);
}

void testEmptyStarTriangles() {
  for (int i = 0; i < 15; i++) {
    Graph G = getRandomGraph(10, 0.7);

    int numEmptyStars = 0;
    for (int a = 0; a < 10; a++) {
      for (int s0 = 0; s0 < 10; s0++) {
        if (s0 == a)
          continue;
        for (int s1 = 0; s1 < 10; s1++) {
          if (s1 == a || s1 == s0)
            continue;
          for (int s2 = 0; s2 < 10; s2++) {
            if (s2 == a || s2 == s0 || s2 == s1)
              continue;
            if (G[a][s0] && G[a][s1] && G[a][s2] && !G[s0][s1] && !G[s0][s2] &&
                !G[s1][s2])
              numEmptyStars++;
          }
        }
      }
    }
    assert((numEmptyStars % 6) ==
           0); // sanity check, we count each permutation of s1, s2, s3

    auto emptyStars = getEmptyStarTriangles(G);
    assert(numEmptyStars == emptyStars.size());
  }
}

int main() {
  init();
  testGraph();
  testGetTriangles();
  testVec();
  testEmptyStarTriangles();
}