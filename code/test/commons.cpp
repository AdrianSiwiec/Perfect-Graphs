#include "commons.h"
#include "testCommons.h"
#include <cassert>
#include <iostream>

using namespace std;

void testGraph() {
  Graph g(10);
  assert(g.n == 10);
  for (int i = 0; i < 10; i++) {
    assert(g[i].size() == 0);
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
      assert(g.areNeighbours(i, j) == ((i == 0) ^ (j == 0)));
    }
  }

  assert(g[0].size() == 4);
  for (int i = 1; i < 5; i++)
    assert(g[i].size() == 1);

  g = getRandomGraph(10, 0.5);
  Graph gc = g.getComplement();
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      if (i == j)
        assert(!gc.areNeighbours(i, j));
      else
        assert(g.areNeighbours(i, j) != gc.areNeighbours(j, i));
    }
  }
}

void testGraphGetInduced() {
  Graph G(7, "\
  ...X...\
  ..X.XX.\
  .X...X.\
  X......\
  .X.....\
  .XX....\
  .......\
  ");

  Graph Gprim = G.getInduced({1, 2, 5});
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 7; j++) {
      assert(Gprim.areNeighbours(i, j) ==
             (i != j && (i == 1 || i == 2 || i == 5) && (j == 1 || j == 2 || j == 5)));
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
          if (G.areNeighbours(i, j) && G.areNeighbours(j, k) && G.areNeighbours(i, k))
            numTriangles++;
        }
      }
    }

    auto triangles = getTriangles(G);
    assert(triangles.size() == numTriangles);
    for (auto t : triangles) {
      assert(G.areNeighbours(t[0], t[1]));
      assert(G.areNeighbours(t[0], t[2]));
      assert(G.areNeighbours(t[1], t[0]));
      assert(G.areNeighbours(t[1], t[2]));
      assert(G.areNeighbours(t[2], t[0]));
      assert(G.areNeighbours(t[2], t[1]));
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

void testSimpleVec() {
  assert(isAllZeros({0, 0, 0, 0}));
  assert(isAllZeros({0}));
  assert(isAllZeros({}));
  assert(!isAllZeros({1}));
  assert(!isAllZeros({0, 2, 0, 0}));
  assert(!isAllZeros({0, 0, 0, -3}));
  assert(!isAllZeros({-6, 0, 0, 0}));

  assert(isDistinctValues({0}));
  assert(!isDistinctValues({0, 0}));
  assert(isDistinctValues({0, 1}));
  assert(isDistinctValues({100}));
  assert(isDistinctValues({}));
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
            if (G.areNeighbours(a, s0) && G.areNeighbours(a, s1) && G.areNeighbours(a, s2) &&
                !G.areNeighbours(s0, s1) && !G.areNeighbours(s0, s2) && !G.areNeighbours(s1, s2))
              numEmptyStars++;
          }
        }
      }
    }
    assert((numEmptyStars % 6) == 0); // sanity check, we count each permutation of s1, s2, s3

    auto emptyStars = getEmptyStarTriangles(G);
    assert(numEmptyStars == emptyStars.size());
  }
}

void testGetCompleteVertices() {
  Graph G(6, "\
  ...X..\
  ....XX\
  ....XX\
  X...X.\
  .XXX..\
  .XX...\
  ");

  assert(isComplete(G, {1, 2}, 0) == false);
  assert(isComplete(G, {1, 2}, 1) == false);
  assert(isComplete(G, {1, 2}, 2) == false);
  assert(isComplete(G, {1, 2}, 3) == false);
  assert(isComplete(G, {1, 2}, 4) == true);
  assert(isComplete(G, {1, 2}, 5) == true);

  assert(getCompleteVertices(G, {4}) == (vec<int>{1, 2, 3}));
  assert(getCompleteVertices(G, {1, 2}) == (vec<int>{4, 5}));
  assert(getCompleteVertices(G, {1, 2, 3}) == (vec<int>{4}));
  assert(getCompleteVertices(G, {5, 4}) == (vec<int>{1, 2}));
}

void testShortestPath() {
  Graph G(5, "\
  .X..X\
  X.X..\
  .X.X.\
  ..X.X\
  X..X.\
  ");

  auto noFour = [](int v) { return v != 4; };

  assert(findShortestPathWithPredicate(G, 2, 2, noFour) == vec<int>{2});
  assert(findShortestPathWithPredicate(G, 2, 2, [](int v) { return false; }) == (vec<int>{2}));
  assert(findShortestPathWithPredicate(G, 0, 3, noFour) == (vec<int>{0, 1, 2, 3}));
  assert(findShortestPathWithPredicate(G, 0, 3, [](int v) { return true; }) == (vec<int>{0, 4, 3}));
}

void testDfsWith() {
  string s;
  auto writeToS = [&](int v) { s += to_string(v); };

  Graph G(6, "\
  .XX...\
  X.XX..\
  XX....\
  .X..X.\
  ...X.X\
  ....X.\
  ");
  vec<int> visited(G.n);

  dfsWith(G, visited, 3, writeToS);
  assert(visited == vec<int>(G.n, true));
  assert(s == "310245");

  s = "";
  visited = vec<int>{0, 0, 0, 0, 1, 0};
  dfsWith(G, visited, 3, writeToS);
  assert(visited == (vec<int>{1, 1, 1, 1, 1, 0}));
  assert(s == "3102");
}

void testComponents() {
  Graph G(3);
  assert(getComponents(G) == (vec<vec<int>>{{0}, {1}, {2}}));

  G = Graph(7, "\
  ...X...\
  ..X.XX.\
  .X...X.\
  X......\
  .X.....\
  .XX....\
  .......\
  ");
  assert(getComponents(G) == (vec<vec<int>>{{0, 3}, {1, 2, 5, 4}, {6}}));
}

void testIsAPath() {
  Graph G(6, "\
  .XX...\
  X.XX..\
  XX....\
  .X..X.\
  ...X.X\
  ....X.\
  ");

  assert(isAPath(G, {0, 1}));
  assert(isAPath(G, {0, 2}));
  assert(!isAPath(G, {0, 3}));
  assert(isAPath(G, {0, 1, 3}));
  assert(isAPath(G, {0, 1, 2}));
  assert(!isAPath(G, {0, 1, 0}));
  assert(!isAPath(G, {0, 1, 2, 0}));
  assert(isAPath(G, {0, 1, 3, 4, 5}));
  assert(!isAPath(G, {0, 1, 3, 5}));
}

void testNextPathInPlace() {
  Graph G(6, "\
  .XX...\
  X.XX..\
  XX....\
  .X..X.\
  ...X.X\
  ....X.\
  ");

  vec<int> v;
  nextPathInPlace(G, v, 2);
  assert(v == (vec<int>{1, 0}));
  nextPathInPlace(G, v, 2);
  assert(v == (vec<int>{2, 0}));
  nextPathInPlace(G, v, 2);
  assert(v == (vec<int>{0, 1}));
  nextPathInPlace(G, v, 2);
  assert(v == (vec<int>{2, 1}));
  nextPathInPlace(G, v, 2);
  assert(v == (vec<int>{3, 1}));
  nextPathInPlace(G, v, 2);
  assert(v == (vec<int>{0, 2}));

  v = vec<int>{4, 5};
  nextPathInPlace(G, v, 2);
  assert(v == vec<int>());

  nextPathInPlace(G, v, 3);
  assert(v == (vec<int>{2, 1, 0}));
  nextPathInPlace(G, v, 3);
  assert(v == (vec<int>{3, 1, 0}));
  nextPathInPlace(G, v, 3);
  assert(v == (vec<int>{1, 2, 0}));

  v = vec<int>();
  nextPathInPlace(G, v, 5);
  assert(v == (vec<int>{5, 4, 3, 1, 0}));
}

int main() {
  init();
  testGraph();
  testGraphGetInduced();
  testSimpleVec();
  testGetTriangles();
  testVec();
  testEmptyStarTriangles();
  testGetCompleteVertices();
  testShortestPath();
  testDfsWith();
  testComponents();
  testIsAPath();
  testNextPathInPlace();
}