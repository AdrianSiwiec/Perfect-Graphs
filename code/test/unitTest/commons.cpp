#include "commons.h"
#include <cassert>
#include <iostream>
#include "testCommons.h"

void testGraph() {
  Graph g(10);
  assert(g.n == 10);
  for (int i = 0; i < 10; i++) {
    assert(g[i].size() == 0);
  }

  bool caught = false;
  try {
    g = Graph(5,
              "\
    XXXXX\
    XXXXX\
    XXXXX");
  } catch (invalid_argument &e) {
    caught = true;
  }
  assert(caught);

  caught = false;
  try {
    g = Graph(5,
              "\
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

  g = Graph(5,
            "\
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
  for (int i = 1; i < 5; i++) assert(g[i].size() == 1);

  g = getRandomGraph(10, 0.5);
  Graph gc = g.getComplement();
  for (int i = 0; i < 10; i++) {
    vec<int> nl;

    for (int j = 0; j < 10; j++) {
      if (i == j) {
        assert(!gc.areNeighbours(i, j));
      } else {
        assert(g.areNeighbours(i, j) != gc.areNeighbours(j, i));
        if (!g.areNeighbours(i, j)) {
          nl.push_back(j);
        }
      }
    }

    assert(nl == gc[i]);
  }
}

void testGraphGetInduced() {
  Graph G(7,
          "\
  ...X...\
  ..X.XX.\
  .X...X.\
  X......\
  .X.....\
  .XX....\
  .......\
  ");

  Graph Gprim = G.getInduced({1, 2, 5});
  assert(Gprim[1] == (vec<int>{2, 5}));
  assert(Gprim[2] == (vec<int>{1, 5}));
  assert(Gprim[5] == (vec<int>{1, 2}));

  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 7; j++) {
      assert(Gprim.areNeighbours(i, j) ==
             (i != j && (i == 1 || i == 2 || i == 5) && (j == 1 || j == 2 || j == 5)));
    }
  }
}

void testGraphGetIncudedStrong() {
  Graph G(7,
          "\
  ...X...\
  ..X.XX.\
  .X...X.\
  X......\
  .X.....\
  .XX....\
  .......\
  ");

  Graph Gprim = G.getInducedStrong({1, 2, 5});
  assert(Gprim[0] == (vec<int>{1, 2}));
  assert(Gprim[1] == (vec<int>{0, 2}));
  assert(Gprim[2] == (vec<int>{0, 1}));

  assert(Gprim.n == 3);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i != j) assert(Gprim.areNeighbours(i, j));
    }
  }
}

void testGetComplementNodes() {
  assert(getComplementNodesVec(8, {0, 1, 4}) == (vec<int>{2, 3, 5, 6, 7}));
  assert(getComplementNodesVec(8, {}) == (vec<int>{0, 1, 2, 3, 4, 5, 6, 7}));
  assert(getComplementNodesVec(8, {0, 1, 2, 3, 4, 5, 6, 7}) == (vec<int>{}));
}

void testGraphGetShuffled() {
  Graph G = getRandomGraph(10, 0.5);
  Graph GS = G.getShuffled();

  int ns = 0;
  int sns = 0;

  int sumSize = 0;
  int ssumSize = 0;

  for (int i = 0; i < G.n; i++) {
    for (int j = 0; j < G.n; j++) {
      assert(GS.areNeighbours(i, j) == GS.areNeighbours(j, i));

      if (G.areNeighbours(i, j)) ns++;
      if (GS.areNeighbours(i, j)) sns++;
    }

    sumSize += G[i].size();
    ssumSize += GS[i].size();
  }

  assert(ns == sns);
  assert(sumSize == ssumSize);
  assert(ns = ssumSize);
}
void testGraphGetNextNeighbor() {
  Graph G(7,
          "\
  ...X...\
  ..X.XX.\
  .X...X.\
  X......\
  .X.....\
  .XX....\
  .......\
  ");

  assert(G.getFirstNeighbour(0) == 3);
  assert(G.getNextNeighbour(0, 3) == -1);
  assert(G.getFirstNeighbour(1) == 2);
  assert(G.getNextNeighbour(1, 2) == 4);
  assert(G.getNextNeighbour(1, 4) == 5);
  assert(G.getNextNeighbour(1, 5) == -1);

  bool caught = false;
  try {
    G.getNextNeighbour(0, 1);
  } catch (invalid_argument &e) {
    caught = true;
  }
  assert(caught);

  assert(G.getFirstNeighbour(6) == -1);

  Graph gPrim({{3}, {2, 4, 5}, {1, 5}, {0}, {1}, {1, 2}, {}});
  assert(gPrim == G);

  assert(gPrim.getFirstNeighbour(0) == 3);
  assert(gPrim.getNextNeighbour(0, 3) == -1);
  assert(gPrim.getFirstNeighbour(1) == 2);
  assert(gPrim.getNextNeighbour(1, 2) == 4);
  assert(gPrim.getNextNeighbour(1, 4) == 5);
  assert(gPrim.getNextNeighbour(1, 5) == -1);

  caught = false;
  try {
    gPrim.getNextNeighbour(0, 1);
  } catch (invalid_argument &e) {
    caught = true;
  }
  assert(caught);

  assert(gPrim.getFirstNeighbour(6) == -1);
}

void testGetLineGraph() {
  Graph G(7,
          "\
  ...X...\
  ..X.XX.\
  .X...X.\
  X......\
  .X.....\
  .XX....\
  .......\
  ");
  assert(G.getLineGraph() == Graph(5,
                                   "\
  .....\
  ..XXX\
  .X.X.\
  .XX.X\
  .X.X.\
  "));

  G = Graph(5,
            "\
  .XXX.\
  X...X\
  X..X.\
  X.X.X\
  .X.X.\
  ");
  assert(G.getLineGraph() == Graph(6,
                                   "\
  .XXX..\
  X.X.X.\
  XX..XX\
  X....X\
  .XX..X\
  ..XXX.\
  "));
}

void testGetTriangles() {
  for (int i = 0; i < 15; i++) {
    Graph G = getRandomGraph(10, 0.7);

    int numTriangles = 0;
    for (int i = 0; i < 10; i++) {
      for (int j = i; j < 10; j++) {
        for (int k = j; k < 10; k++) {
          if (G.areNeighbours(i, j) && G.areNeighbours(j, k) && G.areNeighbours(i, k)) numTriangles++;
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

  assert(countNonZeros({0, 1, 0, 2}) == 2);
  assert(countNonZeros({0, 1, 0, 0}) == 1);
  assert(countNonZeros({2, 1, 0, -1}) == 3);
  assert(countNonZeros({}) == 0);

  assert(getPrefSum({1, 2, 3, -10, 240, 0, 1}) == (vec<int>{1, 3, 6, -4, 236, 236, 237}));
  assert(getPrefSum({}) == (vec<int>{}));
}

void testEmptyStarTriangles() {
  for (int i = 0; i < 15; i++) {
    Graph G = getRandomGraph(10, 0.7);

    int numEmptyStars = 0;
    for (int a = 0; a < 10; a++) {
      for (int s0 = 0; s0 < 10; s0++) {
        if (s0 == a) continue;
        for (int s1 = 0; s1 < 10; s1++) {
          if (s1 == a || s1 == s0) continue;
          for (int s2 = 0; s2 < 10; s2++) {
            if (s2 == a || s2 == s0 || s2 == s1) continue;
            if (G.areNeighbours(a, s0) && G.areNeighbours(a, s1) && G.areNeighbours(a, s2) &&
                !G.areNeighbours(s0, s1) && !G.areNeighbours(s0, s2) && !G.areNeighbours(s1, s2))
              numEmptyStars++;
          }
        }
      }
    }
    assert((numEmptyStars % 6) == 0);  // sanity check, we count each permutation of s1, s2, s3

    auto emptyStars = getEmptyStarTriangles(G);
    assert(numEmptyStars == emptyStars.size());
  }
}

void testGetCompleteVertices() {
  Graph G(6,
          "\
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
  Graph G(5,
          "\
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

void testAllShortestPaths() {
  Graph G(5,
          "\
  .X..X\
  X.X..\
  .X.X.\
  ..X.X\
  X..X.\
  ");

  auto noFour = [](int v) { return v != 4; };

  vec<vec<vec<int>>> R(G.n, vec<vec<int>>(G.n, vec<int>()));
  for (int i = 0; i < G.n; i++) {
    for (int j = 0; j < G.n; j++) {
      R[i][j] = findShortestPathWithPredicate(G, i, j, noFour);
    }
  }

  vec<vec<int>> Rlen;
  vec<vec<int>> penultimate;
  Rlen = allShortestPathsWithPredicate(G, noFour, penultimate);

  for (int i = 0; i < G.n; i++) {
    for (int j = 0; j < G.n; j++) {
      assert(Rlen[i][j] == R[i][j].size());
      if (R[i][j].size() > 1) {
        assert(penultimate[i][j] == R[i][j][R[i][j].size() - 2]);
      }
    }
  }

  G = Graph(6,
            "\
  .XX...\
  X.XX..\
  XX.X..\
  .XX.X.\
  ...X.X\
  ....X.\
  ");
  auto noOne = [](int v) { return v != 1; };
  R = vec<vec<vec<int>>>(G.n, vec<vec<int>>(G.n, vec<int>()));
  for (int i = 0; i < G.n; i++) {
    for (int j = 0; j < G.n; j++) {
      R[i][j] = findShortestPathWithPredicate(G, i, j, noOne);
    }
  }

  Rlen = allShortestPathsWithPredicate(G, noOne, penultimate);

  for (int i = 0; i < G.n; i++) {
    for (int j = 0; j < G.n; j++) {
      assert(Rlen[i][j] == R[i][j].size());
      if (R[i][j].size() > 1) {
        assert(penultimate[i][j] == R[i][j][R[i][j].size() - 2]);
      }
    }
  }
}

void testDfsWith() {
  string s;
  auto writeToS = [&](int v) { s += std::to_string(v); };

  Graph G(6,
          "\
  .XX...\
  X.XX..\
  XX....\
  .X..X.\
  ...X.X\
  ....X.\
  ");
  vec<int> visited(G.n);

  dfsWith(G, visited, 3, writeToS,
          [](int v) -> bool { return v != 3; });  // test should do nothing, we start from 3
  assert(visited == vec<int>(G.n, true));
  assert(s == "310245");

  s = "";
  visited = vec<int>{0, 0, 0, 0, 1, 0};
  dfsWith(G, visited, 3, writeToS);
  assert(visited == (vec<int>{1, 1, 1, 1, 1, 0}));
  assert(s == "3102");

  s = "";
  visited = vec<int>{0, 0, 0, 0, 1, 0};
  dfsWith(G, visited, 3, writeToS, [](int v) -> bool { return v > 0; });
  assert(visited == (vec<int>{0, 1, 1, 1, 1, 0}));
  assert(s == "312");
}

void testComponents() {
  Graph G(3);
  assert(getComponents(G) == (vec<vec<int>>{{0}, {1}, {2}}));

  G = Graph(7,
            "\
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

void testGetComponentsOfInducedGraph() {
  Graph G(7,
          "\
  ...X...\
  ..X.XX.\
  .X...X.\
  X......\
  .X.....\
  .XX....\
  .......\
  ");
  assert(getComponentsOfInducedGraph(G, vec<int>{1, 2, 3, 4}) == (vec<vec<int>>{{1, 2, 4}, {3}}));
}

void testIsAPath() {
  Graph G(6,
          "\
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
  assert(!isAPath(G, {0, 1, 2}));
  assert(!isAPath(G, {0, 1, 0}));
  assert(!isAPath(G, {0, 1, 2, 0}));
  assert(isAPath(G, {0, 1, 3, 4, 5}));
  assert(!isAPath(G, {0, 1, 3, 5}));

  assert(isAPath(G, {0, 1, 2}, true));
  assert(!isAPath(G, {2, 0, 1, 3}, true, false));
  assert(isAPath(G, {2, 0, 1, 3}, true, true));

  G = Graph(6,
            "\
  .XX...\
  X.XX..\
  XX.X..\
  .XX.X.\
  ...X.X\
  ....X.\
  ");

  assert(isAPath(G, {2, 0, 1, 3}, true, true));
  assert(isAPath(G, {0, 1, 2, 3}, true, true));
}

void testNextPathInPlace() {
  Graph G(6,
          "\
  .XX...\
  X.XX..\
  XX....\
  .X..X.\
  ...X.X\
  ....X.\
  ");

  // Tests below are implementation-dependent and should be carefully updated when nextPathInPlace is modified
  vec<int> v;
  nextPathInPlace(G, v, 2);
  assert(v == (vec<int>{0, 1}));
  nextPathInPlace(G, v, 2);
  assert(v == (vec<int>{0, 2}));
  nextPathInPlace(G, v, 2);
  assert(v == (vec<int>{1, 0}));
  nextPathInPlace(G, v, 2);
  assert(v == (vec<int>{1, 2}));
  nextPathInPlace(G, v, 2);
  assert(v == (vec<int>{1, 3}));
  nextPathInPlace(G, v, 2);
  assert(v == (vec<int>{2, 0}));

  v = vec<int>{5, 4};
  nextPathInPlace(G, v, 2);
  assert(v == vec<int>());

  nextPathInPlace(G, v, 3);
  assert(v == (vec<int>{0, 1, 3}));
  nextPathInPlace(G, v, 3);
  assert(v == (vec<int>{1, 3, 4}));
  nextPathInPlace(G, v, 3);
  assert(v == (vec<int>{2, 1, 3}));

  v = vec<int>();
  nextPathInPlace(G, v, 3, true);
  assert(v == (vec<int>{0, 1, 2}));
  nextPathInPlace(G, v, 3, true);
  assert(v == (vec<int>{0, 1, 3}));
  nextPathInPlace(G, v, 3, true);
  assert(v == (vec<int>{0, 2, 1}));

  v = vec<int>();
  nextPathInPlace(G, v, 5);
  assert(v == (vec<int>{0, 1, 3, 4, 5}));

  v = vec<int>();
  int counter = 0;
  do {
    nextPathInPlace(G, v, 4);
    counter++;
  } while (!v.empty());
  assert(counter == 7);

  v = vec<int>();
  counter = 0;
  do {
    nextPathInPlace(G, v, 3);
    counter++;
  } while (!v.empty());
  assert(counter == 9);

  v = vec<int>();
  counter = 0;
  do {
    nextPathInPlace(G, v, 3, true);
    counter++;
  } while (!v.empty());
  assert(counter == 15);

  G = Graph(6,
            "\
  .XX...\
  X.XX..\
  XX.X..\
  .XX.X.\
  ...X.X\
  ....X.\
  ");

  v = vec<int>();
  nextPathInPlace(G, v, 4, true, true);
  assert(v == (vec<int>{0, 1, 2, 3}));

  nextPathInPlace(G, v, 4, true, true);
  assert(v == (vec<int>{0, 1, 3, 2}));

  nextPathInPlace(G, v, 4, true, true);
  assert(v == (vec<int>{0, 1, 3, 4}));

  // Partial paths as an input
  G = Graph(6,
            "\
  .XX...\
  X.XX..\
  XX....\
  .X..X.\
  ...X.X\
  ....X.\
  ");
  v = vec<int>{1, 3};
  nextPathInPlace(G, v, 3);
  assert(v == (vec<int>{1, 3, 4}));

  v = vec<int>{3};
  nextPathInPlace(G, v, 3);
  assert(v == (vec<int>{3, 1, 0}));
}

int main() {
  init();
  testGraph();
  testGraphGetInduced();
  testGraphGetIncudedStrong();
  testGraphGetShuffled();
  testGraphGetNextNeighbor();
  testGetLineGraph();
  testSimpleVec();
  testGetTriangles();
  testVec();
  testGetComplementNodes();
  testEmptyStarTriangles();
  testGetCompleteVertices();
  testShortestPath();
  testAllShortestPaths();
  testDfsWith();
  testComponents();
  testGetComponentsOfInducedGraph();
  testIsAPath();
  testNextPathInPlace();
}
