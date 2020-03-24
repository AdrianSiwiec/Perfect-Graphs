#include "pyramids.h"
#include "commons.h"
#include "testCommons.h"

void testCheckPrerequisites() {
  // Only b=[4, 5, 6], s=[1, 3, 4] with a=2 should pass checkPrerequisites
  Graph G(6, "\
  .X....\
  X.XX..\
  .X....\
  .X..XX\
  ...X.X\
  ...XX.\
  ");

  assert(checkPrerequisites(G, {3, 4, 5}, 1, {3, 2, 0}));

  int passedCount = 0;
  for (int a = 0; a < 6; a++) {
    auto bTuples = getTriangles(G);
    auto eStarTriangles = getEmptyStarTriangles(G);
    vec<vec<int>> sTuples;
    for (auto e : eStarTriangles) {
      sTuples.push_back(e.nd);
    }

    for (auto &b : bTuples) {
      for (auto &s : sTuples) {
        bool passed = checkPrerequisites(G, b, a, s);

        if (passed) {
          passedCount++;
          assert(a == 1);
        }
      }
    }
  }

  assert(passedCount == 2);
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

void testVectorsCutEmpty() {
  vec<int> a{1, 2, 3, 4, 5};
  vec<int> b{5, 6, 7, 8};

  assert(!vectorsCutEmpty(a.begin(), a.end(), b.begin(), b.end()));
  assert(!vectorsCutEmpty(b.begin(), b.end(), a.begin(), a.end()));

  assert(vectorsCutEmpty(a.begin(), a.end(), a.begin(), a.begin()));
  assert(vectorsCutEmpty(a.begin(), a.begin(), a.begin(), a.end()));

  assert(vectorsCutEmpty(a.begin(), a.end() - 1, b.begin(), b.end()));
  assert(vectorsCutEmpty(a.begin(), a.end(), b.begin() + 1, b.end()));
}

void testNoEdgeBetweenVectors() {
  Graph G(7, "\
  .X..X..\
  X.X.X..\
  .X.X...\
  ..X....\
  XX...X.\
  ....X.X\
  .....X.\
  ");

  vec<int> a{0, 1, 2, 3};
  vec<int> b{0, 4, 5, 6};

  assert(!noEdgesBetweenVectors(G, a.begin(), a.end(), b.begin(), b.end()));
  assert(!noEdgesBetweenVectors(G, a.begin() + 1, a.end(), b.begin() + 1, b.end()));

  assert(noEdgesBetweenVectors(G, a.begin(), a.end(), a.begin(), a.begin()));
  assert(noEdgesBetweenVectors(G, a.begin(), a.begin(), a.begin(), a.end()));

  assert(noEdgesBetweenVectors(G, a.begin() + 2, a.end(), b.begin(), b.end()));
  assert(noEdgesBetweenVectors(G, a.begin(), a.end(), b.begin() + 2, b.end()));
}

void testPyramidsSimple() {
  Graph G(7, "\
  .XXX...\
  X.X..X.\
  XX..X..\
  X.....X\
  ..X...X\
  .X....X\
  ...XXX.\
  ");

  auto pyramide = findPyramide(G);

  assert(get<0>(pyramide) == (vec<int>{0, 1, 2}));
  assert(get<1>(pyramide) == 6);
  assert(get<2>(pyramide) == (vec<vec<int>>{vec<int>{3, 0}, vec<int>{5, 1}, vec<int>{4, 2}}));

  // TODO more tests
}

int main() {
  init();
  testCheckPrerequisites();
  testShortestPath();
  testVectorsCutEmpty();
  testNoEdgeBetweenVectors();
  testPyramids();
}