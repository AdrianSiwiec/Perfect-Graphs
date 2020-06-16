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

  G = Graph(9, "\
  ....X.X.X\
  ...X.X.X.\
  ...XX.X.X\
  .XX....X.\
  X.X..X...\
  .X..X..X.\
  X.X....X.\
  .X.X.XX..\
  X.X......\
  ");
  assert(checkPrerequisites(G, {1, 5, 7}, 2, {3, 4, 6}) == false);

  G = Graph(7, "\
  .XXXXX.\
  X.X..XX\
  XX.X.X.\
  X.X...X\
  X....XX\
  XXX.X..\
  .X.XX..\
  ");
  assert(checkPrerequisites(G, {1, 2, 5}, 0, {1, 3, 4}) == false);
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

void testIsPyramid() {
  Graph G(7, "\
  .XXX...\
  X.X..X.\
  XX..X..\
  X.....X\
  ..X...X\
  .X....X\
  ...XXX.\
  ");
  assert(isPyramid(G, {0, 1, 2}, 6, (vec<vec<int>>{vec<int>{3, 0}, vec<int>{5, 1}, vec<int>{4, 2}})));
  assert(!isPyramid(G, {0, 1}, 6, (vec<vec<int>>{vec<int>{3, 0}, vec<int>{5, 1}, vec<int>{4, 2}})));
  assert(!isPyramid(G, {0, 1, 2}, 5, (vec<vec<int>>{vec<int>{3, 0}, vec<int>{5, 1}, vec<int>{4, 2}})));
  assert(!isPyramid(G, {0, 1, 2}, 6, (vec<vec<int>>{vec<int>{3}, vec<int>{5, 1}, vec<int>{4, 2}})));
  assert(!isPyramid(G, {0, 1, 2}, 6, (vec<vec<int>>{vec<int>{3, 0}, vec<int>{5}, vec<int>{4, 2}})));
  assert(!isPyramid(G, {0, 1, 2}, 6, (vec<vec<int>>{vec<int>{3, 0}, vec<int>{5, 1}, vec<int>{4}})));
  assert(!isPyramid(G, {0, 1, 2}, 6, (vec<vec<int>>{vec<int>{3, 0}, vec<int>{5, 1}, vec<int>{2}})));
  assert(!isPyramid(G, {0, 1, 2}, 6, (vec<vec<int>>{vec<int>{3, 0}, vec<int>{1}, vec<int>{4, 2}})));
  assert(!isPyramid(G, {0, 1, 3}, 6, (vec<vec<int>>{vec<int>{3, 0}, vec<int>{5, 1}, vec<int>{4, 2}})));
  assert(!isPyramid(G, {2, 1, 0}, 6, (vec<vec<int>>{vec<int>{3, 0}, vec<int>{5, 1}, vec<int>{4, 2}})));
  assert(!isPyramid(G, {2, 0, 1}, 6, (vec<vec<int>>{vec<int>{3, 0}, vec<int>{5, 1}, vec<int>{4, 2}})));
  assert(!isPyramid(G, {1, 2, 0}, 6, (vec<vec<int>>{vec<int>{3, 0}, vec<int>{5, 1}, vec<int>{4, 2}})));
  assert(!isPyramid(G, {1, 0, 2}, 6, (vec<vec<int>>{vec<int>{3, 0}, vec<int>{5, 1}, vec<int>{4, 2}})));
  assert(!isPyramid(G, {0, 2, 1}, 6, (vec<vec<int>>{vec<int>{3, 0}, vec<int>{5, 1}, vec<int>{4, 2}})));

  G = Graph(9, "\
  ....X.X.X\
  ...X.X.X.\
  ...X..X.X\
  .XX....X.\
  X....X...\
  .X..X....\
  X.X....X.\
  .X.X..X..\
  X.X......\
  ");

  assert(isPyramid(G, {1, 3, 7}, 6, (vec<vec<int>>{vec<int>{0, 4, 5, 1}, vec<int>{2, 3}, vec<int>{7}})));
  assert(!isPyramid(G, {1, 3, 7}, 6, (vec<vec<int>>{vec<int>{0, 5, 1}, vec<int>{2, 3}, vec<int>{7}})));
  assert(!isPyramid(G, {1, 3, 7}, 6, (vec<vec<int>>{vec<int>{4, 5, 1}, vec<int>{2, 3}, vec<int>{7}})));
  assert(!isPyramid(G, {1, 3, 7}, 6, (vec<vec<int>>{vec<int>{0, 4, 1}, vec<int>{2, 3}, vec<int>{7}})));
  assert(!isPyramid(G, {1, 3, 7}, 6, (vec<vec<int>>{vec<int>{0, 4, 5}, vec<int>{2, 3}, vec<int>{7}})));
}

void testPyramidsHand() {
  Graph G(7, "\
  .XXX...\
  X.X..X.\
  XX..X..\
  X.....X\
  ..X...X\
  .X....X\
  ...XXX.\
  ");

  auto pyramid = findPyramid(G);
  assert(isPyramid(G, get<0>(pyramid), get<1>(pyramid), get<2>(pyramid)));
  assert(get<0>(pyramid) == (vec<int>{0, 1, 2}));
  assert(get<1>(pyramid) == 6);
  assert(get<2>(pyramid) == (vec<vec<int>>{vec<int>{3, 0}, vec<int>{5, 1}, vec<int>{4, 2}}));
  assert(containsPyramid(G));

  G = Graph(9, "\
  .XXX.....\
  X...X....\
  X....X...\
  X.....X..\
  .X.....X.\
  ..X....XX\
  ...X....X\
  ....XX..X\
  .....XXX.\
  ");

  pyramid = findPyramid(G);
  assert(isPyramid(G, get<0>(pyramid), get<1>(pyramid), get<2>(pyramid)));
  assert(get<0>(pyramid) == (vec<int>{5, 7, 8}));
  assert(get<1>(pyramid) == 0);
  assert(get<2>(pyramid) == (vec<vec<int>>{vec<int>{2, 5}, vec<int>{1, 4, 7}, vec<int>{3, 6, 8}}));
  assert(containsPyramid(G));

  G = Graph(9, "\
  ....X...X\
  ...X.X.X.\
  ...X..X.X\
  .XX....X.\
  X....X...\
  .X..X....\
  ..X....X.\
  .X.X..X..\
  X.X......\
  ");

  pyramid = findPyramid(G);
  assert(isPyramid(G, get<0>(pyramid), get<1>(pyramid), get<2>(pyramid)));
  assert(get<0>(pyramid) == (vec<int>{1, 3, 7}));
  assert(get<1>(pyramid) == 2);
  assert(get<2>(pyramid) == (vec<vec<int>>{vec<int>{8, 0, 4, 5, 1}, vec<int>{3}, vec<int>{6, 7}}));
  assert(containsPyramid(G));

  G = Graph(9, "\
  ....X...X\
  ...X.X.X.\
  ...X...XX\
  .XX....X.\
  X....X...\
  .X..X....\
  .........\
  .XXX.....\
  X.X......\
  ");

  pyramid = findPyramid(G);
  assert(get<0>(pyramid).empty());
  assert(get<1>(pyramid) == -1);
  assert(get<2>(pyramid).empty());
  assert(!containsPyramid(G));

  G = Graph(9, "\
  ....X...X\
  ...X.X.X.\
  ...X..X.X\
  .XX....X.\
  X....X..X\
  .X..X....\
  ..X....X.\
  .X.X..X..\
  X.X.X....\
  ");

  pyramid = findPyramid(G);
  assert(isPyramid(G, get<0>(pyramid), get<1>(pyramid), get<2>(pyramid)));
  assert(get<0>(pyramid) == (vec<int>{1, 3, 7}));
  assert(get<1>(pyramid) == 2);
  assert(get<2>(pyramid) == (vec<vec<int>>{vec<int>{8, 4, 5, 1}, vec<int>{3}, vec<int>{6, 7}}));

  G = Graph(9, "\
  ....X...X\
  ...X.X.X.\
  ...X..X.X\
  .XX....X.\
  X....X..X\
  .X..X...X\
  ..X....X.\
  .X.X..X..\
  X.X.XX...\
  ");

  pyramid = findPyramid(G);
  assert(isPyramid(G, get<0>(pyramid), get<1>(pyramid), get<2>(pyramid)));
  assert(get<0>(pyramid) == (vec<int>{1, 3, 7}));
  assert(get<1>(pyramid) == 2);
  assert(get<2>(pyramid) == (vec<vec<int>>{vec<int>{8, 5, 1}, vec<int>{3}, vec<int>{6, 7}}));

  G = Graph(9, "\
  ....X.X.X\
  ...X.X.X.\
  ...X..X.X\
  .XX....X.\
  X....X...\
  .X..X....\
  X.X....X.\
  .X.X..X..\
  X.X......\
  ");

  pyramid = findPyramid(G);
  assert(isPyramid(G, get<0>(pyramid), get<1>(pyramid), get<2>(pyramid)));
  assert(get<0>(pyramid) == (vec<int>{1, 3, 7}));
  assert(get<1>(pyramid) == 6);
  assert(get<2>(pyramid) == (vec<vec<int>>{vec<int>{0, 4, 5, 1}, vec<int>{2, 3}, vec<int>{7}}));

  G = Graph(9, "\
  ....X.X.X\
  ...X.X.X.\
  ...XX.X.X\
  .XX....X.\
  X.X..X...\
  .X..X....\
  X.X....X.\
  .X.X..X..\
  X.X......\
  ");

  pyramid = findPyramid(G);
  assert(isPyramid(G, get<0>(pyramid), get<1>(pyramid), get<2>(pyramid)));
  assert(get<0>(pyramid) == (vec<int>{1, 3, 7}));
  assert(get<1>(pyramid) == 2);
  assert(get<2>(pyramid) == (vec<vec<int>>{vec<int>{4, 5, 1}, vec<int>{3}, vec<int>{6, 7}}));

  G = Graph(9, "\
  ....X.X.X\
  ...X.X.X.\
  ...XX.X.X\
  .XX....X.\
  X.X..X...\
  .X..X..X.\
  X.X....X.\
  .X.X.XX..\
  X.X......\
  ");

  pyramid = findPyramid(G);
  assert(get<0>(pyramid).empty());
  assert(get<1>(pyramid) == -1);
  assert(get<2>(pyramid).empty());
}

void testPyramidsAreCorrect() {
  int numPyramids = 0;
  for (int iTest = 0; iTest < 1000; iTest++) {
    Graph G = getRandomGraph(10, 0.5);
    auto pyramid = findPyramid(G);
    if (get<0>(pyramid).size() > 0) {
      numPyramids++;
      assert(isPyramid(G, get<0>(pyramid), get<1>(pyramid), get<2>(pyramid)));
    }
  }
  // cout << "Num pyramids: " << numPyramids << endl;
}

int main() {
  init();
  testCheckPrerequisites();
  testVectorsCutEmpty();
  testNoEdgeBetweenVectors();
  testIsPyramid();
  testPyramidsHand();
  testPyramidsAreCorrect();
}