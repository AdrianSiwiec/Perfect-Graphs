#include "color.h"

#include "colorTestCommons.h"
#include "commons.h"
#include "perfect.h"
#include "testCommons.h"

using namespace std;

void testGetGraphEdges() {
  Graph G(6,
          "\
  .XX.X.\
  X.X..X\
  XX.X..\
  ..X.X.\
  X..X.X\
  .X..X.\
  ");

  auto t = getGraphEdges(G);
  assert(get<0>(t) == 6);
  assert(get<1>(t) == 8);
  assert(get<2>(t) == (vec<int>{1, 1, 1, 2, 2, 3, 4, 5}));
  assert(get<3>(t) == (vec<int>{2, 3, 5, 3, 6, 4, 5, 6}));

  t = getGraphEdges(G, {0, 1, 0, 0});
  assert(get<0>(t) == 5);
  assert(get<1>(t) == 5);
  assert(get<2>(t) == (vec<int>{1, 1, 2, 3, 4}));
  assert(get<3>(t) == (vec<int>{2, 4, 3, 4, 5}));

  t = getGraphEdges(G, {1, 1, 0, 0});
  assert(get<0>(t) == 4);
  assert(get<1>(t) == 3);
  assert(get<2>(t) == (vec<int>{1, 2, 3}));
  assert(get<3>(t) == (vec<int>{2, 3, 4}));
}

void testGetTheta() {
  Graph G(8,
          "\
  ..X..X.X\
  ......XX\
  X...XXX.\
  ....XX..\
  ..XX..X.\
  X.XX....\
  .XX.X..X\
  XX....X.\
");

  assert(getTheta(G) == 3);
}

void testMaxStableSetHand() {
  Graph G(8,
          "\
  ..X..X.X\
  ......XX\
  X...XXX.\
  ....XX..\
  ..XX..X.\
  X.XX....\
  .XX.X..X\
  XX....X.\
");

  assert(getMaxCardStableSet(G) == (vec<int>{4, 5, 7}));

  assert(getMaxCardClique(G) == (vec<int>{2, 4, 6}));
}

void testMaxCardStableSet() {
  for (int i = 0; i < (bigTests ? 100 : 10); i++) {
    Graph G = getRandomPerfectGraph(bigTests ? 8 : 7, 0.5);

    vec<int> maxSS = getMaxCardStableSet(G);

    assert(maxSS.size() == getTheta(G));
    assert(isStableSet(G, maxSS));
  }
}

void testMaxSSIntersectingCliques() {
  Graph G(5,
          "\
  .XX..\
  X.X..\
  XX.XX\
  ..X.X\
  ..XX.\
  ");
  vec<vec<int>> K = {{0, 1, 2}, {2, 3, 4}};

  assert(getSSIntersectingCliques(G, K) == (vec<int>{2}));

  G = Graph(10,
            "\
  .XXX......\
  X.XX......\
  XX.XXXXX..\
  XXX...XX..\
  ..X..XX...\
  ..X.X.X...\
  ..XXXX.XXX\
  ..XX..X.XX\
  ......XX.X\
  ......XXX.\
  ");

  K = vec<vec<int>>{{0, 1, 2, 3}, {6, 7, 8, 9}};
  assert(getSSIntersectingCliques(G, K) == (vec<int>{3, 9}));

  K = vec<vec<int>>{{0, 1, 2, 3}, {4, 2, 6, 5}, {6, 7, 8, 9}};
  assert(getSSIntersectingCliques(G, K) == (vec<int>{3, 5, 9}));

  K = vec<vec<int>>{{2, 3, 7, 6}, {3, 1, 0, 2}, {8, 9, 6, 7}};
  assert(getSSIntersectingCliques(G, K) == (vec<int>{3, 9}));

  K = vec<vec<int>>{{2, 3, 7, 6}, {2, 5, 6, 4}, {3, 2, 0, 1}, {8, 9, 6, 7}};
  assert(getSSIntersectingCliques(G, K) == (vec<int>{3, 5, 9}));
}

void testGetSSIntersectingAllMaxCardCliquesHand() {
  Graph G(5,
          "\
  .XX..\
  X.X..\
  XX.XX\
  ..X.X\
  ..XX.\
  ");

  assert(getSSIntersectingAllMaxCardCliques(G) == (vec<int>{2}));

  G = Graph(10,
            "\
  .XXX......\
  X.XX......\
  XX.XXXXX..\
  XXX...XX..\
  ..X..XX...\
  ..X.X.X...\
  ..XXXX.XXX\
  ..XX..X.XX\
  ......XX.X\
  ......XXX.\
  ");

  assert(getSSIntersectingAllMaxCardCliques(G) == (vec<int>{3, 5, 9}));
}

void testGetSSIntersectingAllMaxCardCliques() {
  for (int i = 0; i < (bigTests ? 50 : 10); i++) {
    Graph G = getRandomPerfectGraph(bigTests ? 5 : 7, 0.5);

    auto SS = getSSIntersectingAllMaxCardCliques(G);
    assert(isStableSet(G, SS));

    Graph Gprim = G.getInducedStrong(getComplementNodesVec(G.n, SS));

    assert(getOmega(G) > getOmega(Gprim));
  }
}

void testColorHand() {
  Graph G(5,
          "\
  .XX..\
  X.X..\
  XX.XX\
  ..X.X\
  ..XX.\
  ");

  assert(color(G) == (vec<int>{2, 1, 0, 2, 1}));

  G = Graph(10,
            "\
  .XXX......\
  X.XX......\
  XX.XXXXX..\
  XXX...XX..\
  ..X..XX...\
  ..X.X.X...\
  ..XXXX.XXX\
  ..XX..X.XX\
  ......XX.X\
  ......XXX.\
  ");

  assert(color(G) == (vec<int>{3, 2, 1, 0, 3, 0, 2, 3, 1, 0}));
}

void testColor() {
  for (int i = 0; i < (bigTests ? 20 : 5); i++) {
    Graph G = getRandomPerfectGraph(bigTests ? 8 : 6, 0.5);

    assert(isColoringValid(G, color(G)));
  }
}

int main() {
  init();
  testGetGraphEdges();
  testGetTheta();
  testMaxStableSetHand();
  testMaxCardStableSet();
  testMaxSSIntersectingCliques();
  testGetSSIntersectingAllMaxCardCliquesHand();
  testGetSSIntersectingAllMaxCardCliques();
  testColorHand();
  testColor();
  return 0;
}
