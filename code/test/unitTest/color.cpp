#include "color.h"
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
  assert(get<0>(t) == 6);
  assert(get<1>(t) == 5);
  assert(get<2>(t) == (vec<int>{1, 1, 3, 4, 5}));
  assert(get<3>(t) == (vec<int>{3, 5, 4, 5, 6}));
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
}

void testMaxCardStableSet() {
  for (int i = 0; i < (bigTests ? 100 : 10); i++) {
    Graph G = getRandomPerfectGraph(bigTests ? 8 : 7, 0.5);

    vec<int> maxSS = getMaxCardStableSet(G);

    assert(maxSS.size() == getTheta(G));
    assert(isStableSet(G, maxSS));
  }
}

void testColor() {
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

  // double c = color(G);
  // std::cout << c << std::endl;

  // assert(false);
}

int main() {
  init();
  testGetGraphEdges();
  testGetTheta();
  testMaxStableSetHand();
  testMaxCardStableSet();
  testColor();

  return 0;
}
