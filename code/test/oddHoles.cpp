#include "oddHoles.h"
#include "commons.h"
#include "testCommons.h"

void testIsT1() {
  Graph G(5, "\
  .X..X\
  X.X..\
  .X.X.\
  ..X.X\
  X..X.\
  ");

  assert(!findT1(G).empty());
  assert(isT1(G, {0, 1, 2, 3, 4}));
  assert(isT1(G, {2, 3, 4, 0, 1}));
  assert(!isT1(G, {3, 4, 0, 1}));

  G = Graph(5, "\
  .XX.X\
  X.X..\
  XX.X.\
  ..X.X\
  X..X.\
  ");
  assert(!isT1(G, {0, 1, 2, 3, 4}));
  assert(findT1(G).empty());

  G = Graph(6, "\
  .X..X.\
  X.X...\
  .X.X..\
  ..X.X.\
  X..X..\
  ......\
  ");

  assert(!findT1(G).empty());
  assert(isT1(G, {0, 1, 2, 3, 4}));

  G = Graph(6, "\
  .X..XX\
  X.X..X\
  .X.X.X\
  ..X.XX\
  X..X.X\
  XXXXX.\
  ");

  assert(!findT1(G).empty());
  assert(isT1(G, {0, 1, 2, 3, 4}));
}

void testFindT2() {
  Graph G(9, "\
  .XX.XX...\
  X.X..XX..\
  XX.X.X...\
  ..X.X....\
  X..X.X..X\
  XXX.X....\
  .X.....X.\
  ......X.X\
  ....X..X.\
  ");

  auto t = findT2(G);
  auto v = get<0>(t);
  auto P = get<1>(t);
  auto X = get<2>(t);

  cout << v << endl << P << endl << X << endl;
  // is
  // [0, 1, 2, 3]
  // [0, 4, 3]
  // [2]
  // OK? X is in v

}

int main() {
  init();
  testIsT1();
  testFindT2();
}