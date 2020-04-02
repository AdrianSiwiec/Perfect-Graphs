#include "oddHoles.h"
#include "commons.h"
#include "testCommons.h"

void testIsHole() {
  Graph G(6, "\
  .X..X.\
  X.X...\
  .X.X..\
  ..X.X.\
  X..X..\
  ......\
  ");

  assert(isHole(G, {0, 1, 2, 3, 4}));
  assert(isHole(G, {3, 2, 1, 0, 4}));
  assert(!isHole(G, {0, 2, 3, 4}));
  assert(!isHole(G, {1, 2, 3, 4}));
  assert(!isHole(G, {0, 2, 1, 3, 4}));

  G = Graph(6, "\
  .X.XX.\
  X.X...\
  .X.X..\
  X.X.X.\
  X..X..\
  ......\
  ");
  assert(!isHole(G, {0, 1, 2, 3, 4}));
  assert(!isHole(G, {3, 2, 1, 0, 4}));
  assert(isHole(G, {0, 1, 2, 3}));
  assert(!isHole(G, {1, 2, 3, 4}));
  assert(!isHole(G, {0, 2, 1, 3, 4}));
}

void testContainsHoleOfSize() {
  Graph G(6, "\
  .XX...\
  X.XX..\
  XX....\
  .X..X.\
  ...X.X\
  ....X.\
  ");

  assert(!constainsHoleOfSize(G, 3));
  assert(!constainsHoleOfSize(G, 4));
  assert(!constainsHoleOfSize(G, 5));
  assert(!constainsHoleOfSize(G, 6));

  G = Graph(6, "\
  .XX...\
  X.XX..\
  XX..X.\
  .X..X.\
  ..XX.X\
  ....X.\
  ");
  assert(constainsHoleOfSize(G, 4));
  assert(findHoleOfSize(G, 4) == (vec<int>{3, 4, 2, 1}));
  assert(!constainsHoleOfSize(G, 3));
  assert(!constainsHoleOfSize(G, 5));
  assert(!constainsHoleOfSize(G, 6));

  G = Graph(6, "\
  .XX...\
  X.XX..\
  XX...X\
  .X..X.\
  ...X.X\
  ..X.X.\
  ");
  assert(constainsHoleOfSize(G, 5));
  assert(!constainsHoleOfSize(G, 3));
  assert(!constainsHoleOfSize(G, 4));
  assert(!constainsHoleOfSize(G, 6));
}

void testContainsOddHoleNaive() {
  Graph G(6, "\
  .XX...\
  X.XX..\
  XX...X\
  .X..X.\
  ...X.X\
  ..X.X.\
  ");
  assert(containsOddHoleNaive(G));
  assert(findOddHoleNaive(G) == (vec<int>{3, 4, 5, 2, 1}));

  G = Graph(6, "\
  .XX...\
  X.XX..\
  XX..XX\
  .X..X.\
  ..XX.X\
  ..X.X.\
  ");
  assert(!containsOddHoleNaive(G));

  G = Graph(5, "\
  .X..X\
  X.X..\
  .X.X.\
  ..X.X\
  X..X.\
  ");
  assert(containsOddHoleNaive(G));
  assert(findOddHoleNaive(G) == (vec<int>{4, 3, 2, 1, 0}));
}

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

void testT1IsOddHole() {
  int t1s = 0;
  int oddHoles = 0;
  for (int i = 0; i < (bigTests ? 1000 : 100); i++) {
    Graph G = getRandomGraph(bigTests ? 10 : 7, 0.5);

    auto t1 = findT1(G);
    if (!t1.empty()) {
      assert(isHole(G, t1));
      assert(containsOddHoleNaive(G));
    }
  }
}

void testFindT2() {
  // Graph G(9, "\
  // .XX.XX...\
  // X.X..XX..\
  // XX.X.X...\
  // ..X.X....\
  // X..X.X..X\
  // XXX.X....\
  // .X.....X.\
  // ......X.X\
  // ....X..X.\
  // ");

  Graph G(9, "\
  .XX.XX...\
  X.X..X...\
  XX.X.X...\
  ..X.X....\
  X..X.X..X\
  XXX.X....\
  .........\
  ........X\
  ....X..X.\
  ");

  auto t = findT2(G);
  auto v = get<0>(t);
  auto P = get<1>(t);
  auto X = get<2>(t);

  cout << v << endl << P << endl << X << endl;
  cout << findHoleOfSize(G, 5) << endl;
  cout << findHoleOfSize(G, 7) << endl;
  // is
  // [0, 1, 2, 3]
  // [0, 4, 3]
  // [2]
  // OK? X is in v
}

int main() {
  init();
  testIsHole();
  testContainsHoleOfSize();
  testContainsOddHoleNaive();
  testIsT1();
  testT1IsOddHole();
  testFindT2();
}