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
  assert(v == (vec<int>{1, 2, 3, 4}));
  assert(P == (vec<int>{1, 6, 7, 8, 4}));
  assert(X == (vec<int>{0}));

  G = Graph(9, "\
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
  t = findT2(G);
  v = get<0>(t);
  P = get<1>(t);
  X = get<2>(t);
  assert(v.empty());
  assert(P.empty());
  assert(X.empty());

  G = Graph(9, "\
  .XX.XX...\
  X.X..XX..\
  XX.X.X...\
  ..X.X....\
  X..X.X..X\
  XXX.X....\
  .X.....XX\
  ......X.X\
  ....X.XX.\
  ");

  t = findT2(G);
  v = get<0>(t);
  P = get<1>(t);
  X = get<2>(t);
  assert(v == (vec<int>{1, 2, 3, 4}));
  assert(P == (vec<int>{1, 6, 8, 4}));
  assert(X == (vec<int>{0}));
}

void testT2IsOddHole() {
  for (int i = 0; i < (bigTests ? 10000 : 200); i++) {
    Graph G = getRandomGraph(8, 0.5);

    auto t2 = findT2(G);
    if (!get<0>(t2).empty()) {
      assert(containsOddHoleNaive(G));
    }
  }
}

void testIsT3() {
  Graph G(8, "\
  .XX..X..\
  X.X.X...\
  XX.X....\
  ..X.XX..\
  .X.X..X.\
  X..X...X\
  ....X..X\
  .....XX.\
  ");

  assert(isT3(G, {1, 2, 3, 4, 5, 6}, {6, 7, 5}, {0}));
  assert(isT3(G, {0, 2, 3, 5, 4, 7}, {7, 6, 4}, {1}));
  assert(isT3(G, {0, 2, 3, 5, 4, 7}, {4, 6, 7}, {1}));
}

void testFindT3() {
  Graph G(8, "\
  .XX..X..\
  X.X.X...\
  XX.X....\
  ..X.XX..\
  .X.X..X.\
  X..X...X\
  ....X..X\
  .....XX.\
  ");

  auto t = findT3(G);
  auto v = get<0>(t);
  auto P = get<1>(t);
  auto X = get<2>(t);
  assert(isT3(G, v, P, X));
  assert(v == (vec<int>{0, 2, 3, 5, 4, 7}));
  assert(P == (vec<int>{7, 6, 4}));
  assert(X == (vec<int>{1}));

  G = Graph(8, "\
  .XXX.X..\
  X.X.X...\
  XX.X....\
  X.X.XX..\
  .X.X..X.\
  X..X...X\
  ....X..X\
  .....XX.\
  ");
  t = findT3(G);
  v = get<0>(t);
  P = get<1>(t);
  X = get<2>(t);
  assert(v.empty());
  assert(P.empty());
  assert(X.empty());

  G = Graph(8, "\
  .XX..X.X\
  X.X.X...\
  XX.X....\
  ..X.XX..\
  .X.X..X.\
  X..X...X\
  ....X..X\
  X....XX.\
  ");
  t = findT3(G);
  v = get<0>(t);
  P = get<1>(t);
  X = get<2>(t);
  assert(v.empty());
  assert(P.empty());
  assert(X.empty());

  G = Graph(8, "\
  .XX..X..\
  X.X.X...\
  XX.X....\
  ..X.XX.X\
  .X.X..X.\
  X..X...X\
  ....X..X\
  ...X.XX.\
  ");
  t = findT3(G);
  v = get<0>(t);
  P = get<1>(t);
  X = get<2>(t);
  assert(isT3(G, v, P, X));
  assert(v == (vec<int>{0, 2, 3, 5, 4, 7}));
  assert(P == (vec<int>{7, 6, 4}));
  assert(X == (vec<int>{1}));

  G = Graph(8, "\
  .XX..X..\
  X.X.X...\
  XX.X....\
  ..X.XX.X\
  .X.X.XX.\
  X..XX..X\
  ....X..X\
  ...X.XX.\
  ");
  t = findT3(G);
  v = get<0>(t);
  P = get<1>(t);
  X = get<2>(t);
  assert(v.empty());
  assert(P.empty());
  assert(X.empty());
}

void testT3IsOddHole() {
  for (int i = 0; i < (bigTests ? 20000 : 400); i++) {
    Graph G = getRandomGraph(8, 0.5);

    auto t = findT3(G);
    auto v = get<0>(t);
    auto P = get<1>(t);
    auto X = get<2>(t);
    if (!v.empty()) {
      assert(isT3(G, v, P, X));
      assert(containsOddHoleNaive(G));
    }
  }
}

int main() {
  init();
  testIsHole();
  testContainsHoleOfSize();
  testContainsOddHoleNaive();
  testIsT1();
  testT1IsOddHole();
  testFindT2();
  testT2IsOddHole();
  testIsT3();
  testFindT3();
  testT3IsOddHole();
}