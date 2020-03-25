#include "jewels.h"
#include "commons.h"
#include "testCommons.h"

void testIsJewel() {
  Graph G(6, "\
  .X..X.\
  X.X..X\
  .X.X..\
  ..X.X.\
  X..X.X\
  .X..X.\
  ");

  assert(isJewel(G, vec<int>{1, 2, 3, 4, 5}));
  assert(isJewel(G, vec<int>{1, 2, 3, 4, 0}));
  assert(!isJewel(G, vec<int>{0, 2, 3, 4, 5}));
  assert(!isJewel(G, vec<int>{2, 1, 3, 4, 5}));
  assert(!isJewel(G, vec<int>{2, 3, 4, 5, 1}));
  assert(!isJewel(G, vec<int>{3, 4, 5, 1, 2}));
  assert(!isJewel(G, vec<int>{4, 5, 1, 2, 3}));

  G = Graph(6, "\
  .X.XX.\
  X.X..X\
  .X.X..\
  X.X.X.\
  X..X.X\
  .X..X.\
  ");

  assert(!isJewel(G, vec<int>{1, 2, 3, 4, 5}));

  G = Graph(6, "\
  .XX.X.\
  X.X..X\
  XX.X..\
  ..X.X.\
  X..X.X\
  .X..X.\
  ");

  assert(!isJewel(G, vec<int>{1, 2, 3, 4, 5}));

  G = Graph(6, "\
  .X..XX\
  X.X..X\
  .X.X..\
  ..X.X.\
  X..X.X\
  XX..X.\
  ");

  assert(!isJewel(G, vec<int>{1, 2, 3, 4, 5}));
}

void testFindJewelNaive() {
  Graph G(6, "\
  .X..X.\
  X.X..X\
  .X.X..\
  ..X.X.\
  X..X.X\
  .X..X.\
  ");

  auto jewel = findJewelNaive(G);
  // Not only one possible, but correct
  assert(jewel == (vec<int>{4, 3, 2, 1, 0}));
}

int main() {
  init();
  testIsJewel();
  testFindJewelNaive();
}