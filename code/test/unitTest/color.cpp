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

  double c = color(G);
  std::cout << c << std::endl;
}

int main() {
  init();
  testGetGraphEdges();
  testColor();

  return 0;
}
