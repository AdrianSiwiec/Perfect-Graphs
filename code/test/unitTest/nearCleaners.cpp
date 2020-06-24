#include "nearCleaners.h"
#include "commons.h"
#include "jewels.h"
#include "oddHoles.h"
#include "pyramids.h"
#include "testCommons.h"

void testIsRelevantTriple() {
  Graph G(7,
          "\
  .XXXXX.\
  X.X..XX\
  XX.X.X.\
  X.X...X\
  X....XX\
  XXX.X..\
  .X.XX..\
  ");

  assert(isRelevantTriple(G, {0, 6, 0}));
  assert(!isRelevantTriple(G, {0, 1, 6}));
  assert(isRelevantTriple(G, {1, 3, 1}));
  assert(isRelevantTriple(G, {1, 3, 4}));
}

int main() {
  init();
  testIsRelevantTriple();
}
