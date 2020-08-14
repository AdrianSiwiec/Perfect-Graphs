#include "commons.h"
#include "cuCommons.h"
#include "cuOddHoles.h"
#include "testCommons.h"

TODO TESTS

void testCuOddHole(context_t &context) {
  Graph G = getRandomGraph(11, 0.5);

  CuGraph CG(G, context);

  cuContainsHoleOfSize(CG, 9, context);

  assert(false); 
}

int main() {
  init();
  standard_context_t context(0);
  testCuOddHole(context);
}