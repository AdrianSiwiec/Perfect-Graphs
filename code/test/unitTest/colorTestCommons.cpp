#include "colorTestCommons.h"

#include "color.h"
#include "commons.h"
#include "testCommons.h"

void testIsColoringValid() {
  Graph G(5,
          "\
  .XX..\
  X.X..\
  XX.XX\
  ..X.X\
  ..XX.\
  ");

  assert(isColoringValid(G, {2, 1, 0, 2, 1}));
  assert(!isColoringValid(G, {2, 1, 0, 2}));
  assert(!isColoringValid(G, {2, 1, 0, 2, 3}));
  assert(!isColoringValid(G, {1, 1, 0, 2, 3}));
}

int main() {
  init();
  testIsColoringValid();
}