#include "diamonds.h"
#include <cassert>
#include <iostream>

using namespace std;

int main() {
  test t;
  t.a = 0;
  increment(t);

  assert(t.a == 1);
}
