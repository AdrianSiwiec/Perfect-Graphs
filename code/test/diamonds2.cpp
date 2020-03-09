#include "diamonds.h"
#include <cassert>
#include <iostream>

using namespace std;

int main() {
  cout << "HELLO!" << endl;
  test t;
  t.a = 0;
  increment(t);
  cout << t.a << endl;

  assert(0 == 0);
}
