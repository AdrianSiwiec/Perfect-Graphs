#include "pyramids.h"
#include "commons.h"
#include "testCommons.h"

void testCheckPrerequisites() {
  // Only b=[4, 5, 6], s=[1, 3, 4] with a=2 should pass checkPrerequisites
  Graph G(6, "\
  .X....\
  X.XX..\
  .X....\
  .X..XX\
  ...X.X\
  ...XX.\
  ");

  assert(checkPrerequisites(G, {3, 4, 5}, 1, {3, 2, 0}));

  int passedCount = 0;
  for (int a = 0; a < 6; a++) {
    auto bTuples = getTriangles(G);
    auto eStarTriangles = getEmptyStarTriangles(G);
    vec<vec<int>> sTuples;
    for (auto e : eStarTriangles) {
      sTuples.push_back(e.nd);
    }

    for (auto &b : bTuples) {
      for (auto &s : sTuples) {
        bool passed = checkPrerequisites(G, b, a, s);

        if (passed) {
          passedCount++;

          cout << b << " " << a << " " << s << endl;
          assert(a == 1);

        } else {
        }
      }
    }
  }

  assert(passedCount > 0);
}

int main() {
  init();
  testCheckPrerequisites();
}