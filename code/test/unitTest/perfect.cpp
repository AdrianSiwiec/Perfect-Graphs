#include "perfect.h"
#include <random>
#include "commons.h"
#include "ctime"
#include "oddHoles.h"
#include "testCommons.h"

double allNaive = 0;
double allPerfect = 0;

void testGraphSimple(const Graph &G) {
  bool naivePerfect = isPerfectGraphNaive(G);
  bool perfect = isPerfectGraph(G);

  assert(naivePerfect == perfect);
}

void testHand() {
  testGraphSimple(Graph(8,
                        "\
  .XXXX...\
  X..XX..X\
  X...X.XX\
  XX...XXX\
  XXX..XX.\
  ...XX..X\
  ..XXX..X\
  .XXX.XX.\
  "));

  testGraphSimple(Graph(8,
                        "\
  ..X..X.X\
  ......XX\
  X...XXX.\
  ....XX..\
  ..XX..X.\
  X.XX....\
  .XX.X..X\
  XX....X.\
"));

  testGraphSimple(Graph(8,
                        "\
  .XXXXX..\
  X..XX.XX\
  X..X.X.X\
  XXX.X..X\
  XX.X.X..\
  X.X.X.XX\
  .X...X.X\
  .XXX.XX.\
  "));
}

void testPerfectVsNaive() {
  RaiiTimer timer("Perfect vs naive ");
  for (int i = 0; i < (bigTests ? 30 : 100); i++) {
    Graph G = getRandomGraph(bigTests ? 8 : 6, 0.5);
    testGraphSimple(G);
  }
}

void testNonPerfect() {
  RaiiTimer timer("Non Perfect");
  for (int i = 0; i < 100; i++) {
    Graph G = getNonPerfectGraph(9, 10, 0.5);
    assert(isPerfectGraph(G) == false);
  }
}

void testBiparite() {
  RaiiTimer timer("Biparite");

  for (int i = 0; i < (bigTests ? 100 : 20); i++) {
    Graph G = getBipariteGraph(bigTests ? 10 : 7, 0.5);
    assert(isPerfectGraph(G) == true);
  }
}

void testLineBiparite() {
  RaiiTimer timer("Line Biparite");
  for (int i = 0; i < (bigTests ? 30 : 20); i++) {
    Graph G = getBipariteGraph(bigTests ? 9 : 7, 0.5).getLineGraph();

    assert(isPerfectGraph(G) == true);
  }
}

int main() {
  init();
  testHand();
  testPerfectVsNaive();
  testNonPerfect();
  testBiparite();
  testLineBiparite();
}
