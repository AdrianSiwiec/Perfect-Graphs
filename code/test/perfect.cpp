#include "perfect.h"
#include "commons.h"
#include "ctime"
#include "oddHoles.h"
#include "testCommons.h"
#include <random>

double allNaive = 0;
double allPerfect = 0;

void testGraph(const Graph &G) {
  bool naivePerfect = isPerfectGraphNaive(G);
  bool perfect = isPerfectGraph(G);

  assert(naivePerfect == perfect);
}

void testHand() {
  testGraph(Graph(8, "\
  .XXXX...\
  X..XX..X\
  X...X.XX\
  XX...XXX\
  XXX..XX.\
  ...XX..X\
  ..XXX..X\
  .XXX.XX.\
  "));

  testGraph(Graph(8, "\
  ..X..X.X\
  ......XX\
  X...XXX.\
  ....XX..\
  ..XX..X.\
  X.XX....\
  .XX.X..X\
  XX....X.\
"));
}

void testPerfectVsNaive() {
  for (int i = 0; i < (bigTests ? 30 : 100); i++) {
    Graph G = getRandomGraph(bigTests? 8 : 6, 0.5);
    testGraph(G);
  }
}

int main() {
  init();
  testHand();
  testPerfectVsNaive();
}