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

void testHandInteresting() {
  testGraphSimple(Graph(15,
                        "\
...X..........X\
...X.X...X...X.\
.......XX.XX...\
XX..X..........\
...X.X...X...X.\
.X..X...X......\
..........X..X.\
..X........XXX.\
..X..X....XX.X.\
.X..X........X.\
..X...X.X..X.X.\
..X....XX.X..X.\
.......X......X\
.X..X.XXXXXX...\
X...........X..\
"));

  testGraphSimple(Graph(17,
                        "\
.........X.......\
..X..X....XX....X\
.X.XXXX.X..X..XX.\
..X.X...X...X.X.X\
..XX..XX..XXXXX.X\
.XX........X..X.X\
..X.X..XX.XXX.X.X\
....X.X.X.....XX.\
..XX..XX.X.XXXX..\
X.......X...X....\
.X..X.X....X..XX.\
.XX.XXX.X.X.X.XXX\
...XX.X.XX.X..X..\
....X...X........\
..XXXXXXX.XXX..XX\
..X....X..XX..X..\
.X.XXXX....X..X..\
"));

  testGraphSimple(Graph(10,
                        "\
.XX..XXXXX\
X.XXX..XXX\
XX...X.XXX\
.X..XXX.XX\
.X.X.XXX.X\
X.XXX.X.X.\
X..XXX.XXX\
XXX.X.X.XX\
XXXX.XXX..\
XXXXX.XX..\
"));

  testGraphSimple(Graph(10,
                        "\
..XX.XXXXX\
...XXX.XXX\
X.....XXXX\
XX..XX.X.X\
.X.X..XXXX\
XX.X..XX..\
X.X.XX..XX\
XXXXXX..XX\
XXX.X.XX..\
XXXXX.XX..\
"));

  testGraphSimple(Graph(10,
                        "\
...X.XXXX.\
...X....X.\
...X......\
XXX...X...\
.......XX.\
X.....XX.X\
X..X.X...X\
X...XX...X\
XX..X.....\
.....XXX..\
"));
}

void testPerfectVsNaive() {
  RaiiTimer timer("Perfect vs naive ");
  for (int i = 0; i < (bigTests ? 300 : 100); i++) {
    Graph G = getRandomGraph(bigTests ? 10 : 6, 0.5);
    testGraphSimple(G);
  }
}

void testNonPerfect() {
  RaiiTimer timer("Non Perfect");
  for (int i = 0; i < 1000; i++) {
    Graph G = getNonPerfectGraph(7 + (rand() % 5) * 2, 15, 0.5);
    assert(isPerfectGraph(G) == false);
    assert(isPerfectGraphNaive(G) == false);
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
  testHandInteresting();
  testPerfectVsNaive();
  testNonPerfect();
  testBiparite();
  testLineBiparite();
}
