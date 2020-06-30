#include "color.h"
#include <ctime>
#include <map>
#include <random>
#include "commons.h"
#include "oddHoles.h"
#include "perfect.h"
#include "testCommons.h"

int main() {
  init(true);
  while (1) {
    int r = rand() % 100;
    Graph G(0);

    // G = getRandomPerfectGraph(sumTimeColor.size() + 5, 0.5);
    G = getBipariteGraph(8 + getDistr() * 6, getDistr()).getLineGraph();

    testColorWithStats(G);

    printStatsColor();
    cout << endl;
  }
}
