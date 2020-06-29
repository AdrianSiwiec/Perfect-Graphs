#include "color.h"
#include <ctime>
#include <map>
#include <random>
#include "commons.h"
#include "oddHoles.h"
#include "perfect.h"
#include "testCommons.h"

map<int, double> sumTimeColor;
map<int, int> casesTestedColor;

void testColorWithStats() {
  int r = rand() % 100;
  Graph G(0);

  // G = getRandomPerfectGraph(sumTimeColor.size() + 5, 0.5);
  G = getBipariteGraph(8 + getDistr() * 6, getDistr()).getLineGraph();

  clock_t start;
  start = clock();

  auto c = color(G);

  double duration = (clock() - start) / static_cast<double>(CLOCKS_PER_SEC);

  assert(isColoringValid(G, c));

  sumTimeColor[G.n] += duration;
  casesTestedColor[G.n]++;
}

void printStatsColor() {
  for (auto it = sumTimeColor.begin(); it != sumTimeColor.end(); it++) {
    int cases = casesTestedColor[it->first];
    cout << "\tn=" << it->first << ", cases=" << cases << ", avgTime=" << it->second / cases << endl;
  }

  cout << endl;
}

int main() {
  init(true);
  while (1) {
    testColorWithStats();

    printStatsColor();
  }
}
