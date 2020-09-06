#include "colorTestCommons.h"

#include <chrono>
#include <iostream>

#include "color.h"
#include "commons.h"
#include "testCommons.h"

using namespace std::chrono;
using std::default_random_engine;
using std::flush;
using std::make_pair;
using std::normal_distribution;

map<int, double> sumTimeColor;
map<int, double> sumClockTimeColor;
map<int, int> casesTestedColor;

void testGraphColor(const Graph &G) {
  StatsFactory::startTestCase(G, algo_color);

  auto coloring = color(G);
  assert(isColoringValid(G, coloring));

  StatsFactory::endTestCase(true);
}

bool isColoringValid(const Graph &G, const vec<int> &coloring) {
  if (coloring.size() != G.n) return false;

  int omegaG = getOmega(G);
  for (int c : coloring) {
    if (c < 0 || c >= omegaG) return false;
  }

  for (int i = 0; i < G.n; i++) {
    for (int j : G[i]) {
      if (coloring[i] == coloring[j]) return false;
    }
  }

  return true;
}