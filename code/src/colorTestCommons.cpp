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

void testColorWithStats(const Graph &G) {
  nanoseconds start_ns = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
  clock_t start_clock = clock();

  auto c = color(G);

  nanoseconds end_ns = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
  clock_t end_clock = clock();

  double duration = (end_ns.count() - start_ns.count()) / 1e9;
  double clock_duration = (end_clock - start_clock) / static_cast<double>(CLOCKS_PER_SEC);

  assert(isColoringValid(G, c));

  sumTimeColor[G.n] += duration;
  sumClockTimeColor[G.n] += clock_duration;
  casesTestedColor[G.n]++;
}
void printStatsColor() {
  cout << "Color stats: " << endl;
  for (auto it = sumTimeColor.begin(); it != sumTimeColor.end(); it++) {
    int cases = casesTestedColor[it->first];
    cout << "\tn=" << it->first << ", cases=" << cases << ", avgTime=" << it->second / cases
         << ", parallel factor=" << sumClockTimeColor[it->first] / it->second << endl;
  }
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