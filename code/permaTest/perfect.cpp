#include "perfect.h"
#include "commons.h"
#include "oddHoles.h"
#include "testCommons.h"
#include <ctime>
#include <map>
#include <random>

double allNaive = 0;
double allPerfect = 0;

map<pair<int, bool>, double> sumTime;
map<pair<int, bool>, int> casesTested;

map<pair<int, bool>, double> sumTimeNaive;
map<pair<int, bool>, int> casesTestedNaive;

bool testWithStats(const Graph &G, bool naive) {
  clock_t start;
  start = clock();
  bool result = naive ? isPerfectGraphNaive(G) : isPerfectGraph(G);
  double duration = (clock() - start) / (double)CLOCKS_PER_SEC;

  if (naive) {
    sumTimeNaive[make_pair(G.n, result)] += duration;
    casesTestedNaive[make_pair(G.n, result)]++;
  } else {
    sumTime[make_pair(G.n, result)] += duration;
    casesTested[make_pair(G.n, result)]++;
  }

  return result;
}

void printStats() {
  cout << "Naive: " << endl;
  for (auto it = sumTimeNaive.begin(); it != sumTimeNaive.end(); it++) {
    int cases = casesTestedNaive[it->first];
    cout << "\tn=" << it->first.first << ", result=" << it->first.second << ", cases=" << cases
         << ", avgTime=" << it->second / cases << endl;
  }
  cout << "Perfect: " << endl;
  for (auto it = sumTime.begin(); it != sumTime.end(); it++) {
    if (!it->first.second)
      continue;
    int cases = casesTested[it->first];
    cout << "\tn=" << it->first.first << ", result=" << it->first.second << ", cases=" << cases
         << ", avgTime=" << it->second / cases << endl;
  }
  for (auto it = sumTime.begin(); it != sumTime.end(); it++) {
    if (it->first.second)
      continue;
    int cases = casesTested[it->first];
    cout << "\tn=" << it->first.first << ", result=" << it->first.second << ", cases=" << cases
         << ", avgTime=" << it->second / cases << endl;
  }
}

std::default_random_engine generator;
std::normal_distribution<double> distribution(0.5, 0.15);
double getDistr() {
  double distr = distribution(generator);
  if (distr <= 0 || distr > 1)
    distr = 0.5;

  return distr;
}

void testGraph(const Graph &G) {
  cout << "Testing " << G.n << " vs naive" << endl;

  bool naivePerfect = testWithStats(G, true);

  bool perfect = testWithStats(G, false);

  if (naivePerfect != perfect) {
    cout << "ERROR: " << endl << "naive=" << naivePerfect << endl << "perfect=" << perfect << endl;
    cout << G << endl;
    if (!naivePerfect)
      cout << findOddHoleNaive(G) << endl;
  }

  assert(naivePerfect == perfect);
}

void testGraph(const Graph &G, bool result) {
  cout << "Testing " << G.n << " vs " << result << endl;

  bool perfect = testWithStats(G, false);

  if (perfect != result) {
    cout << "Error Test Graph" << endl;
    cout << G << endl;
    cout << "Expected " << result << ", got " << perfect << endl;
  }

  assert(perfect == result);
}

void testPerfectVsNaive() {
  int r = rand() % 100;

  if (r == 0) {
    Graph G = getRandomGraph(9, getDistr());
    testGraph(G);
  } else {
    Graph G = getRandomGraph(8, getDistr());
    testGraph(G);
  }
}

void testLineBiparite() {
  Graph G = getBipariteGraph(6 + (getDistr()*5), getDistr()).getLineGraph();
  testGraph(G, true);
}

void testNonPerfect() {
  Graph G = getNonPerfectGraph(5 + (rand() % 7) * 2, 3 + (getDistr() * 20), getDistr());
  testGraph(G, false);
}

int main() {
  init(true);
  while (1) {
    testPerfectVsNaive();
    testLineBiparite();
    testNonPerfect();

    printStats();
  }
}