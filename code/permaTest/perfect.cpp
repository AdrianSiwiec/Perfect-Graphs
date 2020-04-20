#include "perfect.h"
#include "commons.h"
#include "oddHoles.h"
#include "testCommons.h"
#include <ctime>
#include <random>

double allNaive = 0;
double allPerfect = 0;

void testGraph(const Graph &G) {
  cout << "Testing " << G.n << endl;

  clock_t startNaive;
  startNaive = clock();
  bool naivePerfect = isPerfectGraphNaive(G);
  double durationNaive = (std::clock() - startNaive) / (double)CLOCKS_PER_SEC;

  clock_t startPerfect;
  startPerfect = clock();
  bool perfect = isPerfectGraph(G);
  double durationPerfect = (std::clock() - startPerfect) / (double)CLOCKS_PER_SEC;

  if (naivePerfect != perfect) {
    cout << "ERROR: " << endl << "naive=" << naivePerfect << endl << "perfect=" << perfect << endl;
    cout << G << endl;
    if (!naivePerfect)
      cout << findOddHoleNaive(G) << endl;
  }

  assert(naivePerfect == perfect);

  cout << "Durations:        ";
  cout << durationNaive << "\t" << durationPerfect << endl;
  allNaive += durationNaive;
  allPerfect += durationPerfect;
  cout << "Sum of durations: ";
  cout << allNaive << "\t" << allPerfect << endl;
}

void testPerfectVsNaive() {
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.5, 0.15);

  cout << rand() << endl;

  for (int i = 0; i < 1000000; i++) {
    int r = rand() % 100;

    double distr = distribution(generator);
    if (distr <= 0 || distr > 1)
      distr = 0.5;

    if (r == 0) {
      Graph G = getRandomGraph(9, distr);
      testGraph(G);
    } else {
      Graph G = getRandomGraph(8, distr);
      testGraph(G);
    }

    cout << "Test #" << i << endl;
  }
}

int main() {
  init(true);
  testPerfectVsNaive();
}