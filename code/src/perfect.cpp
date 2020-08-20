#include "perfect.h"
#include "commons.h"
#include "jewels.h"
#include "nearCleaners.h"
#include "oddHoles.h"
#include "pyramids.h"
#include "testCommons.h"

bool containsSimpleProhibited(const Graph &G) {
  return containsJewelNaive(G) || containsPyramid(G) || containsT1(G) || containsT2(G) || containsT3(G);
}

bool isPerfectGraph(const Graph &G, bool gatherStats) {
  const bool printInterestingGraphs = false;

  if (gatherStats) StatsFactory::startTestCasePart("Simple Structures");

  Graph GC = G.getComplement();
  if (containsSimpleProhibited(G) || containsSimpleProhibited(GC)) return false;

  if (gatherStats) StatsFactory::startTestCasePart("Get Near Cleaners");
  auto Xs = getPossibleNearCleaners(G);

  for (auto X : Xs) {
    if (containsOddHoleWithNearCleanerX(G, X, gatherStats)) {
      if (printInterestingGraphs) cout << "Interesting graph: " << G << endl;
      return false;
    }
  }

  if (gatherStats) StatsFactory::startTestCasePart("Get Near Cleaners");
  auto XsC = getPossibleNearCleaners(GC);

  for (auto X : XsC) {
    if (containsOddHoleWithNearCleanerX(GC, X, gatherStats)) {
      if (printInterestingGraphs) cout << "Interesting graph: " << G << endl;
      return false;
    }
  }

  return true;
}

bool isPerfectGraphNaive(const Graph &G, bool gatherStats) {
  return !containsOddHoleNaive(G, gatherStats) && !containsOddHoleNaive(G.getComplement(), gatherStats);
}
