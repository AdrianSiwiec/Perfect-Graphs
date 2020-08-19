#include "perfect.h"
#include "commons.h"
#include "jewels.h"
#include "nearCleaners.h"
#include "oddHoles.h"
#include "pyramids.h"
#include "testCommons.h"

bool isPerfectGraph(const Graph &G, bool gatherStats) {
  if (gatherStats) {
    StatsFactory::startTestCasePart("Simple Structures");
  }

  Graph GC = G.getComplement();
  if (containsJewelNaive(G) || containsJewelNaive(GC)) return false;

  if (containsPyramid(G) || containsPyramid(GC)) return false;

  if ((containsT1(G) || containsT1(GC)) || (containsT2(G) || containsT2(GC)) ||
      (containsT3(G) || containsT3(GC)))
    return false;

  if (gatherStats) {
    StatsFactory::startTestCasePart("Get Near Cleaners");
  }

  auto Xs = getPossibleNearCleaners(G);

  if (gatherStats) {
    StatsFactory::startTestCasePart("Check Near Cleaners");
  }

  for (auto X : Xs) {
    if (containsOddHoleWithNearCleanerX(G, X)) return false;
  }

  auto XsC = getPossibleNearCleaners(GC);

  for (auto X : XsC) {
    if (containsOddHoleWithNearCleanerX(GC, X)) return false;
  }

  return true;
}

bool isPerfectGraphNaive(const Graph &G, bool gatherStats) {
  return !containsOddHoleNaive(G, gatherStats) && !containsOddHoleNaive(G.getComplement(), gatherStats);
}
