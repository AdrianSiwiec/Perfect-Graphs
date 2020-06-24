#include "perfect.h"
#include "commons.h"
#include "jewels.h"
#include "nearCleaners.h"
#include "oddHoles.h"
#include "pyramids.h"

bool isPerfectGraph(const Graph &G) {
  Graph GC = G.getComplement();
  if (containsJewelNaive(G) || containsJewelNaive(GC)) return false;

  if (containsPyramid(G) || containsPyramid(GC)) return false;

  if ((containsT1(G) || containsT1(GC)) || (containsT2(G) || containsT2(GC)) ||
      (containsT3(G) || containsT3(GC)))
    return false;

  auto Xs = getPossibleNearCleaners(G);

  for (auto X : Xs) {
    if (containsOddHoleWithNearCleanerX(G, X)) return false;
  }

  auto XsC = getPossibleNearCleaners(GC);
  for (auto X : XsC) {
    if (containsOddHoleWithNearCleanerX(GC, X)) return false;
  }

  return true;
}

bool isPerfectGraphNaive(const Graph &G) {
  return !containsOddHoleNaive(G) && !containsOddHoleNaive(G.getComplement());
}
