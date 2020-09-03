#pragma once

#include <set>
#include "commons.h"

// Returns true if G has an odd hole.
// Returns false if there is no shortest odd hole C such that X is a near-cleaner for C.
// G should contain no pyramid or jewel.
bool containsOddHoleWithNearCleanerX(const Graph &G, const set<int> &sX, const vec<vec<int>> &triplePaths,
                                     bool gatherStats = false);

bool cudaContainsOddHoleWithNearCleaners(const Graph &G, const set<set<int>> &Xs);

bool isRelevantTriple(const Graph &G, vec<int> v);

// Returns X(a, b, c) for v=[a, b, c].
// (a, b, c) should be relevant triple.
set<int> getXforRelevantTriple(const Graph &G, vec<int> v);

// 9.2
set<set<int>> getPossibleNearCleaners(const Graph &G, bool gatherStats = false);
