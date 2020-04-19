#pragma once

#include "commons.h"
#include <set>

// Returns true if G has an odd hole.
// Returns false if there is no shortest odd hole C such that X is a near-cleaner for C.
// G should contain no pyramid or jewel.
bool containsOddHoleWithNearCleanerX(const Graph &G, const vec<int> &X);

bool isRelevantTriple(const Graph &G, vec<int> v);

// Returns X(a, b, c) for v=[a, b, c]. 
// (a, b, c) should be relevant triple.
vec<int> getXforRelevantTriple(const Graph &G, vec<int> v);

// 9.2
set<vec<int>> getPossibleNearCleaners(const Graph &G);