#pragma once

#include "commons.h"
#include <functional>

bool checkPrerequisites(const Graph &G, const vec<int> &b, const int a, const vec<int> &s);

// Finds shortest path from start to end in G, where every vertex inside satisfies predicate.
// Returns empty vector if none exist
vec<int> findShortestPathWithPredicate(const Graph &G, int start, int end, function<bool(int)> test);

// Returns empty vector, 0 and emty vector if none found
tuple<vec<int>, int, vec<vec<int>>> findPyramide(const Graph &G);

bool containsPyramide(const Graph &G);