#pragma once

#include "commons.h"

bool checkPrerequisites(const Graph &G, const vec<int> &b, const int a,
                        const vec<int> &s);

// Returns empty vector, 0 and emty vector if none found
tuple<vec<int>, int, vec<vec<int>>> findPyramide(const Graph &G);

bool containsPyramide(const Graph &G) {
  auto t = findPyramide(G);
  return get<0>(t).size() > 0;
}