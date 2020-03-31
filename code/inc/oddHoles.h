#pragma once

#include "commons.h"

bool isT1(const Graph &G, const vec<int> &v);
vec<int> findT1(const Graph &G);

// If T2 is found returns [x1, ..., x4], P, X
// Returns three empty vectors if none found
tuple<vec<int>, vec<int>, vec<int>> findT2(const Graph &G);