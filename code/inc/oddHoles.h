#pragma once

#include "commons.h"

bool isHole(const Graph &G, const vec<int> &v);
// Checks whether graph contains hole of the given size
bool constainsHoleOfSize(const Graph &G, int size);
bool containsOddHoleNaive(const Graph &G);

bool isT1(const Graph &G, const vec<int> &v);
vec<int> findT1(const Graph &G);

// If T2 is found returns [x1, ..., x4], P, X
// Returns three empty vectors if none found
tuple<vec<int>, vec<int>, vec<int>> findT2(const Graph &G);

