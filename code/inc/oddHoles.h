#pragma once

#include "commons.h"

bool isHole(const Graph &G, const vec<int> &v);
// Checks whether graph contains hole of the given size. If not, empty vector is returned.
vec<int> findHoleOfSize(const Graph &G, int size);
bool constainsHoleOfSize(const Graph &G, int size);

// Slow - for tests. Returns empty vector if none found.
vec<int> findOddHoleNaive(const Graph &G);
bool containsOddHoleNaive(const Graph &G);

bool isT1(const Graph &G, const vec<int> &v);
vec<int> findT1(const Graph &G);

// If T2 is found returns [v1, ..., v4], P, X
// Returns three empty vectors if none found
tuple<vec<int>, vec<int>, vec<int>> findT2(const Graph &G);

// If T3 is found returns [v1, ..., v6], P, X
// Returns three empty vectors if none found
tuple<vec<int>, vec<int>, vec<int>> findT3(const Graph &G);
