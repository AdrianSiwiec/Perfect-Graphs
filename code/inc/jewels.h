#pragma once

#include "commons.h"

bool isJewel(const Graph &G, const vec<int> &v);

// returns [v1, ..., v5] or empty vector if none found
// returned[0] = v1, returned[1] = v2, ...
vec<int> findJewelNaive(const Graph &G);