#include <tuple>
#include "commons.h"

// Returns: n, m, from, to suitable for theta function from csdp. Nodes start from 1, unlike in Graph.
tuple<int, int, vec<int>, vec<int>> getGraphEdges(const Graph &G);

double color(const Graph &G);
