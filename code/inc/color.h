#include <tuple>
#include "commons.h"

// Returns: n, m, from, to suitable for theta function from csdp. Nodes start from 1, unlike in Graph.
// isNodeRemoved is a vector of booleans. If isNodeRemoved[i] is present and true, this node is removed from
// graph before processing
tuple<int, int, vec<int>, vec<int>> getGraphEdges(const Graph &G, const vec<int> &isNodeRemoved = vec<int>());

int getTheta(const Graph &G, const vec<int> &isNodeRemoved = vec<int>());

bool isStableSet(const Graph &G, vec<int> nodes);

vec<int> getMaxCardStableSet(const Graph &G);

double color(const Graph &G, const vec<int> &);
