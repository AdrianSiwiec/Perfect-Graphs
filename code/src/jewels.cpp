#include "jewels.h"
#include "commons.h"
#include <set>

bool isJewel(const Graph &G, const vec<int> &v) {
  if (v.size() != 5)
    return false;

  set<int> S;
  for (int i : v)
    S.insert(i);
  if (S.size() != 5)
    return false;

  for (int i = 0; i < 5; i++) {
    if (!G.areNeighbours(v[i], v[(i + 1) % 5]))
      return false;
  }

  if (G.areNeighbours(v[0], v[2]) || G.areNeighbours(v[1], v[3]) || G.areNeighbours(v[0], v[3]))
    return false;

  auto noNeighbours = [&](int p) {
    if (p == v[1] || p == v[2] || p == v[4])
      return false;

    for (int i : G[p]) {
      if (i == v[1] || i == v[2] || i == v[4])
        return false;
    }
    return true;
  };

  vec<int> P = findShortestPathWithPredicate(G, v[0], v[3], noNeighbours);

  if (P.empty())
    return false;

  return true;
}

// returns [v1, ..., v5] or empty vector if none found
vec<int> findJewelNaive(const Graph &G) {
  return vec<int>();
  ;
}