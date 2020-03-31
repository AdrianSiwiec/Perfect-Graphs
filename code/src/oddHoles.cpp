#include "oddHoles.h"
#include "commons.h"
#include <set>

bool isT1(const Graph &G, const vec<int> &v) {
  if (v.size() != 5)
    return false;

  if (!isDistinctValues(v))
    return false;

  for (int i : v)
    if (i < 0 || i >= G.n)
      return false;

  for (int i = 0; i < 5; i++) {
    for (int j = i + 1; j < 5; j++) {
      if (abs(i - j) == 1 || abs(i - j) == 4) {
        if (!G.areNeighbours(v[i], v[j]))
          return false;
      } else {
        if (G.areNeighbours(v[i], v[j]))
          return false;
      }
    }
  }

  return true;
}
vec<int> findT1(const Graph &G) {
  vec<int> v(5);
  do {
    if (isT1(G, v))
      return v;

    nextTupleInPlace(v, G.n);
  } while (!isAllZeros(v));

  return vec<int>();
}

tuple<vec<int>, vec<int>, vec<int>> findT2(const Graph &G) {
  for (int v1 = 0; v1 < G.n; v1++) {
    for (int v2 : G[v1]) {
      for (int v3 : G[v2]) {
        if (v3 == v1)
          continue;
        for (int v4 : G[v3]) {
          if (v4 == v1 || v4 == v2)
            continue;
          auto Y = getCompleteVertices(G, {v1, v2, v4});
          auto antiCY = getComponents(G.getComplement().getInduced(Y));

          set<int> sY;
          for (int i : Y)
            sY.insert(i);

          for (auto X : antiCY) {
            bool containsOnlyY = true;
            for (auto v : X) {
              if (sY.count(v) == 0) {
                containsOnlyY = false;
                break;
              }
            }
            if (!containsOnlyY)
              continue;

            auto P = findShortestPathWithPredicate(G, v1, v4, [&](int v) -> bool {
              if (v == v2 || v == v3)
                return false;
              if (G.areNeighbours(v, v2) || G.areNeighbours(v, v3))
                return false;
              if (isComplete(G, X, v))
                return false;

              return true;
            });

            if (!P.empty())
              return make_tuple(vec<int>{v1, v2, v3, v4}, P, X);
          }
        }
      }
    }
  }

  return make_tuple(vec<int>(), vec<int>(), vec<int>());
}