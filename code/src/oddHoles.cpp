#include "oddHoles.h"
#include "commons.h"
#include <set>

bool isHole(const Graph &G, const vec<int> &v) {
  if (v.size() <= 3)
    return false;

  if (!isDistinctValues(v))
    return false;

  for (int i : v)
    if (i < 0 || i >= G.n)
      return false;

  for (int i = 0; i < v.size(); i++) {
    for (int j = i + 1; j < v.size(); j++) {
      if (abs(i - j) == 1 || abs(i - j) == (v.size() - 1)) {
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

vec<int> findHoleOfSize(const Graph &G, int size) {
  if (size <= 3)
    return vec<int>();

  vec<int> v;
  while (true) {
    nextPathInPlace(G, v, size, true);
    if (v.size() == size && isHole(G, v))
      return v;

    if (v.empty())
      break;
  }

  return vec<int>();
}

bool constainsHoleOfSize(const Graph &G, int size) { return !findHoleOfSize(G, size).empty(); }

vec<int> findOddHoleNaive(const Graph &G) {
  for (int size = 5; size <= G.n; size += 2) {
    auto v = findHoleOfSize(G, size);
    if (!v.empty())
      return v;
  }

  return vec<int>();
}

bool containsOddHoleNaive(const Graph &G) { return !findOddHoleNaive(G).empty(); }

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
          if (!isAPath(G, vec<int>{v1, v2, v3, v4}))
            continue;
          auto Y = getCompleteVertices(G, {v1, v2, v4});
          auto antiCY = getComponentsOfInducedGraph(G.getComplement(), Y);

          for (auto X : antiCY) {
            auto P = findShortestPathWithPredicate(G, v1, v4, [&](int v) -> bool {
              if (v == v2 || v == v3)
                return false;
              if (G.areNeighbours(v, v2) || G.areNeighbours(v, v3))
                return false;
              if (isComplete(G, X, v))
                return false;

              return true;
            });

            if (!P.empty()) {
              return make_tuple(vec<int>{v1, v2, v3, v4}, P, X);
            }
          }
        }
      }
    }
  }

  return make_tuple(vec<int>(), vec<int>(), vec<int>());
}

tuple<vec<int>, vec<int>, vec<int>> findT3(const Graph &G) {
  for (int v1 = 0; v1 < G.n; v1++) {
    for (int v2 : G[v1]) {
      for (int v5 = 0; v5 < G.n; v5++) {
        if (v5 == v1 || v5 == v2 || G.areNeighbours(v5, v1) || G.areNeighbours(v5, v2))
          continue;

        auto Y = getCompleteVertices(G, {v1, v2, v5});
        auto antiCY = getComponentsOfInducedGraph(G.getComplement(), Y);
        for (auto X : antiCY) {
          set<int> Fprim;
          vec<int> visited(G.n);
          dfsWith(G, visited, v5, [&](int v) -> void { Fprim.insert(v); },
                  [&](int v) -> bool {
                    if (G.areNeighbours(v1, v) || G.areNeighbours(v2, v))
                      return false;
                    if (isComplete(G, X, v))
                      return false;

                    return true;
                  });

          set<int> F(F.begin(), F.end());
          for (int fp : Fprim) {
            for (int v : G[fp]) {
              if (F.count(v) == 0 && isComplete(G, X, v) && !G.areNeighbours(v, v1) &&
                  !G.areNeighbours(v, v2) && !G.areNeighbours(v, v5))
                F.insert(v);
            }
          }

          for (int v4 : G[v1]) {
            if (G.areNeighbours(v4, v2) || G.areNeighbours(v4, v5))
              continue;
            int v6 = -1;
            for (int f : F) {
              if (G.areNeighbours(v4, f)) {
                v6 = f;
                break;
              }
            }
            if (v6 == -1)
              continue;

            bool v4HasNonNeighbourInX = false;
            for (int x : X) {
              if (!G.areNeighbours(v4, x)) {
                v4HasNonNeighbourInX = true;
                break;
              }
            }
            if (!v4HasNonNeighbourInX)
              continue;

            for (int v3 : G[v2]) {
              if (!G.areNeighbours(v3, v4) || !G.areNeighbours(v3, v5) || G.areNeighbours(v3, v1))
                continue;

              bool v3HasNonNeighbourInX = false;
              for (int x : X) {
                if (!G.areNeighbours(v3, x)) {
                  v3HasNonNeighbourInX = true;
                  break;
                }
              }

              auto P =
                  findShortestPathWithPredicate(G, v6, v5, [&](int v) -> bool { return Fprim.count(v) > 0; });

              if (P.empty()) {
                throw logic_error("Algorithm Error: Could not find path P in T3.");
              }

              return make_tuple(vec<int>{v1, v2, v3, v4, v5, v6}, P, X);
            }
          }
        }
      }
    }
  }

  return make_tuple(vec<int>(), vec<int>(), vec<int>());
}