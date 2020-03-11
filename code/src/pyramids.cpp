#include "pyramids.h"
#include "commons.h"
#include <functional>
#include <queue>

bool checkPrerequisites(const Graph &G, const vec<int> &b, const int a, const vec<int> &s) {
  // We assume b is a Triangle and s is an EmptyStarTriangle

  int aAdjB = 0;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      if (i == j)
        continue;
      if (b[i] == s[j]) {
        return false;
      }
      if ((G.areNeighbours(b[i], s[j]) && b[j] != s[j]) || G.areNeighbours(s[i], s[j])) {
        return false;
      }
    }
  }

  for (int i = 0; i < 2; i++) {
    if (!G.areNeighbours(a, s[i]))
      return false;

    if (G.areNeighbours(a, b[i])) {
      aAdjB++;
      if (aAdjB > 1)
        return false;
      if (b[i] != s[i])
        return false;
    }
  }

  return true;
}

vec<int> findShortestPathWithPredicate(const Graph &G, int start, int end, function<bool(int)> test) {
  vector<int> father(G.n, -1);
  queue<int> Q;
  Q.push(start);

  while (!Q.empty()) {
    int v = Q.front();
    Q.pop();
    for (int i : G[v]) {
      if (father[i] == -1 && test(i)) {
        father[i] = v;
        Q.push(i);
        if (i == end)
          break;
      }
    }
  }

  if (father[end] == -1)
    return vec<int>();
  else {
    vec<int> ret;
    for (int v = end; v != start; v = father[v]) {
      ret.push_back(v);
    }
    ret.push_back(start);
    reverse(ret.begin(), ret.end());
    return ret;
  }
}

bool containsPyramide(const Graph &G) {
  auto t = findPyramide(G);
  return get<0>(t).size() > 0;
}

tuple<vec<int>, int, vec<vec<int>>> findPyramide(const Graph &G) {
  auto triangles = getTriangles(G);
  auto emptyStars = getEmptyStarTriangles(G);

  for (auto triangle : triangles) {
    for (auto eStar : emptyStars) {
      const vec<int> &b = triangle;
      const int a = eStar.st;
      const vec<int> &s = eStar.nd;

      if (!checkPrerequisites(G, b, a, s))
        continue;

      vec<int> M(G.n, 1);
      for (int i = 0; i < 3; i++)
        M[s[i]] = M[b[i]] = 0;
      vec<vec<int>> S[3]; // S[a][b][c] = S_a(b)[c], c-th vertex of the Sa(b) path
      vec<vec<int>> T[3]; 
      for (int i = 0; i < 3; i++) {
        for (int m = 0; m < G.n; m++) {
          if (!M[m])
            continue;

          auto noNeighbours = [&](int v) {
            if (v == s[i] || v == m)
              return true;
            if (!M[v])
              return false;
            for (int i = 0; i < 3; i++) {
              if (b[i] == v || s[i] == v)
                continue;
              if (G.areNeighbours(b[i], v) || G.areNeighbours(s[i], v))
                return false;
            }

            return true;
          };

          S[i][m] = findShortestPathWithPredicate(G, s[i], m, noNeighbours);
          T[i][m] = findShortestPathWithPredicate(G, m, b[i], noNeighbours);
          //TODO test it somehow
        }
      }
    }
  }
}