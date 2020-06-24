#include "nearCleaners.h"
#include <set>
#include "commons.h"

bool containsOddHoleWithNearCleanerX(const Graph &G, const set<int> &sX) {
  auto R = allShortestPathsWithPredicate(G, [&](int v) -> bool { return sX.count(v) == 0; });

  for (int y1 = 0; y1 < G.n; y1++) {
    if (sX.count(y1) > 0) continue;

    vec<int> x;
    while (1) {
      nextPathInPlace(G, x, 3);
      if (x.empty()) break;

      bool containsY1 = false;
      for (int i = 0; i < 3; i++) {
        if (x[i] == y1) {
          containsY1 = true;
          break;
        }
      }
      if (containsY1) continue;

      int x1 = x[0], x3 = x[1], x2 = x[2];
      if (R[x1][y1].empty() || R[x2][y1].empty()) continue;

      int y2 = R[x2][y1][R[x2][y1].size() - 2];

      int n = R[x2][y1].size();
      if (R[x1][y1].size() + 1 != n || R[x1][y2].size() != n) continue;

      if ((R[x3][y1].size() < n) || (R[x3][y2].size() < n)) continue;

      cout << "sX: " << sX << endl;
      cout << "x1: " << x1 << endl;
      cout << "x2: " << x2 << endl;
      cout << "x3: " << x3 << endl;
      cout << "y1: " << y1 << endl;
      cout << "y2: " << y2 << endl;
      cout << "R(x1, y1): " << R[x1][y1] << endl;
      cout << "R(x2, y1): " << R[x2][y1] << endl;
      cout << "R(x3, y1): " << R[x3][y1] << endl;
      cout << "R(x3, y2): " << R[x3][y2] << endl;
      cout << G << endl;

      return true;
    }
  }

  return false;
}

bool isRelevantTriple(const Graph &G, vec<int> v) {
  if (v.size() != 3) return false;

  for (int i = 0; i < 3; i++) {
    if (v[i] < 0 || v[i] >= G.n) return false;
  }

  int a = v[0], b = v[1], c = v[2];

  if (a == b || G.areNeighbours(a, b)) return false;

  if (G.areNeighbours(a, c) && G.areNeighbours(b, c)) return false;

  return true;
}

set<int> getXforRelevantTriple(const Graph &G, vec<int> v) {
  int a = v[0], b = v[1], c = v[2];

  auto antiCompsNab = getComponentsOfInducedGraph(G.getComplement(), getCompleteVertices(G, {a, b}));
  int r = 0;
  for (auto comp : antiCompsNab) {
    if (comp.size() <= r) continue;

    bool containsNonNofC = false;
    for (int v : comp) {
      if (!G.areNeighbours(c, v)) containsNonNofC = true;
    }

    if (containsNonNofC) r = comp.size();
  }

  vec<int> Y;
  for (auto comp : antiCompsNab) {
    if (comp.size() > r) {
      Y.insert(Y.end(), comp.begin(), comp.end());
    }
  }

  vec<int> W;
  for (auto comp : antiCompsNab) {
    for (int v : comp) {
      if (!G.areNeighbours(v, c)) W = comp;
    }
  }
  W.push_back(c);

  W.insert(W.end(), Y.begin(), Y.end());
  auto Z = getCompleteVertices(G, W);

  set<int> sX(Y.begin(), Y.end());
  sX.insert(Z.begin(), Z.end());

  return sX;
}

set<set<int>> getPossibleNearCleaners(const Graph &G) {
  vec<vec<int>> Ns;
  for (int u = 0; u < G.n; u++) {
    for (int v : G[u]) {
      Ns.push_back(getCompleteVertices(G, {u, v}));
    }
  }

  vec<set<int>> Xs;
  for (int a = 0; a < G.n; a++) {
    for (int b = 0; b < G.n; b++) {
      if (a == b || G.areNeighbours(a, b)) continue;
      for (int c = 0; c < G.n; c++) {
        if (isRelevantTriple(G, {a, b, c})) {
          Xs.push_back(getXforRelevantTriple(G, {a, b, c}));
        }
      }
    }
  }

  set<set<int>> res;
  for (auto N : Ns) {
    for (auto X : Xs) {
      set<int> tmpS(X);
      tmpS.insert(N.begin(), N.end());
      res.insert(tmpS);
    }
  }

  return res;
}
