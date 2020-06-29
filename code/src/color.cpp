#include "color.h"
#include <tuple>
#include "commons.h"
#include "perfect.h"
using namespace std;

extern "C" {
#include "theta.h"
}

tuple<int, int, vec<int>, vec<int>> getGraphEdges(const Graph &G, const vec<int> &isNodeRemoved) {
  vec<int> from;
  vec<int> to;

  int m = 0;
  for (int i = 0; i < G.n; i++) {
    if (isNodeRemoved.size() > i && isNodeRemoved[i]) continue;

    for (auto j : G[i]) {
      if (isNodeRemoved.size() > j && isNodeRemoved[j]) continue;

      if (j > i) {
        from.push_back(i + 1);
        to.push_back(j + 1);
        m++;
      }
    }
  }

  return {G.n, m, from, to};
}

int getTheta(const Graph &G, const vec<int> &isNodeRemoved) {
  if (G.n == 0) {
    return 0;
  }

  auto e = getGraphEdges(G, isNodeRemoved);
  double th = theta(get<0>(e), get<1>(e), get<2>(e).data(), get<3>(e).data());

  if (th == -1) {
    throw logic_error("Theta returned -1");
  }

  int thInt = th + 0.5;

  double eps = 0.001;
  if (abs(th - thInt) > eps) {
    if (!isPerfectGraph(G))
      throw invalid_argument("Argument for getTheta should be a perfect graph. Non-perfect graph was given");
    else
      throw logic_error("Theta returned non-integer for a Perfect Graph");
  }

  return thInt - countNonZeros(isNodeRemoved);
}

int getOmega(const Graph &G) {
  // TODO(Adrian) faster?

  return getTheta(G.getComplement());
}

bool isStableSet(const Graph &G, vec<int> nodes) {
  if (!isDistinctValues(nodes)) return false;

  for (int i = 0; i < nodes.size(); i++) {
    for (int j = i + 1; j < nodes.size(); j++) {
      if (G.areNeighbours(nodes[i], nodes[j])) return false;
    }
  }

  return true;
}

vec<int> getMaxCardStableSet(const Graph &G) {
  int thetaG = getTheta(G);

  vec<int> isNodeRemoved(G.n, 0);
  vec<int> res;

  for (int i = 0; i < G.n; i++) {
    isNodeRemoved[i] = 1;
    int newTheta = getTheta(G, isNodeRemoved);

    if (newTheta != thetaG) {
      isNodeRemoved[i] = 0;
      res.push_back(i);
    }
  }

  return res;
}

vec<int> getMaxCardClique(const Graph &G) { return getMaxCardStableSet(G.getComplement()); }

vec<int> getSSIntersectingCliques(const Graph &G, vec<vec<int>> K) {
  vec<int> c(G.n);

  for (auto k : K) {
    for (auto v : k) {
      c[v]++;
    }
  }

  vec<int> prefC = getPrefSum(c);

  vec<vec<int>> nneighbors(prefC.back());

  for (int i = 0; i < G.n; i++) {
    for (int j = i + 1; j < G.n; j++) {
      if (G.areNeighbours(i, j)) {
        for (int ni = i == 0 ? 0 : prefC[i - 1]; ni < prefC[i]; ni++) {
          for (int nj = prefC[j - 1]; nj < prefC[j]; nj++) {
            nneighbors[ni].push_back(nj);
            nneighbors[nj].push_back(ni);
          }
        }
      }
    }
  }

  Graph nG(nneighbors);

  vec<int> nSS = getMaxCardStableSet(Graph(nneighbors));

  vec<int> ret;
  int wsk = 0;
  for (int nSSnode : nSS) {
    while (wsk < G.n && prefC[wsk] <= nSSnode) wsk++;

    if (ret.empty() || ret.back() != wsk) {
      ret.push_back(wsk);
    }
  }

  return ret;
}

vec<int> getSSIntersectingAllMaxCardCliques(const Graph &G) {
  vec<vec<int>> K;
  K.push_back(getMaxCardClique(G));

  int omegaG = getOmega(G);

  while (true) {
    vec<int> S = getSSIntersectingCliques(G, K);

    vec<int> compS = getComplementNodesVec(G.n, S);

    Graph Gprim = G.getInducedStrong(compS);

    if (getOmega(Gprim) < omegaG) {
      return S;
    } else {
      K.push_back(getMaxCardClique(Gprim));

      for (int i = 0; i < K.back().size(); i++) {
        K.back()[i] = compS[K.back()[i]];
      }
    }
  }
}

vec<int> color(const Graph &G) {
  if (G.n == 0) return vec<int>();
  if (G.n == 1) return vec<int>{0};

  vec<int> ret(G.n);

  vec<int> SS = getSSIntersectingAllMaxCardCliques(G);
  vec<int> compSS = getComplementNodesVec(G.n, SS);

  vec<int> colorCompSS = color(G.getInducedStrong(compSS));
  for (int i = 0; i < compSS.size(); i++) {
    ret[compSS[i]] = colorCompSS[i] + 1;
  }

  return ret;
}