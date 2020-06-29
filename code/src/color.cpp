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

double color(const Graph &G) { return getTheta(G); }