#include "color.h"
#include <tuple>
#include "perfect.h"
using namespace std;

extern "C" {
#include "theta.h"
}

tuple<int, int, vec<int>, vec<int>> getGraphEdges(const Graph &G) {
  vec<int> from;
  vec<int> to;

  int m = 0;
  for (int i = 0; i < G.n; i++) {
    for (auto j : G[i]) {
      if (j > i) {
        from.push_back(i + 1);
        to.push_back(j + 1);
        m++;
      }
    }
  }

  return {G.n, m, from, to};
}

int getTheta(const Graph &G) {
  auto e = getGraphEdges(G);
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

  return thInt;
}

double color(const Graph &G) { return getTheta(G); }