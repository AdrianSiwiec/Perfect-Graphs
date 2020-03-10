#include "commons.h"

vec<vec<int>> getTriangles(const Graph &G) {
  vec<vec<int>> ret;
  for (int i = 0; i < G.n; i++) {
    for (int j = i + 1; j < G.n; j++) {
      if (!G[i][j])
        continue;
      for (int k = j + 1; k < G.n; k++) {
        if (G[k][j] && G[k][i])
          ret.push_back(vec<int>{i, j, k});
      }
    }
  }

  return ret;
}