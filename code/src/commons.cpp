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

vec<pair<int, vec<int>>> getEmptyStarTriangles(const Graph &G) {
  vec<pair<int, vec<int>>> ret;
  for (int a = 0; a < G.n; a++) {
    for (int s1 = 0; s1 < G.n; s1++) {
      if (!G[a][s1])
        continue;
      for (int s2 = s1 + 1; s2 < G.n; s2++) {
        if (!G[a][s2] || G[s1][s2])
          continue;
        for (int s3 = s2 + 1; s3 < G.n; s3++) {
          if (!G[a][s3] || G[s1][s3] || G[s2][s3])
            continue;
          ret.push_back(mp(a, vec<int>{s1, s2, s3}));
        }
      }
    }
  }

  return ret;
}