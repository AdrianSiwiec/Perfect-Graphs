#include "commons.h"

Graph::Graph(int n) : n(n), _tab(n) {
  for (int i = 0; i < n; i++)
    _tab[i].resize(n);
}

Graph::Graph(int n, string s) : Graph(n) {
  s.erase(remove_if(s.begin(), s.end(), ::isspace), s.end());
  if (s.size() != n * n) {
    char buff[100];
    sprintf(buff, "Graph initialization from string failed. Expected string of "
                  "size %d, got %d.",
            n * n, s.size());
    throw invalid_argument(buff);
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j)
        _tab[i][j] = 0;
      else
        _tab[i][j] = (s[i * n + j] == 'X');
    }
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (_tab[i][j] != _tab[j][i]) {
        throw invalid_argument("Graph initialization from string failed. Input "
                               "graph is not symmetrical.");
      }
    }
  }
}

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
      for (int s2 = 0; s2 < G.n; s2++) {
        if (s2 == s1)
          continue;
        if (!G[a][s2] || G[s1][s2])
          continue;
        for (int s3 = 0; s3 < G.n; s3++) {
          if (s3 == s2 || s3 == s1)
            continue;
          if (!G[a][s3] || G[s1][s3] || G[s2][s3])
            continue;
          ret.push_back(mp(a, vec<int>{s1, s2, s3}));
        }
      }
    }
  }

  return ret;
}