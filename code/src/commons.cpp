#include "commons.h"
#include <queue>
#include <set>

Graph::Graph(int n) : n(n), _neighbours(n), _matrix(n) {
  for (int i = 0; i < n; i++) {
    _matrix[i].resize(n);
  }
}

void Graph::checkSymmetry() {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (_matrix[i][j] != _matrix[j][i]) {
        throw invalid_argument("Graph initialization from string failed. Input graph is not symmetrical.");
      }
    }
  }
}

Graph::Graph(int n, string s) : Graph(n) {
  s.erase(remove_if(s.begin(), s.end(), ::isspace), s.end());
  if (s.size() != n * n) {
    char buff[100];
    sprintf(buff, "Graph initialization from string failed. Expected string of size %d, got %d.", n * n,
            s.size());
    throw invalid_argument(buff);
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j || s[i * n + j] != 'X')
        _matrix[i][j] = 0;
      else {
        _neighbours[i].push_back(j);
        _matrix[i][j] = 1;
      }
    }
  }

  checkSymmetry();
}

Graph::Graph(vec<vec<int>> neighbours) : n(neighbours.size()), _neighbours(neighbours) {
  _matrix.resize(n);
  for (int i = 0; i < n; i++) {
    _matrix[i].resize(n);
  }

  for (int i = 0; i < n; i++) {
    for (int j : _neighbours[i]) {
      if (j < 0 || j >= n) {
        throw invalid_argument("Graph initialization from neighbours array failed. Neighbour out of range.");
      }
      _matrix[i][j] = true;
    }
  }

  checkSymmetry();
}

vec<vec<int>> getTriangles(const Graph &G) {
  vec<vec<int>> ret;
  for (int i = 0; i < G.n; i++) {
    for (int j : G[i]) {
      for (int k : G[j]) {
        if (i < j && j < k && G.areNeighbours(i, k))
          ret.push_back(vec<int>{i, j, k});
      }
    }
  }

  return ret;
}

vec<pair<int, vec<int>>> getEmptyStarTriangles(const Graph &G) {
  vec<pair<int, vec<int>>> ret;
  for (int a = 0; a < G.n; a++) {
    for (int s1 : G[a]) {
      for (int s2 : G[a]) {
        if (s2 == s1 || G.areNeighbours(s1, s2))
          continue;
        for (int s3 : G[a]) {
          if (s3 == s2 || s3 == s1)
            continue;
          if (G.areNeighbours(s1, s3) || G.areNeighbours(s2, s3))
            continue;
          ret.push_back(mp(a, vec<int>{s1, s2, s3}));
        }
      }
    }
  }

  return ret;
}

bool isAllZeros(const vec<int> &v) {
  for (int a : v)
    if (a != 0)
      return false;
  return true;
}
bool isDistinctValues(const vec<int> &v) {
  set<int> s;
  for (int i : v)
    s.insert(i);

  return s.size() == v.size();
}

void nextTupleInPlace(vec<int> &v, int max) {
  v[0]++;
  for (int i = 0; i < v.size() && v[i] >= max; i++) {
    v[i] = 0;
    if (i + 1 < v.size())
      v[i + 1]++;
  }
}

vec<int> nextTuple(vec<int> v, int max) {
  nextTupleInPlace(v, max);

  return v;
}

vec<vec<int>> generateTuples(int size, int max) {
  vec<vec<int>> ret;
  vec<int> current = vec<int>(size);
  do {
    ret.push_back(current);
    current = nextTuple(current, max);
  } while (!isAllZeros(current));

  return ret;
}

ostream &operator<<(ostream &os, Graph const &G) {
  for (int i = 0; i < G.n; i++) {
    for (int j = 0; j < G.n; j++) {
      if (G.areNeighbours(i, j))
        cout << "X";
      else
        cout << ".";
    }
    cout << endl;
  }

  for (int i = 0; i < G.n; i++) {
    cout << i << ": ";
    for (int j : G[i])
      cout << j << " ";
    cout << endl;
  }

  return os;
}

vec<int> findShortestPathWithPredicate(const Graph &G, int start, int end, function<bool(int)> test) {
  if (start == end)
    return vec<int>{start};

  vector<int> father(G.n, -1);
  queue<int> Q;
  Q.push(start);

  while (!Q.empty()) {
    int v = Q.front();
    Q.pop();
    for (int i : G[v]) {
      if (father[i] == -1 && (test(i) || i == end)) {
        father[i] = v;
        Q.push(i);
        if (i == end)
          break;
      }
    }
  }

  if (father[end] == -1) {
    return vec<int>();
  } else {
    vec<int> ret;
    for (int v = end; v != start; v = father[v]) {
      ret.push_back(v);
    }
    ret.push_back(start);
    reverse(ret.begin(), ret.end());

    return ret;
  }
}