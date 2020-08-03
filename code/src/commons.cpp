#include "commons.h"
#include <map>
#include <queue>
#include <set>
#include <unordered_set>

using std::invalid_argument;
using std::make_pair;
using std::map;
using std::queue;
using std::to_string;

Graph::Graph(int n) : n(n), _neighbours(n), _matrix(n) {
  for (int i = 0; i < n; i++) {
    _matrix[i].resize(n);
  }

  calculateFirstNextNeighbours();
}

void Graph::calculateNeighboursLists() {
  _neighbours.clear();
  for (int i = 0; i < n; i++) {
    _neighbours.push_back(vec<int>());
    for (int j = 0; j < n; j++) {
      if (areNeighbours(i, j)) _neighbours[i].push_back(j);
    }
  }
}

void Graph::calculateFirstNextNeighbours() {
  _first_neighbour = vec<int>(n, -1);
  _next_neighbour = vec<vec<int>>(n);
  for (int i = 0; i < n; i++) {
    _next_neighbour[i] = vec<int>(n, -2);
  }

  for (int i = 0; i < n; i++) {
    if (!_neighbours[i].empty()) {
      _first_neighbour[i] = _neighbours[i].front();
      _next_neighbour[i][_neighbours[i].back()] = -1;
    }

    for (int j = 1; j < _neighbours[i].size(); j++) {
      _next_neighbour[i][_neighbours[i][j - 1]] = _neighbours[i][j];
    }
  }
}

void Graph::checkSymmetry() {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (_matrix[i][j] != _matrix[j][i]) {
        throw invalid_argument("Graph is not symmetrical. (" + to_string(i) + ", " + to_string(j) + ").");
      }
    }
  }
}

Graph::Graph(int n, string s) : Graph(n) {
  s.erase(remove_if(s.begin(), s.end(), ::isspace), s.end());
  if (s.size() != n * n) {
    char buff[100];
    snprintf(buff, sizeof(buff),
             "Graph initialization from string failed. Expected string of size %d, got %d.", n * n, s.size());
    throw invalid_argument(buff);
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j || (s[i * n + j] != 'X' && s[i * n + j] != '1'))
        _matrix[i][j] = 0;
      else
        _matrix[i][j] = 1;
    }
  }

  calculateNeighboursLists();
  calculateFirstNextNeighbours();
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

  calculateFirstNextNeighbours();
  checkSymmetry();
}

int Graph::getFirstNeighbour(int a) const { return _first_neighbour[a]; }

int Graph::getNextNeighbour(int a, int b) const {
  int ret = _next_neighbour[a][b];

  if (ret == -2) {
    char buff[100];
    snprintf(buff, sizeof(buff), "Graph getNextNeighbour failed. %d is not a neighbour of %d.", b, a);

    throw invalid_argument(buff);
  }

  return ret;
}

Graph Graph::getComplement() const {
  Graph ret(n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i != j) ret._matrix[i][j] = !_matrix[i][j];
    }
  }

  ret.calculateNeighboursLists();
  ret.calculateFirstNextNeighbours();
  return ret;
}

Graph Graph::getInduced(vec<int> X) const {
  if (!isDistinctValues(X)) {
    throw invalid_argument("getInducedStrong X is not distinc values.");
  };

  set<int> S;
  for (int i : X) S.insert(i);

  Graph ret(n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (areNeighbours(i, j) && S.count(i) > 0 && S.count(j) > 0) {
        ret._matrix[i][j] = true;
      }
    }
  }

  ret.calculateNeighboursLists();
  ret.calculateFirstNextNeighbours();
  return ret;
}

Graph Graph::getInducedStrong(vec<int> X) const {
  if (!isDistinctValues(X)) {
    throw invalid_argument("getInducedStrong X is not distinc values.");
  }

  Graph ret(X.size());

  for (int i = 0; i < X.size(); i++) {
    for (int j = 0; j < X.size(); j++) {
      if (areNeighbours(X[i], X[j])) {
        ret._matrix[i][j] = true;
      }
    }
  }

  ret.calculateNeighboursLists();
  ret.calculateFirstNextNeighbours();
  return ret;
}

Graph Graph::getShuffled() const {
  vec<int> v(n);
  for (int i = 0; i < n; i++) v[i] = i;

  random_shuffle(v.begin(), v.end());

  Graph ret(n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      ret._matrix[i][j] = _matrix[v[i]][v[j]];
    }
  }

  ret.calculateNeighboursLists();
  ret.calculateFirstNextNeighbours();
  ret.checkSymmetry();
  return ret;
}

Graph Graph::getLineGraph() const {
  map<pair<int, int>, int> M;
  map<int, pair<int, int>> rM;

  for (int i = 0; i < n; i++) {
    for (int v : operator[](i)) {
      if (v > i) {
        int size = M.size();
        M[make_pair(i, v)] = size;
        rM[M.size() - 1] = make_pair(i, v);
      }
    }
  }

  vec<vec<int>> neighbours(M.size());

  for (int i = 0; i < n; i++) {
    for (int v0 : operator[](i)) {
      for (int v1 : operator[](i)) {
        if (v0 == v1) continue;
        auto p0 = v0 < i ? make_pair(v0, i) : make_pair(i, v0);
        auto p1 = v1 < i ? make_pair(v1, i) : make_pair(i, v1);
        neighbours[M[p0]].push_back(M[p1]);
      }
    }
  }

  return Graph(neighbours);
}

vec<vec<int>> getTriangles(const Graph &G) {
  vec<vec<int>> ret;
  for (int i = 0; i < G.n; i++) {
    for (int j : G[i]) {
      for (int k : G[j]) {
        if (i < j && j < k && G.areNeighbours(i, k)) ret.push_back(vec<int>{i, j, k});
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
        if (s2 == s1 || G.areNeighbours(s1, s2)) continue;
        for (int s3 : G[a]) {
          if (s3 == s2 || s3 == s1) continue;
          if (G.areNeighbours(s1, s3) || G.areNeighbours(s2, s3)) continue;
          ret.push_back(mp(a, vec<int>{s1, s2, s3}));
        }
      }
    }
  }

  return ret;
}

bool isComplete(const Graph &G, const vec<int> &X, int v) {
  bool isComplete = true;
  for (int i : X) {
    if (v == i || !G.areNeighbours(v, i)) {
      isComplete = false;
      break;
    }
  }

  return isComplete;
}

vec<int> getCompleteVertices(const Graph &G, const vec<int> &X) {
  vec<int> ret;
  for (int v = 0; v < G.n; v++) {
    if (isComplete(G, X, v)) ret.push_back(v);
  }

  return ret;
}

void dfsWith(const Graph &G, vec<int> &visited, int start, function<void(int)> action,
             function<bool(int)> test) {
  if (visited[start]) return;
  action(start);
  visited[start] = true;
  for (int i : G[start]) {
    if (!visited[i] && test(i)) dfsWith(G, visited, i, action, test);
  }
}

vec<vec<int>> getComponents(const Graph &G) {
  vec<int> visited(G.n);
  vec<vec<int>> components;
  for (int i = 0; i < G.n; i++) {
    if (!visited[i]) {
      components.push_back(vec<int>());
      dfsWith(G, visited, i, [&](int v) -> void { components.back().push_back(v); });
    }
  }

  return components;
}

vec<vec<int>> getComponentsOfInducedGraph(const Graph &G, const vec<int> &X) {
  set<int> sX(X.begin(), X.end());
  vec<vec<int>> components = getComponents(G.getInduced(X));
  vec<vec<int>> ret;
  for (auto c : components) {
    bool isOk = true;
    for (auto v : c) {
      if (sX.count(v) == 0) {
        isOk = false;
        break;
      }
    }
    if (isOk) ret.push_back(c);
  }

  return ret;
}

bool isAllZeros(const vec<int> &v) {
  for (int a : v)
    if (a != 0) return false;
  return true;
}

// This is a bottleneck in both Perfect and Naive, so it is optimized for speed over readibility
bool isDistinctValues(const vec<int> &v) {
  // TODO(Adrian) Create fallback for values outside of [-2,997]
  static vec<int> stamp(1000, 0);
  static int counter = 1;

  counter++;
  if (counter > std::numeric_limits<int>::max() - 2) {
    stamp = vec<int>(1000, 0);
    counter = 1;
  }

  for (int i = 0; i < v.size(); ++i) {
    //+2 to accomodate values of -1 and -2
    if (stamp[v[i] + 2] == counter) return false;
    stamp[v[i] + 2] = counter;
  }
  return true;
}

int countNonZeros(const vec<int> &v) {
  int res = 0;

  for (int i : v)
    if (i) res++;

  return res;
}

vec<int> getPrefSum(const vec<int> &v) {
  if (v.empty()) return vec<int>();

  vec<int> ret(v.size());
  ret[0] = v[0];
  for (int i = 1; i < v.size(); i++) {
    ret[i] = ret[i - 1] + v[i];
  }

  return ret;
}

vec<int> getComplementNodesVec(int n, const vec<int> &X) {
  for (int i = 1; i < X.size(); i++) {
    if (X[i - 1] >= X[i]) {
      throw invalid_argument("X for getComplementNodesVec should be sorted");
    }
  }

  vec<int> res;

  res.reserve(n - X.size());
  int wsk = 0;
  for (int i = 0; i < n; i++) {
    while (wsk < X.size() && X[wsk] < i) wsk++;

    if (wsk >= X.size() || X[wsk] != i) res.push_back(i);
  }

  return res;
}

void nextTupleInPlace(vec<int> &v, int max) {
  v[0]++;
  for (int i = 0; i < v.size() && v[i] >= max; i++) {
    v[i] = 0;
    if (i + 1 < v.size()) v[i + 1]++;
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
  cout << G.n << endl;
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
    for (int j : G[i]) cout << j << " ";
    cout << endl;
  }

  return os;
}

vec<int> findShortestPathWithPredicate(const Graph &G, int start, int end, function<bool(int)> test) {
  if (start == end) return vec<int>{start};

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
        if (i == end) break;
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

vec<vec<vec<int>>> allShortestPathsWithPredicate(const Graph &G, function<bool(int)> test) {
  int inf = G.n + 10;

  vec<vec<int>> dist(G.n, vec<int>(G.n, inf));
  vec<vec<int>> lastOnPath(G.n, vec<int>(G.n, -1));
  for (int i = 0; i < G.n; i++) {
    dist[i][i] = 0;
    for (int j : G[i]) {
      dist[i][j] = 1;
      lastOnPath[i][j] = i;
    }
  }

  for (int k = 0; k < G.n; k++) {
    for (int i = 0; i < G.n; i++) {
      if (i == k) continue;
      for (int j = 0; j < G.n; j++) {
        if (i == j || k == j) continue;
        if (dist[i][j] > dist[i][k] + dist[k][j] && test(k)) {
          dist[i][j] = dist[i][k] + dist[k][j];
          lastOnPath[i][j] = lastOnPath[k][j];
        }
      }
    }
  }

  vec<vec<vec<int>>> R(G.n, vec<vec<int>>(G.n, vec<int>()));
  for (int i = 0; i < G.n; i++) {
    for (int j = 0; j < G.n; j++) {
      if (dist[i][j] == inf) continue;

      R[i][j].reserve(dist[i][j] + 1);
      R[i][j].push_back(j);
      if (i == j) continue;

      // TODO(Adrian) smarter: if(tmp < i) ~insert(R[i][tmp])
      int tmp = lastOnPath[i][j];
      R[i][j].push_back(tmp);
      while (tmp != i) {
        tmp = lastOnPath[i][tmp];
        R[i][j].push_back(tmp);
      }

      std::reverse(R[i][j].begin(), R[i][j].end());
    }
  }

  return R;
}

bool isAPath(const Graph &G, const vec<int> &v, bool isCycleOk, bool areChordsOk) {
  if (v.size() <= 1) return false;

  if (!isDistinctValues(v)) return false;

  for (int i = v.size() - 1; i > 0; i--) {
    for (int j = 0; j < i; j++) {
      if (j == i - 1) {
        if (!G.areNeighbours(v[i], v[j])) return false;
      } else if (!areChordsOk) {
        if (isCycleOk && i == v.size() - 1 && j == 0) continue;
        if (G.areNeighbours(v[i], v[j])) return false;
      }
    }
  }

  return true;
}

vec<int> getFirstPath(const Graph &G, int len, bool isCycleOk) {
  vec<int> ret;
  for (int start = 0; start < G.n; start++) {
    ret = {start};
    while (ret.size() < len) {
      ret.push_back(G.getFirstNeighbour(ret.back()));

      if (ret.back() == -1) {
        ret.clear();
        break;
      }
    }

    if (ret.size() == len) return ret;
  }

  return ret;
}

void nextPathInPlaceInternal(const Graph &G, vec<int> &v, int len, bool isCycleOk, bool areChordsOk) {
  while (true) {
    if (v.back() == -1) {
      v.pop_back();
      if (v.size() == 1) {
        v[0]++;
        if (v[0] >= G.n) {
          v = {};
          return;
        }
        continue;
        // return nextPathInPlaceInternal(G, v, len, isCycleOk);
      } else {
        v.back() = G.getNextNeighbour(v[v.size() - 2], v.back());
        continue;
        // return nextPathInPlaceInternal(G, v, len, isCycleOk);
      }
    }

    if (v.size() < len) {
      v.push_back(G.getFirstNeighbour(v.back()));
      if (v.size() == len && isAPath(G, v, isCycleOk, areChordsOk)) {
        return;
      } else {
        continue;
        // return nextPathInPlaceInternal(G, v, len, isCycleOk);
      }
    }

    do {
      v.back() = G.getNextNeighbour(v[v.size() - 2], v.back());
    } while (v.back() != -1 && !isAPath(G, v, isCycleOk, areChordsOk));

    if (v.back() == -1) continue;

    return;
  }
}

void nextPathInPlace(const Graph &G, vec<int> &v, int len, bool isCycleOk, bool areChordsOk) {
  if (len <= 1) {
    throw invalid_argument("Length of next path must be at least 2");
  }
  if (!v.empty() && v.size() != len) {
    throw invalid_argument("Length of next path must be equal to length of given path.");
  }

  if (v.empty()) {
    v = {0};
  }

  v.reserve(len);

  return nextPathInPlaceInternal(G, v, len, isCycleOk, areChordsOk);
}
