#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

using std::cout;
using std::endl;
using std::function;
using std::ostream;
using std::pair;
using std::set;
using std::string;
using std::tuple;
using std::vector;

#define mp make_pair
#define st first
#define nd second

template <typename T>
struct vec : public vector<T> {
  using vector<T>::vector;

  // Range checking
  T &operator[](int i) { return vector<T>::at(i); }
  const T &operator[](int i) const { return vector<T>::at(i); }
};
template <typename T>
ostream &operator<<(ostream &os, vec<T> const &v) {
  os << "[";
  for (int i = 0; i < v.size(); i++) {
    os << v[i] << (i + 1 < v.size() ? ", " : "");
  }
  os << "]";

  return os;
}

struct Graph {
  int n;

  explicit Graph(int n);
  Graph(int n, string s);
  explicit Graph(vec<vec<int>> neighbours);
  // List of neighbors is not necessarily sorted
  vec<int> &operator[](int index) { return _neighbours[index]; }
  const vec<int> &operator[](int index) const { return _neighbours[index]; }
  bool operator==(const Graph &c) const { return _matrix == c._matrix; }
  bool areNeighbours(int a, int b) const { return _matrix[a][b]; }
  // Returns first neighbor of a. Returns -1 if a has no neighbors.
  int getFirstNeighbour(int a) const;
  // Returns next neighbor of a, after b. Returns -1 if b is last, throws invalid_argument if b is not a
  // neighbor of a. Guaranteed to be consistent with G[a] ordering.
  int getNextNeighbour(int a, int b) const;

  Graph getComplement() const;
  // Returns G' - Graph induced on X. Set of vertices is left the same, and edge is in G' if it is in G and
  // both it's ends are in X.
  Graph getInduced(vec<int> X) const;
  Graph getShuffled() const;
  Graph getLineGraph() const;

 private:
  vec<vec<int>> _neighbours;
  vec<vec<int>> _matrix;
  vec<int> _first_neighbour;
  vec<vec<int>> _next_neighbour;
  void checkSymmetry();
  void calculateNeighboursLists();
  // Assumes _neighbours has been calculated;
  void calculateFirstNextNeighbours();
};

ostream &operator<<(ostream &os, Graph const &G);
template <typename T>
ostream &operator<<(ostream &os, const set<T> &G) {
  os << "{";
  for (auto it = G.begin(); it != G.end(); it++) {
    os << *it << (next(it) == G.end() ? "" : ", ");
  }
  os << "}";

  return os;
}

// Finds shortest path from start to end in G, where every vertex inside satisfies predicate.
// Returns empty vector if none exist
vec<int> findShortestPathWithPredicate(const Graph &G, int start, int end, function<bool(int)> test);
// Finds all shortest paths, where every vertex x inside path a -- b satisfies predicate f(a,b,x)
// returnVal[a][b] is empty if none exist.
vec<vec<vec<int>>> allShortestPathsWithPredicate(const Graph &G, function<bool(int)> test);

// Returns triangles: [b1, b2, b3], such that b1<b2<b3
vec<vec<int>> getTriangles(const Graph &G);

// EmptyStarTriangle is a four (a, s1, s2, s3), where each si is connected to
// a and none si and sj are connected to each other.
// Returns every permutation.
vec<pair<int, vec<int>>> getEmptyStarTriangles(const Graph &G);

// Return whether v is X-complete in G
// v is X-complete, if v is not in X and v is adjacent to every vertex of X.
bool isComplete(const Graph &G, const vec<int> &X, int v);

// Returns a vector of all X-complete vertices in G.
vec<int> getCompleteVertices(const Graph &G, const vec<int> &X);

// Runs dfs on a Graph G, with visited as an input-output of visited vertices. In addition action(v) will be
// performed on each visited vertex. Each visited vertex (except start) must satisfy test predicate.
void dfsWith(const Graph &G, vec<int> &visited, int start, function<void(int)> action,
             function<bool(int)> test = [](int v) -> bool { return true; });

// Returns a vector of all components of G.
vec<vec<int>> getComponents(const Graph &G);

// Returns components of G induced on X. This is not the same of getComponents(G.induced(X)), as here every
// component must all be contained in X.
vec<vec<int>> getComponentsOfInducedGraph(const Graph &G, const vec<int> &X);

vec<vec<int>> generateTuples(int size, int max);
bool isAllZeros(const vec<int> &v);
bool isDistinctValues(const vec<int> &v);
int countNonZeros(const vec<int> &v);

vec<int> nextTuple(vec<int> v, int max);
void nextTupleInPlace(vec<int> &v, int max);

// Returns whether v is a path in G. A path must have all vertices distinct, and edges only between neighbors.
// If isCycleOk a cycle is also considered a path e.g. [1,2,3,4] if 1-4 is an edge
bool isAPath(const Graph &G, const vec<int> &v, bool isCycleOk = false, bool areChordsOk = false);

// Returns (in &v) next path of length len in G (in some order). If v is empty returns first path. If v is
// the last path returns empty vec. Iterating from empty vector to empty vector gives all paths of length len.
void nextPathInPlace(const Graph &G, vec<int> &v, int len, bool isCycleOk = false, bool areChordsOk = false);
