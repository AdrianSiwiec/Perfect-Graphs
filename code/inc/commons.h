#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <tuple>
#include <vector>
using namespace std;

#define mp make_pair
#define st first
#define nd second

template <typename T> struct vec : public vector<T> {
  using vector<T>::vector;

  // Range checking
  T &operator[](int i) { return vector<T>::at(i); }
  const T &operator[](int i) const { return vector<T>::at(i); }
};
template <typename T> ostream &operator<<(ostream &os, vec<T> const &v) {
  os << "[";
  for (int i = 0; i < v.size(); i++) {
    os << v[i] << (i + 1 < v.size() ? ", " : "");
  }
  os << "]";

  return os;
}

struct Graph {
  int n;

  Graph(int n);
  Graph(int n, string s);
  Graph(vec<vec<int>> neighbours);
  vec<int> &operator[](int index) { return _neighbours[index]; }
  const vec<int> &operator[](int index) const { return _neighbours[index]; }
  bool areNeighbours(int a, int b) const { return _matrix[a][b]; }

  Graph getComplement() const;
  // Returns G' - Graph induced on X. Set of vertices is left the same, and edge is in G' if it is in G and
  // both it's ends are in X.
  Graph getInduced(vec<int> X) const;

private:
  vec<vec<int>> _neighbours;
  vec<vec<int>> _matrix;
  void checkSymmetry();
};

ostream &operator<<(ostream &os, Graph const &G);

// Finds shortest path from start to end in G, where every vertex inside satisfies predicate.
// Returns empty vector if none exist
vec<int> findShortestPathWithPredicate(const Graph &G, int start, int end, function<bool(int)> test);

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
// performed on each visited vertex.
void dfsWith(const Graph &G, vec<int> &visited, int start, function<void(int)> action);

// Returns a vector of all components of G.
vec<vec<int>> getComponents(const Graph &G);

vec<vec<int>> generateTuples(int size, int max);
bool isAllZeros(const vec<int> &v);
bool isDistinctValues(const vec<int> &v);
vec<int> nextTuple(vec<int> v, int max);
void nextTupleInPlace(vec<int> &v, int max);