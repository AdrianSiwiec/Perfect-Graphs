#pragma once

#include <algorithm>
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
}

struct Graph {
  int n;

  Graph(int n);
  Graph(int n, string s);
  vec<int> &operator[](int index) { return _tab[index]; }
  const vec<int> &operator[](int index) const { return _tab[index]; }

private:
  vec<vec<int>> _tab;
};

// Returns triangles: [b1, b2, b3], such that b1<b2<b3
vec<vec<int>> getTriangles(const Graph &G);

// EmptyStarTriangle is a four (a, s1, s2, s3), where each si is connected to
// a and none si and sj are connected to each other.
// Returns every permutation.
vec<pair<int, vec<int>>> getEmptyStarTriangles(const Graph &G);