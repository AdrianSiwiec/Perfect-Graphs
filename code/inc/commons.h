#pragma once

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

struct Graph {
  int n;

  Graph(int n) : n(n), _tab(n) {
    for (int i = 0; i < n; i++)
      _tab[i].resize(n);
  }
  vec<int> &operator[](int index) { return _tab[index]; }
  const vec<int> &operator[](int index) const { return _tab[index]; }

private:
  vec<vec<int>> _tab;
};

vec<vec<int>> getTriangles(const Graph &G);
// EmptyStarTriangle is a four (a, s1, s2, s3), where each si is connected to a
// and none si and sj are connected to each other
vec<pair<int, vec<int>>> getEmptyStarTriangles(const Graph &G);