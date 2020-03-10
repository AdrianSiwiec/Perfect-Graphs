#pragma once

#include <vector>
using namespace std;

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