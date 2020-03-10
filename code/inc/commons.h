#include <vector>

template <typename T> struct ranged_vec : public std::vector<T> {
  using std::vector<T>::vector;

  // Range checking
  T &operator[](int i) { return std::vector<T>::at(i); }
  const T &operator[](int i) const { return std::vector<T>::at(i); }
};

struct Graph {
  const int n;
  Graph(int n) : n(n), _tab(n) {
    for (int i = 0; i < n; i++)
      _tab[i].resize(n);
  }
  ranged_vec<int> &operator[](int index) { return _tab[index]; }

private:
  ranged_vec<ranged_vec<int>> _tab;
};