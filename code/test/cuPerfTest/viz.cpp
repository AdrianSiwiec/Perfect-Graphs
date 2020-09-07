#include <algorithm>
#include <istream>
#include "commons.h"
#include "testCommons.h"

bool isEmpty(std::istream &pFile) { return pFile.peek() == std::istream::traits_type::eof(); }

int main() {
  std::ios_base::sync_with_stdio(0);

  int n;
  string m;
  std::cin >> n;
  for (int i = 0; i < n; i++) {
    string tmp;
    std::cin >> tmp;
    m += tmp;
  }

  Graph G(n, m);

  cout << n << endl;
  for (int i = 0; i < n; i++) {
    cout << i << endl;
  }
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (G.areNeighbours(i, j)) cout << i << " " << j << endl;
    }
  }

  cout << endl;

  Graph CG = G.getComplement();
  cout << n << endl;
  for (int i = 0; i < n; i++) {
    cout << i << endl;
  }
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (CG.areNeighbours(i, j)) cout << i << " " << j << endl;
    }
  }
}
