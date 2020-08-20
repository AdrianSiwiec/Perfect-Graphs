#include <algorithm>
#include <istream>
#include "commons.h"
#include "testCommons.h"

#include "src/devCode.dev"

bool isEmpty(std::istream &pFile) { return pFile.peek() == std::istream::traits_type::eof(); }

int main() {
  std::ios_base::sync_with_stdio(0);

  int n;
  vector<Graph> Gs;
  while (std::cin >> n) {
    string m;
    for (int i = 0; i < n; i++) {
      string tmp;
      std::cin >> tmp;
      m += tmp;
    }

    Graph G(n, m);
    Gs.push_back(G);
  }

  // To make time predictions better...
  random_shuffle(Gs.begin(), Gs.end());

  {
    cout << "N=" << n << endl;

    RaiiProgressBar bar(Gs.size());

    for (int i = 0; i < Gs.size(); i++) {
      testGraph(Gs[i], {algoCudaPerfect}, {cuIsPerfect});
      bar.update(i + 1);
    }
  }
  cout << endl;

  StatsFactory::printStats2();
}
