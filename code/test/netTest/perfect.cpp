#include "commons.h"
#include "testCommons.h"
#include <algorithm>
#include <istream>

bool isEmpty(istream &pFile) { return pFile.peek() == istream::traits_type::eof(); }

int main() {
  ios_base::sync_with_stdio(0);

  int n;
  vector<Graph> Gs;
  while (cin >> n) {
    string m;
    for (int i = 0; i < n; i++) {
      string tmp;
      cin >> tmp;
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
      testGraph(Gs[i], true, false);
      bar.update(i + 1);
    }
  }
  cout << endl;

  printStats();
}