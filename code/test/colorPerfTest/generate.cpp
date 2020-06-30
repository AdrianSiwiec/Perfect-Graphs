#include <ctime>
#include <map>
#include <random>
#include "commons.h"
#include "oddHoles.h"
#include "perfect.h"
#include "testCommons.h"

int main() {
  srand(time(0));

  while (true) {
    Graph G = getRandomGraph(6, getDistr());
    int counter = 0;
    int howManyToGenerate = 1;
    if (isPerfectGraph(G)) {
      cout << G << endl;
      counter++;
      if (counter == howManyToGenerate) return 0;
    }
  }
}