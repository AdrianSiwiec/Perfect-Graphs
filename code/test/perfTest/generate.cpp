#include <ctime>
#include <map>
#include <random>
#include "commons.h"
#include "oddHoles.h"
#include "perfect.h"
#include "testCommons.h"

int main() {
  srand(time(0));

  for(int i=15; i<=19; i++) {
    for(int k=0; k<5; k++) {
      Graph G = getRandomPerfectGraph(i, getDistr());
      cout<<G<<endl;
    }
  }
}