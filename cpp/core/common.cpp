#include "common.h"

using namespace std;

int divide_and_round_up(int num, int denom){
  if (num % denom > 0) {
    return num / denom + 1;
  }
  else {
    return num / denom;
  }
}


