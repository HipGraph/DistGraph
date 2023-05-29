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


void prefix_sum(vector<int> &values, vector<int> &offsets) {
  int sum = 0;
  for(int i = 0; i < values.size(); i++) {
    offsets.push_back(sum);
    sum += values[i];
  }
}
