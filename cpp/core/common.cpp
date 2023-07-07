#include "common.h"


using namespace std;

MPI_Datatype distblas::core::SPTUPLE;
MPI_Datatype distblas::core::DENSETUPLE;

int distblas::core::divide_and_round_up(int num, int denom){
  if (num % denom > 0) {
    return num / denom + 1;
  }
  else {
    return num / denom;
  }
}

void distblas::core::prefix_sum(vector<int> &values, vector<int> &offsets) {
  int sum = 0;
  for(int i = 0; i < values.size(); i++) {
    offsets.push_back(sum);
    sum += values[i];
  }
}

vector<uint64_t> distblas::core::generate_random_numbers(int lower_bound, int upper_bound, int seed,
                                                    int ns) {
  vector<uint64_t> vec(ns);
  std::minstd_rand generator(seed);

  // Define the range of the uniform distribution
  std::uniform_int_distribution<int> distribution(lower_bound, upper_bound);

  // Generate and print random numbers
//#pragma omp parallel
  for (int i = 0; i < ns; ++i) {
    int random_number = distribution(generator);
    vec[i]=static_cast<uint64_t>(random_number);
  }
  return vec;
}
