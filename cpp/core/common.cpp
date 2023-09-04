#include "common.h"


using namespace std;
using namespace std::chrono;
MPI_Datatype distblas::core::SPTUPLE;
MPI_Datatype distblas::core::DENSETUPLE;

vector<string> distblas::core::perf_counter_keys = {"Computation Time", "Communication Time","Memory usage", "Data transfers"};

 map<string, int> distblas::core::call_count;
 map<string, double> distblas::core::total_time;

int distblas::core::divide_and_round_up(uint64_t num, int denom){
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

size_t distblas::core::get_memory_usage(){
  std::ifstream statm("/proc/self/statm");
  if (statm.is_open()) {
    unsigned long size, resident, shared, text, lib, data, dt;
    statm >> size >> resident >> shared >> text >> lib >> data >> dt;

    // Memory values are in pages, typically 4 KB each on Linux
    size_t pageSize = sysconf(_SC_PAGESIZE);
    size_t virtualMemUsed = size * pageSize;
    size_t residentSetSize = resident * pageSize;

    size_t mem_usage = virtualMemUsed/(1024*1024);
    return mem_usage;
  }
  return 0;
}

void distblas::core::reset_performance_timers() {
  for (auto it = perf_counter_keys.begin(); it != perf_counter_keys.end();
       it++) {
    call_count[*it] = 0;
    total_time[*it] = 0.0;
  }
}

void distblas::core::stop_clock_and_add(my_timer_t &start, string counter_name) {
  if (find(perf_counter_keys.begin(), perf_counter_keys.end(),
           counter_name) != perf_counter_keys.end()) {
    call_count[counter_name]++;
    total_time[counter_name] += stop_clock_get_elapsed(start);
  } else {
    cout << "Error, performance counter " << counter_name
         << " not registered." << endl;
    exit(1);
  }
}

void distblas::core::add_memory(size_t mem, string counter_name) {
  if (find(perf_counter_keys.begin(), perf_counter_keys.end(),
           counter_name) != perf_counter_keys.end()) {
    call_count[counter_name]++;
    total_time[counter_name] += mem;
  } else {
    cout << "Error, performance counter " << counter_name
         << " not registered." << endl;
    exit(1);
  }
}

void distblas::core::add_datatransfers(uint64_t count, string counter_name) {
  if (find(perf_counter_keys.begin(), perf_counter_keys.end(),
           counter_name) != perf_counter_keys.end()) {
    call_count[counter_name]++;
    total_time[counter_name] += count;
  } else {
    cout << "Error, performance counter " << counter_name
         << " not registered." << endl;
    exit(1);
  }
}

void distblas::core::print_performance_statistics() {
  // This is going to assume that all timing starts and ends with a barrier,
  // so that all processors enter and leave the call at the same time. Also,
  // I'm taking an average over several calls by all processors; might want to
  // compute the variance as well.

  int rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (rank == 0) {
    cout << endl;
    cout << "================================" << endl;
    cout << "==== Performance Statistics ====" << endl;
    cout << "================================" << endl;
    //      print_algorithm_info();
  }

  cout << json_perf_statistics().dump(4);

  if (rank == 0) {
    cout << "=================================" << endl;
  }
}

json distblas::core::json_perf_statistics() {
  json j_obj;
  int rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  for (auto it = perf_counter_keys.begin(); it != perf_counter_keys.end();
       it++) {
    double val = total_time[*it];

    MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // We also have the call count for each statistic timed
    val /= world_size;

    if (rank == 0) {
      j_obj[*it] = val;
    }
  }
  return j_obj;
}

my_timer_t distblas::core::start_clock() { return std::chrono::steady_clock::now(); }

double distblas::core::stop_clock_get_elapsed(my_timer_t &start) {
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff = end - start;
  return diff.count();
}


