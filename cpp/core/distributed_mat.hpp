#pragma once

namespace distblas::core {

class DistributedMat {

public:
  shared_ptr<vector<vector<Tuple<VALUE_TYPE>>>> sparse_data_collector;

  shared_ptr<vector<INDEX_TYPE>> sparse_data_counter;

  shared_ptr<vector<vector<VALUE_TYPE>>> dense_collector;

};

}
