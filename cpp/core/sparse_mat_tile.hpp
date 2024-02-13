#pragma once
#include <algorithm>
#include <memory>

namespace distblas::core {

template <typename INDEX_TYPE, typename VALUE_TYPE> class SparseTile {

private:
public:
  int id;
  INDEX_TYPE row_starting_index;
  INDEX_TYPE row_end_index;
  INDEX_TYPE col_start_index;
  INDEX_TYPE col_end_index;
  std::unique_ptr<std::vector<INDEX_TYPE>> col_indices;
  int mode = 0; // mode=0 local and mode=1 remote

  SparseTile(int id, INDEX_TYPE row_starting_index, INDEX_TYPE row_end_index,
             INDEX_TYPE col_start_index, INDEX_TYPE col_end_index)
      : id(id), row_starting_index(row_starting_index),
        row_end_index(row_end_index), col_start_index(col_start_index),
        col_end_index(col_end_index) {

    col_indices = make_unique<std::vector<INDEX_TYPE>>();
  }
};

} // namespace distblas::core
