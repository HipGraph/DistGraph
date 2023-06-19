#pragma once
#include <iostream>
#include <mpi.h>
#include <vector>
#include "common.h"
#include "distributed_mat.hpp"
#include "csr_local.hpp"


using namespace std;

namespace distblas::core {

/**
 * This class represents the Sparse Matrix
 */
template <typename T>
class SpMat: public DistributedMat {

public:
  int gRows, gCols, gNNz;
  vector<Tuple<T>> coords;


  vector<uint64_t> block_starts;
  vector<CSRLocal<T>*> csr_blocks;

  /**
   * Constructor for Sparse Matrix representation of  Adj matrix
   * @param coords  (src, dst, value) Tuple vector loaded as input
   * @param gRows   total number of Rows in Distributed global Adj matrix
   * @param gCols   total number of Cols in Distributed global Adj matrix
   * @param gNNz     total number of NNz in Distributed global Adj matrix
   */
  SpMat(vector<Tuple<T>> &coords, int gRows, int gCols, int gNNz) {
    this->gRows = gRows;
    this->gCols = gCols;
    this->gNNz = gNNz;
    this->coords = coords;
  }

  SpMat() {

  }


  void divide_block_cols(int block_width, int target_divisions, bool mod_ind) {
    block_starts.clear();
    // Locate block starts within the local sparse matrix (i.e. divide a long
    // block row into subtiles)
    int current_start = 0;
    for(uint64_t i = 0; i < coords.size(); i++) {
      while(coords[i].col >= current_start) {
        block_starts.push_back(i);
        current_start += block_width;
      }

      // This modding step helps indexing.
      if(mod_ind) {
        coords[i].col %= block_width;
      }
    }

    assert(block_starts.size() <= target_divisions + 1);

    while(block_starts.size() < target_divisions + 1) {
      block_starts.push_back(coords.size());
    }

  }


  void initialize_CSR_blocks(int block_rows, int block_cols, int max_nnz, bool transpose) {
    if(max_nnz == -1) {
      cout<<" size "<<block_starts.size() <<endl;

      for(int i = 0; i < block_starts.size() - 1; i++) {
        int num_coords = block_starts[i + 1] - block_starts[i];

        if(num_coords > 0) {
          cout<<" i "<<i<<"  coords "<<num_coords<<endl;

          CSRLocal<T>* block
              = new CSRLocal<T>(block_rows, block_cols, num_coords,
                                coords.data() + block_starts[i],
                                num_coords, transpose);

          cout<<" i "<<i<<"  coords "<<num_coords<<" csr creation completed "<<endl;
          csr_blocks.push_back(block);
        }
        else {
          csr_blocks.push_back(nullptr);
        }
      }
    }
    else {
      int num_coords = block_starts[1] - block_starts[0];
      CSRLocal<T>* block = new CSRLocal<T>(block_rows, block_cols,
                                           max_nnz, coords.data(),
                                           num_coords, transpose);
      csr_blocks.push_back(block);
    }
  }


  ~SpMat() {
    for(int i = 0; i < csr_blocks.size(); i++) {
      if(csr_blocks[i] != nullptr) {
        delete csr_blocks[i];
      }
    }
  }

};

}
