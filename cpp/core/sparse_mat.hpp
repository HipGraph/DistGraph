#pragma once
#include "common.h"
#include "csr_linked_list.hpp"
#include "csr_local.hpp"
#include "distributed_mat.hpp"
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <parallel/algorithm>
#include <vector>

using namespace std;

namespace distblas::core {

/**
 * This class represents the Sparse Matrix
 */
template <typename T> class SpMat : public DistributedMat {

public:
  int gRows, gCols, gNNz;
  vector<Tuple<T>> coords;

  vector<uint64_t> block_col_starts;
  vector<CSRLocal<T> *> csr_blocks;
  vector<uint64_t> block_row_starts;

  vector<CSRLinkedList<T>> csr_linked_lists;

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

  SpMat() {}

  void divide_block_cols(int block_width, int target_divisions, bool mod_ind) {
    block_col_starts.clear();
    // Locate block starts within the local sparse matrix (i.e. divide a long
    // block row into subtiles)
    int current_start = 0;

    for (uint64_t i = 0; i < coords.size(); i++) {
      while (coords[i].col >= current_start) {
        block_col_starts.push_back(i);
        current_start += block_width;
      }

      // This modding step helps indexing.
      if (mod_ind) {
        coords[i].col %= block_width;
      }
    }

    assert(block_col_starts.size() <= target_divisions + 1);

    while (block_col_starts.size() < target_divisions + 1) {
      block_col_starts.push_back(coords.size());
    }
  }

  void sort_by_rows() {
    for (int i = 0; i < block_col_starts.size() - 1; i++) {
      __gnu_parallel::sort(coords.begin() + block_col_starts[i],
                           coords.begin() + block_col_starts[i + 1],
                           row_major<T>);
    }
  }

  void divide_block_rows(int block_width_row, int block_width_col,
                         int target_divisions, bool mod_ind) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    block_row_starts.clear();
    for (uint64_t i = 0; i < block_col_starts.size() - 1; i++) {

      int current_start = block_width_col * rank;
      for (uint64_t j = block_col_starts[i]; j < block_col_starts[i + 1]; j++) {
        while (coords[j].row >= current_start) {
          block_row_starts.push_back(j);
          int current_step =
              std::min(static_cast<int>(block_width_row),
                       static_cast<int>(coords[block_col_starts[i + 1]].row));
          current_start += block_width_row;
        }

        // This modding step helps indexing.
        if (mod_ind) {
          coords[j].row %= block_width_row;
        }
      }
    }

    block_row_starts.push_back(coords.size());
  }


  void initialize_CSR_blocks(int block_rows, int block_cols, int max_nnz,
                             bool transpose) {

    int current_col_block = 0;
    csr_linked_lists = vector<CSRLinkedList<T>>(block_row_starts.size() - 1);
    int current_vector_pos = 0;
    for (int j = 0; j < block_row_starts.size() - 1; j++) {

      if (block_row_starts[j] >= block_col_starts[current_col_block + 1]) {
        ++current_col_block;
        current_vector_pos = 0;
      }

      if (current_col_block == 0) {
        CSRLinkedList<T> CSRlist;
        csr_linked_lists[j] = CSRlist;
      }

      int num_coords = block_row_starts[j + 1] - block_row_starts[j];

      if (num_coords > 0) {
        CSRLocal<T> *block = new CSRLocal<T>(
            block_rows, block_cols, num_coords,
            coords.data() + block_row_starts[j], num_coords, transpose);
        csr_linked_lists[current_vector_pos].insert(block);
        ++current_vector_pos;
      }
    }
  }

  void print_blocks_and_cols() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int current_col_block = 0;
    for (int j = 0; j < csr_linked_lists.size() ; j++) {

      CSRLinkedList<T> linkedList = csr_linked_lists[j];
      CSRLocalNode<T> *head = linkedList.getHeadNode();
      int count=0;
      while (head != nullptr) {
        string output_path = "blocks_rank" + to_string(rank) + "_col_" +
                             to_string(count) + "_row_" +
                             to_string(j) + ".txt";
        char stats[500];
        strcpy(stats, output_path.c_str());
        ofstream fout(stats, std::ios_base::app);


        CSRLocal<T>* csr_data= head->data;


        distblas::core::CSRHandle* handle = csr_data->handler;
        int numRows = handle->rowStart.size()-1;

        for (int i = 0; i < numRows; i++) {
          int start = handle->rowStart[i];
          int end = handle->rowStart[i + 1];

          fout << "Row " << i << ": ";
          for (int j = start; j < end; j++) {
            int col = handle->col_idx[j];
            int value = handle->values[j];

            fout << "(" << col << ", " << value << ") ";
          }
          fout <<endl;
        }
        head = head->next;
        ++count;
      }
    }
  }

  ~SpMat() {
//    for (int i = 0; i < csr_linked_lists.size(); i++) {
//      if (csr_linked_lists[i] != nullptr) {
//        delete csr_linked_lists[i];
//      }
//    }
  }
};

} // namespace distblas::core
