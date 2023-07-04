#pragma once
#include "common.h"
#include "csr_linked_list.hpp"
#include "csr_local.hpp"
#include "distributed_mat.hpp"
#include <algorithm>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <parallel/algorithm>
#include <unordered_set>
#include <vector>

using namespace std;

namespace distblas::core {

/**
 * This class represents the Sparse Matrix
 */
template <typename T> class SpMat : public DistributedMat {

private:
  vector<uint64_t> block_col_starts;
  vector<uint64_t> block_row_starts;

  vector<shared_ptr<CSRLinkedList<T>>> csr_linked_lists;

public:
  int gRows, gCols, gNNz;
  vector<Tuple<T>> coords;
  int block_row_width, block_col_width;
  int proc_col_width, proc_row_width;

  /**
   * Constructor for Sparse Matrix representation of  Adj matrix
   * @param coords  (src, dst, value) Tuple vector loaded as input
   * @param gRows   total number of Rows in Distributed global Adj matrix
   * @param gCols   total number of Cols in Distributed global Adj matrix
   * @param gNNz     total number of NNz in Distributed global Adj matrix
   */
  SpMat(vector<Tuple<T>> &coords, int &gRows, int &gCols, int &gNNz,
        int &block_row_width, int &block_col_width, int &proc_row_width,
        int &proc_col_width) {
    this->gRows = gRows;
    this->gCols = gCols;
    this->gNNz = gNNz;
    this->coords = coords;
    this->block_row_width = block_row_width;
    this->block_col_width = block_col_width;
    this->proc_col_width = proc_col_width;
    this->proc_row_width = proc_row_width;
  }

  SpMat() {}

  void divide_block_cols(int batch_size, int col_block_with,
                         int target_divisions, bool mod_ind, bool trans) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    block_col_starts.clear();

    // Locate block starts within the local sparse matrix (i.e. divide a long
    // block row into subtiles)
    int current_start = 0;
    if (trans) {
      current_start = col_block_with * rank;
    }

    cout << "rank " << rank << " trans" << trans << " current_start "
         << current_start << endl;

    // TODO: introduce atomic capture
    for (uint64_t i = 0; i < coords.size(); i++) {
      while (coords[i].col >= current_start) {
        block_col_starts.push_back(i);
        cout << "rank " << rank << " trans" << trans << " col adding i " << i
             << endl;
        current_start += batch_size;
      }

      // This modding step helps indexing.
      if (mod_ind) {
        coords[i].col %= batch_size;
      }
    }

    block_col_starts.push_back(coords.size());
  }

  void sort_by_rows() {
    for (int i = 0; i < block_col_starts.size() - 1; i++) {
      __gnu_parallel::sort(coords.begin() + block_col_starts[i],
                           coords.begin() + block_col_starts[i + 1],
                           row_major<T>);
    }
  }

  void divide_block_rows(int block_width_row, int block_width_col, bool mod_ind,
                         bool trans) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    block_row_starts.clear();

    for (uint64_t i = 0; i < block_col_starts.size() - 1; i++) {

      int current_start = block_width_col * rank;

      if (trans) {
        current_start = 0;
      }

      // TODO: introduce atomic capture
      int matched_count = 0;
      for (uint64_t j = block_col_starts[i]; j < block_col_starts[i + 1]; j++) {
        while (coords[j].row >= current_start) {
          block_row_starts.push_back(j);
          cout << "rank " << rank << " trans" << trans << "  current start "
               << current_start << " row adding j " << j << endl;
          current_start += block_width_row;
          cout << "rank " << rank << " trans" << trans
               << " updated current start " << current_start << " row adding j "
               << j << endl;
          ++matched_count;
        }

        // This modding step helps indexing.
        if (mod_ind) {
          coords[j].row %= block_width_row;
        }
      }
      int expected_matched_count =
          std::max(1, (block_width_col / block_width_row));
      if (matched_count < expected_matched_count) {
        cout << "rank " << rank << " trans" << trans << " current start "
             << current_start << " not matching adding row adding j "
             << block_col_starts[i + 1] << endl;
        block_row_starts.push_back(block_col_starts[i + 1]);
      }
    }
    block_row_starts.push_back(coords.size());
  }

  void initialize_CSR_blocks(int block_rows, int block_cols,
                             int local_max_row_width, int local_max_col_width,
                             int max_nnz, bool transpose) {

    int col_block = 0;

    int no_of_nodes = (transpose) ? (gRows / block_rows) : (gCols / block_cols);

    int no_of_lists = (transpose) ? (local_max_col_width / block_cols)
                                  : (local_max_row_width / block_rows);
    csr_linked_lists =
        std::vector<std::shared_ptr<CSRLinkedList<T>>>(no_of_lists);

#pragma omp parallel
    for (int i = 0; i < no_of_lists; i++) {
      csr_linked_lists[i] = std::make_shared<CSRLinkedList<T>>();
    }

    for (int j = 0; j < block_row_starts.size() - 1; j++) {
      int current_vector_pos = 0;
      if (!transpose) {
        current_vector_pos = j % no_of_lists;
        if (j > 0 and current_vector_pos == 0) {
          ++col_block;
        }
      } else {
        current_vector_pos = j / no_of_nodes;
        col_block = current_vector_pos;
      }

      int num_coords = block_row_starts[j + 1] - block_row_starts[j];
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      cout << " rank " << rank << "_" << transpose
           << " csr_block_initating_index" << block_row_starts[j]
           << " current vec pos" << current_vector_pos << " col_block"
           << col_block << endl;

      Tuple<T> *coords_ptr = (coords.data() + block_row_starts[j]);
      (csr_linked_lists[current_vector_pos].get())
          ->insert(block_rows, block_cols, num_coords, coords_ptr, num_coords,
                   transpose, j);
    }
  }

  void fill_col_ids(int block_row_id, int block_col_id,
                    vector<uint64_t> &col_ids, bool transpose,
                    bool return_global_ids) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int csr_linked_list_id = (transpose) ? block_col_id : block_row_id;
    int batch_id = (transpose) ? block_row_id : block_col_id;

    auto linkedList = csr_linked_lists[csr_linked_list_id];

    auto head = (linkedList.get())->getHeadNode();
    int count = 0;
    while (count < batch_id && (head.get())->next != nullptr) {
      head = (head.get())->next;
      ++count;
    }
    if (count == batch_id) {
      auto csr_data = (head.get())->data;
      int block_row_width = this->block_row_width;
      int block_col_width = this->block_col_width;
      distblas::core::CSRHandle *handle = (csr_data.get())->handler.get();
      col_ids = vector<uint64_t>((handle->col_idx).size());
      std::transform(
          std::begin((handle->col_idx)), std::end((handle->col_idx)),
          std::begin(col_ids),
          [&return_global_ids, &rank, &transpose, &batch_id, &block_col_id,
           &block_row_width, &block_col_width](MKL_INT value) {
            if (!return_global_ids) {
              return static_cast<uint64_t>(value);
            } else {
              uint64_t base_id = 0;
              if (transpose) {
                base_id = static_cast<uint64_t>(batch_id * block_row_width);
              } else {
                base_id = static_cast<uint64_t>(batch_id * block_col_width);
              }
              uint64_t g_index = static_cast<uint64_t>(value) + base_id;
              return g_index;
            }
          });
    }
  }

  void print_blocks_and_cols(bool trans) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cout << " rank " << rank << "_" << trans
         << " printing print_blocks_and_cols" << endl;
    int current_col_block = 0;
    for (int j = 0; j < csr_linked_lists.size(); j++) {
      cout << " rank " << rank << " j " << j << endl;
      auto linkedList = csr_linked_lists[j];

      auto head = (linkedList.get())->getHeadNode();

      int count = 0;
      while (head != nullptr) {
        cout << " rank " << rank << " j " << j << " count " << count
             << " node id " << (head.get())->id << endl;
        string output_path = "blocks_rank" + to_string(rank) + "_trans" +
                             to_string(trans) + "_col_" +
                             to_string((trans) ? j : count) + "_row_" +
                             to_string((trans) ? count : j) + ".txt";
        char stats[500];
        strcpy(stats, output_path.c_str());
        ofstream fout(stats, std::ios_base::app);

        auto csr_data = (head.get())->data;

        int num_coords = (csr_data.get())->num_coords;

        distblas::core::CSRHandle *handle = (csr_data.get())->handler.get();
        int numRows = handle->rowStart.size() - 1;

        for (int i = 0; i < numRows; i++) {
          int start = handle->rowStart[i];
          int end = handle->rowStart[i + 1];

          fout << "Row " << i << ": ";
          if (num_coords > 0) {
            for (int k = start; k < end; k++) {

              int col = handle->col_idx[k];
              int value = handle->values[k];

              fout << "(" << col << ", " << value << ") ";
            }
          }
          fout << endl;
        }
        head = (head.get())->next;
        ++count;
      }
    }
  }

  void print_coords(bool trans) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    string output_path =
        "coords" + to_string(rank) + "trans" + to_string(trans) + ".txt";
    char stats[500];
    strcpy(stats, output_path.c_str());
    ofstream fout(stats, std::ios_base::app);

    for (int i = 0; i < coords.size(); i++) {
      fout << coords[i].row << " " << coords[i].col << " " << endl;
    }
  }

  ~SpMat() {}
};

} // namespace distblas::core
