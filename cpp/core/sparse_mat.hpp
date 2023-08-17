#pragma once
#include "common.h"
#include "csr_linked_list.hpp"
#include "csr_local.hpp"
#include "distributed_mat.hpp"
#include <algorithm>
#include <chrono>
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
  uint64_t gRows, gCols, gNNz;
  vector<Tuple<T>> coords;
  int batch_size;
  int proc_col_width, proc_row_width;
  bool col_merged = false;
  bool transpose = false;

  /**
   * Constructor for Sparse Matrix representation of  Adj matrix
   * @param coords  (src, dst, value) Tuple vector loaded as input
   * @param gRows   total number of Rows in Distributed global Adj matrix
   * @param gCols   total number of Cols in Distributed global Adj matrix
   * @param gNNz     total number of NNz in Distributed global Adj matrix
   */
  SpMat(vector<Tuple<T>> &coords, uint64_t &gRows, uint64_t &gCols,
        uint64_t &gNNz, int &batch_size, int &proc_row_width,
        int &proc_col_width, bool col_merged, bool transpose) {
    this->gRows = gRows;
    this->gCols = gCols;
    this->gNNz = gNNz;
    this->coords = coords;
    this->batch_size = batch_size;
    this->proc_col_width = proc_col_width;
    this->proc_row_width = proc_row_width;
    this->col_merged = col_merged;
    this->transpose = transpose;
  }

  SpMat() {}

  void divide_block_cols() {
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    block_col_starts.clear();

    int start_index = 0;
    int end_index = 0;
    uint64_t checking_index = rank * proc_col_width;
    //      uint64_t checking_index = 1;
    uint64_t checking_end_index = (rank + 1) * proc_col_width - 1;

    if (rank == world_size - 1) {
      checking_end_index = std::max(
          static_cast<uint64_t>((rank + 1) * proc_col_width) - 1, gCols - 1);
    }

    auto startIt = std::find_if(coords.begin(), coords.end(),
                                [&checking_index](const auto &tuple) {
                                  return tuple.col >= checking_index;
                                });

    auto endIt = std::find_if(coords.rbegin(), coords.rend(),
                              [&checking_end_index](const auto &tuple) {
                                return tuple.col <= checking_end_index;
                              });
    std::rotate(coords.begin(), startIt, std::next(endIt).base());
    uint64_t startIndex = std::distance(coords.begin(), startIt);
    uint64_t endIndex = std::distance(coords.begin(), std::next(endIt).base());
    uint64_t first_batch_len = (endIndex + 1) - startIndex;
    //    uint64_t second_batch_len = coords.size() - first_batch_len;

    //    int considered_col_width = proc_col_width;
    //    if (rank == world_size - 1) {
    //      considered_col_width = gCols - proc_col_width * (world_size - 1);
    //    }

    //    if (mod_ind) {
    //      std::transform(coords.begin(), coords.begin() + first_batch_len,
    //                     coords.begin(),
    //                     [&considered_col_width](const auto &tuple) {
    //                       const auto &[row, col, value] = tuple;
    //                       int64_t modifiedCol = col % (considered_col_width);
    //                       return Tuple<T>{row, modifiedCol, value};
    //                     });
    //      std::transform(coords.begin() + first_batch_len, coords.end(),
    //                     coords.begin() + first_batch_len, [&](const auto
    //                     &tuple) {
    //                       const auto &[row, col, value] = tuple;
    //                       int64_t modifiedCol = col;
    //                       return Tuple<T>{row, modifiedCol, value};
    //                     });
    //    }

    block_col_starts.push_back(0);
    block_col_starts.push_back(first_batch_len);
    block_col_starts.push_back(coords.size());
  }

  void sort_by_rows() {
    for (int i = 0; i < block_col_starts.size() - 1; i++) {
      __gnu_parallel::sort(coords.begin() + block_col_starts[i],
                           coords.begin() + block_col_starts[i + 1],
                           row_major<T>);
    }
  }

  void divide_block_rows(int batch_size, bool mod_ind, bool trans) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    block_row_starts.clear();

    int expected_batch_count =
        (trans) ? ((proc_row_width % batch_size == 0)
                       ? (gRows / batch_size)
                       : (((proc_row_width / batch_size) + 1) * world_size))
                : ((proc_row_width % batch_size == 0)
                       ? (proc_row_width / batch_size)
                       : (proc_row_width / batch_size) + 1);

    bool divided_equallaly = true;
    int last_proc_batch_size = batch_size;
    int batch_count = proc_row_width / batch_size;

    int current_batch_size = batch_size;

    if (proc_row_width % batch_size != 0) {
      divided_equallaly = false;
      last_proc_batch_size = proc_row_width - batch_size * batch_count;
      batch_count = batch_count + 1;
    }

    for (uint64_t i = 0; i < block_col_starts.size() - 1; i++) {

      int current_start = proc_row_width * rank;
      int next_start = current_start + batch_size;
      if (trans) {
        current_start = 0;
        next_start = batch_size;
      }

      // TODO: introduce atomic capture
      int matched_count = 0;
      for (uint64_t j = block_col_starts[i]; j < block_col_starts[i + 1]; j++) {
        while (coords[j].row >= current_start) {
          block_row_starts.push_back(j);
          ++matched_count;
          //          if (coords[j].row > next_start) {
          //            while (coords[j].row > next_start) {
          //              if (!divided_equallaly) {
          //                if (j > block_col_starts[i] and
          //                    (matched_count) % (batch_count) == 0) {
          //                  current_start += last_proc_batch_size;
          //                  next_start += batch_size;
          //                } else if (j > block_col_starts[i] and
          //                           (matched_count + 1) % (batch_count) == 0)
          //                           {
          //                  current_start += batch_size;
          //                  next_start += last_proc_batch_size;
          //                } else {
          //                  current_start += batch_size;
          //                  next_start += batch_size;
          //                }
          //              } else {
          //                current_start += batch_size;
          //                next_start += batch_size;
          //              }
          //              block_row_starts.push_back(j);
          //              ++matched_count;
          //              if (rank == 3 and i==22 and trans) {
          //                cout << " current row start: " << current_start<< "
          //                size: " << matched_count<<" " << j<<"
          //                "<<coords[j].row << endl;}
          //            }
          //          } else {
          if (!divided_equallaly) {
            if (j > block_col_starts[i] and
                matched_count % (batch_count) == 0) {
              current_start += last_proc_batch_size;
              next_start += batch_size;
            } else if (j > block_col_starts[i] and
                       (matched_count + 1) % (batch_count) == 0) {
              current_start += batch_size;
              next_start += last_proc_batch_size;
            } else {
              current_start += batch_size;
              next_start += batch_size;
            }
          } else {
            current_start += batch_size;
            next_start += batch_size;
          }
          //          }
          //          if (col_merged and rank==1) {
          //            cout << " current row start: " << current_start
          //                 << " size: " << matched_count << " " << j << " "
          //                 << coords[j].row << endl;
          //          }
        }

        // This modding step helps indexing.
        if (mod_ind) {
          coords[j].row %= batch_size;
        }
      }

      while (matched_count < expected_batch_count) {
        block_row_starts.push_back(block_col_starts[i + 1]);
        matched_count++;
      }
    }
    block_row_starts.push_back(coords.size());
  }

  void initialize_CSR_blocks() {

    if (col_merged) {
      this->divide_block_cols();
    }

    csr_linked_lists = std::vector<std::shared_ptr<CSRLinkedList<T>>>(1);

    for (int i = 0; i < 1; i++) {
      csr_linked_lists[i] = std::make_shared<CSRLinkedList<T>>();
    }

    int node_index = 0;

    for (uint64_t i = 0; i < coords.size(); i++) {
      if (transpose) {
        coords[i].col %= proc_col_width;
      } else {
        coords[i].row %= proc_row_width;
      }
    }

    Tuple<T> *coords_ptr = coords.data();

    if (col_merged) {

      for (int i = 0; i < block_col_starts.size() - 1; i++) {

        coords_ptr = coords_ptr + block_col_starts[i];
        int num_coords = block_col_starts[i + 1] - block_col_starts[i];
        (csr_linked_lists[0].get())
            ->insert(proc_row_width, gCols, num_coords, coords_ptr, num_coords,
                     transpose, node_index);
        node_index++;
      }

    } else {

      (csr_linked_lists[0].get())
          ->insert((transpose) ? gRows : proc_row_width,
                   (transpose) ? proc_col_width : gCols, coords.size(),
                   coords_ptr, coords.size(), transpose, node_index);
    }
  }

  void fill_col_ids(int batch_id,
                    vector<vector<uint64_t>> &proc_to_id_mapping) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto linkedList = csr_linked_lists[0];

    auto head = (linkedList.get())->getHeadNode();

    auto starting_index = batch_id * batch_size;

    auto end_index = std::min((batch_id + 1) * batch_size,
                              (transpose) ? proc_col_width : proc_row_width) -
                     1;

    //    while ((head.get())->data != nullptr) {
    auto csr_data = (head.get())->data;
    distblas::core::CSRHandle *handle = (csr_data.get())->handler.get();
    auto size =
        handle->rowStart[end_index + 1] - handle->rowStart[starting_index];
    cout << " rank " << rank << " starting_index " << starting_index
         << " end_index " << end_index << " batch " << batch_id << " size "
         << size << endl;
    int row_index = starting_index;
    int count = 0;
    for (auto i = handle->rowStart[starting_index];
         i < handle->rowStart[starting_index] + size; i++) {
      auto col_val = handle->col_idx[i];
      if (transpose) {
        // calculation of sending row_ids
        int owner_rank = col_val / proc_row_width;

        int diff = handle->rowStart[row_index + 1] - handle->rowStart[row_index];

        if (count >= diff) {
          count=0;
          row_index++;
        }
        cout << " rank " << rank << " col_val " << col_val << " target rank"
             << owner_rank << " row_id" << row_index <<" diff"<<diff<< endl;
        if (owner_rank != rank and diff > 0) {
          proc_to_id_mapping[owner_rank].push_back(row_index);
        }
        if (diff > 0) {
          count++;
        }

      } else {
        // calculation of receiving col_ids
        int owner_rank = col_val / proc_col_width;
        if (owner_rank != rank) {
          //          cout<<" rank "<<rank<<" trans"<<transpose<<" owner_rank
          //          "<<owner_rank<<" col_val "<<col_val  <<endl;
          proc_to_id_mapping[owner_rank].push_back(col_val);
        }
      }
    }
    //    }
  }

  CSRLinkedList<T> *get_batch_list(int batch_id) {
    return csr_linked_lists[batch_id].get();
  }

  void print_blocks_and_cols(bool trans) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //    cout << " rank " << rank << "_" << trans
    //         << " printing print_blocks_and_cols" << endl;
    int current_col_block = 0;
    for (int j = 0; j < csr_linked_lists.size(); j++) {
      if (j == 0 or
          j == csr_linked_lists.size() - 1) { // print first and last one
        cout << " rank " << rank << " j " << j << endl;
        auto linkedList = csr_linked_lists[j];

        auto head = (linkedList.get())->getHeadNode();

        int count = 0;
        while (head != nullptr) {
          string output_path = "blocks_rank" + to_string(rank) + "_trans" +
                               to_string(trans) + "_col_" +
                               to_string((trans) ? j : count) + "_row_" +
                               to_string((trans) ? count : j) + ".txt";
          char stats[500];
          strcpy(stats, output_path.c_str());
          ofstream fout(stats, std::ios_base::app);

          auto csr_data = (head.get())->data;

          int num_coords = (csr_data.get())->num_coords;

          cout << " rank " << rank << " j " << j << " num_coords " << num_coords
               << "_col_" + to_string((trans) ? j : count) + "_row_" +
                      to_string((trans) ? count : j)
               << endl;
          if (num_coords > 0) {
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

                  if (value > 60000) {
                    cout << "Rank " << rank << " j " << j
                         << " Large value encountered "
                         << " Row " << i << " col " << col << " value " << value
                         << endl;
                  }
                  fout << "(" << col << ", " << value << ") ";
                }
              }
              fout << endl;
            }
          }
          head = (head.get())->next;
          ++count;
        }
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
      fout << coords[i].row << " " << coords[i].value << " " << endl;
    }
  }

  ~SpMat() {}
};

} // namespace distblas::core
