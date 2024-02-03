#pragma once
#include "../core/dense_mat.hpp"
#include "../core/sparse_mat.hpp"
#include "../net/process_3D_grid.hpp"
#include "CombBLAS/CombBLAS.h"
#include <mpi.h>
#include <string>
#include <vector>

using namespace std;
using namespace distblas::core;
using namespace combblas;
namespace distblas::io {

typedef SpParMat<int64_t, int, SpDCCols<int32_t, int>> PSpMat_s32p64_Int;

/**
 * This class implements IO operations of DistBlas library.
 */
class ParallelIO {
private:
public:
  ParallelIO();
  ~ParallelIO();

  /**
   * Interface for parallel reading of Matrix Market formatted files
   * @param file_path
   */
  template <typename T>
  void parallel_read_MM(string file_path, distblas::core::SpMat<T> *sp_mat,
                        bool copy_col_to_value) {
    MPI_Comm WORLD;
    MPI_Comm_dup(MPI_COMM_WORLD, &WORLD);

    int proc_rank, num_procs;
    MPI_Comm_rank(WORLD, &proc_rank);
    MPI_Comm_size(WORLD, &num_procs);

    shared_ptr<CommGrid> simpleGrid;
    simpleGrid.reset(new CommGrid(WORLD, num_procs, 1));

    unique_ptr<PSpMat_s32p64_Int> G =
        unique_ptr<PSpMat_s32p64_Int>(new PSpMat_s32p64_Int(simpleGrid));

    uint64_t nnz;

    G.get()->ParallelReadMM(file_path, true, maximum<T>());

    nnz = G.get()->getnnz();
    if (proc_rank == 0) {
      cout << "File reader read " << nnz << " nonzeros." << endl;
    }
    SpTuples<int64_t, T> tups(G.get()->seq());
    tuple<int64_t, int64_t, T> *values = tups.tuples;

    vector<Tuple<T>> coords;
    coords.resize(tups.getnnz());

#pragma omp parallel for
    for (int i = 0; i < tups.getnnz(); i++) {
      coords[i].row = get<0>(values[i]);
      coords[i].col = get<1>(values[i]);
      if (copy_col_to_value) {
        coords[i].value = get<1>(values[i]);
      } else {
        coords[i].value = get<2>(values[i]);
      }
    }

    int rowIncrement = G->getnrow() / num_procs;

#pragma omp parallel for
    for (int i = 0; i < coords.size(); i++) {
      coords[i].row += rowIncrement * proc_rank;
    }

    sp_mat->coords = coords;
    sp_mat->gRows = G.get()->getnrow();
    sp_mat->gCols = G.get()->getncol();
    sp_mat->gNNz = G.get()->getnnz();
  }

  template <typename T, typename SPT>
  void parallel_write(string file_path, T *nCoordinates, uint64_t rows,
                      uint64_t cols, Process3DGrid *grid,
                      distblas::core::SpMat<SPT> *sp_mat) {
    MPI_File fh;
    MPI_File_open(grid->col_world, file_path.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    uint64_t expected_rows = rows;
    if (grid->rank_in_col == grid->col_world_size - 1) {
      auto expected_last_rows = sp_mat->gRows - rows * grid->rank_in_col;
      cout << " expected rows " << expected_last_rows << endl;
      expected_rows = min(expected_last_rows, rows);
    }

    //    int offset= rows-expected_rows;
    cout << " rank :" << grid->rank_in_col << " expected rows " << expected_rows
         << endl;
    size_t total_size = 0;
    for (uint64_t i = 0; i < expected_rows; ++i) {
      total_size +=
          snprintf(nullptr, 0, "%lu", i + 1 + grid->rank_in_col * rows);
      for (int j = 0; j < cols; ++j) {
        total_size += snprintf(nullptr, 0, " %.5f", nCoordinates[i * cols + j]);
      }
      total_size += snprintf(nullptr, 0, "\n");
    }

    // Allocate memory dynamically
    char *buffer =
        (char *)malloc(total_size + 1); // +1 for the null-terminating character
    if (buffer == nullptr) {
      // Handle allocation failure
      cout << "Memory allocation failed." << endl;
      return;
    }

    char *current_position = buffer;

    for (uint64_t i = 0; i < expected_rows; ++i) {
      current_position += snprintf(current_position, total_size, "%lu",
                                   i + 1 + grid->rank_in_col * rows);
      for (int j = 0; j < cols; ++j) {
        current_position += snprintf(current_position, total_size, " %.5f",
                                     nCoordinates[i * cols + j]);
      }
      current_position += snprintf(current_position, total_size, "\n");
    }

    MPI_File_write_ordered(fh, buffer, current_position - buffer, MPI_CHAR,
                           MPI_STATUS_IGNORE);

    // Free the dynamically allocated memory
    free(buffer);

    MPI_File_close(&fh);
  }

  template <typename VALUE_TYPE>
  void build_sparse_random_matrix(int rows, int cols, float density, int seed,
                                  vector<Tuple<VALUE_TYPE>> &sparse_coo,
                                  Process3DGrid *grid, bool save_output,
                                  string output = "random.txt") {
    MPI_File fh;
    MPI_File_open(grid->col_world, (char*)output.c_str(),MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> uni_dist(0, 1);
    std::normal_distribution<float> norm_dist(0, 1);

    size_t total_size = 0;
    // follow row major order
    for (uint64_t j = 0; j < rows; ++j) {
      for (uint64_t i = 0; i < cols; ++i) {
        // take value at uniformly at random and check value is greater than
        // density.If so make that entry empty
        if (uni_dist(gen) <= density) {
          // normal distribution for generate projection matrix.
          VALUE_TYPE val = (VALUE_TYPE)norm_dist(gen);
          Tuple<VALUE_TYPE> t;
          t.row = j;
          t.col = i;
          t.value = val;
          (sparse_coo).push_back(t);
          total_size += snprintf(nullptr, 0, "%lu", t.row + 1 + grid->rank_in_col * rows);
          total_size += snprintf(nullptr, 0, "%lu", t.col + 1);
          total_size += snprintf(nullptr, 0, " %.5f", t.value);
          total_size += snprintf(nullptr, 0, "\n");
        }
      }
    }
    cout<<" total file size "<<total_size<<" nnz "<<sparse_coo.size()<<endl;
    if (save_output) {
      char *buffer = (char *)malloc(total_size +1); // +1 for the null-terminating character
      if (buffer == nullptr) {
        // Handle allocation failure
        cout << "Memory allocation failed." << endl;
        return;
      }
      char *current_position = buffer;
      cout<<" intial current_position "<<*current_position<<" "<<"buffer"<< *buffer<<endl;
      size_t remain =  total_size - (current_position - buffer);
      for (size_t i = 0; i < sparse_coo.size(); ++i) {
        remain =  total_size - (current_position - buffer);
        current_position += snprintf(current_position, remain, "%lu",sparse_coo[i].row+ 1 + grid->rank_in_col * rows);
        remain =  total_size - (current_position - buffer);
        current_position += snprintf(current_position, remain, "%lu",sparse_coo[i].col+ 1);
        remain =  total_size - (current_position - buffer);
        current_position += snprintf(current_position, remain, " %.5f", sparse_coo[i].value);
        remain =  total_size - (current_position - buffer);
        current_position += snprintf(current_position, remain, "\n");
      }
      cout<<" final current_position "<<*current_position<<" "<<"buffer"<< *buffer<<endl;

      MPI_Status status;

      MPI_File_write_ordered(fh, buffer, current_position - buffer, MPI_CHAR, &status);

      // Ensure that all processes have completed their writes
      MPI_Barrier(MPI_COMM_WORLD);

      // Now you can use the 'status' variable to get information about the completed operation
      int error_code;
      MPI_Error_class(status.MPI_ERROR, &error_code);

      if (error_code != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int length;
        MPI_Error_string(error_code, error_string, &length);
        cout<<"MPI error: %s"<<error_string<<endl;
      }
      // Free the dynamically allocated memory
      MPI_File_close(&fh);
      free(buffer);
    }
  }
};
} // namespace distblas::io
