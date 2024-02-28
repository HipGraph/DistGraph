/**
 * This implements the CSR data structure. We use MKL routines to create CSR
 * from COO.
 */
#pragma once
#include "common.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <mkl.h>
#include <mkl_spblas.h>
#include <mpi.h>
#include <numeric>
#include <parallel/algorithm>
#include <string.h>

using namespace std;

namespace distblas::core {

template <typename VALUE_TYPE> class CSRLocal {

public:
  MKL_INT rows, cols;

  int max_nnz, num_coords = 0;

  bool transpose;

  unique_ptr<CSRHandle> handler = nullptr;

  CSRLocal() {}

  CSRLocal(MKL_INT rows, MKL_INT cols, MKL_INT max_nnz, Tuple<VALUE_TYPE> *coords,
           int num_coords, bool transpose) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout<<" rank "<<rank<<"num_coords "<<num_coords<<" cols "<<cols<<endl;
    if (num_coords > 0) {
      this->transpose = transpose;
      this->num_coords = num_coords;
      this->rows = rows;
      this->cols = cols;

      this->handler = unique_ptr<CSRHandle>(new CSRHandle());


      // This setup is really clunky, but I don't have time to fix it.
      vector<MKL_INT> rArray(num_coords, 0);
      vector<MKL_INT> cArray(num_coords, 0);
      vector<double> vArray(num_coords, 0.0);

#pragma omp parallel for schedule(static)
      for (int i = 0; i < num_coords; i++) {
        rArray[i] = coords[i].row;
        cArray[i] = coords[i].col;
        vArray[i] = static_cast<double>(coords[i].value);
      }

      sparse_operation_t op;
      if (transpose) {
        op = SPARSE_OPERATION_TRANSPOSE;
      } else {
        op = SPARSE_OPERATION_NON_TRANSPOSE;
      }

      sparse_matrix_t tempCOO, tempCSR;

      sparse_status_t status_coo = mkl_sparse_d_create_coo(
          &tempCOO, SPARSE_INDEX_BASE_ZERO, rows, cols, max(num_coords, 1),
          rArray.data(), cArray.data(), vArray.data());

      sparse_status_t status_csr =
          mkl_sparse_convert_csr(tempCOO, op, &tempCSR);

      mkl_sparse_destroy(tempCOO);
      vector<MKL_INT>().swap(rArray);
      vector<MKL_INT>().swap(cArray);
      vector<double>().swap(vArray);

      sparse_index_base_t indexing;
      MKL_INT *rows_start, *rows_end, *col_idx;
      double *values;

      mkl_sparse_d_export_csr(tempCSR, &indexing, &(this->rows), &(this->cols),
                              &rows_start, &rows_end, &col_idx, &values);
      int rv = 0;
      for (int i = 0; i < num_coords; i++) {
        while (rv < this->rows && i >= rows_start[rv + 1]) {
          rv++;
        }
        coords[i].row = rv;
        coords[i].col = col_idx[i];
        coords[i].value = static_cast<VALUE_TYPE>(values[i]);
      }

      //      assert(num_coords <= max_nnz);

      (handler.get())->values.resize(max_nnz == 0 ? 1 : max_nnz);
      (handler.get())->col_idx.resize(max_nnz == 0 ? 1 : max_nnz);
      (handler.get())->row_idx.resize(max_nnz == 0 ? 1 : max_nnz);
      (handler.get())->rowStart.resize(this->rows + 1);
      // Copy over row indices
#pragma omp parallel for schedule(static)
      for (int i = 0; i < num_coords; i++) {
        (handler.get())->row_idx[i] = coords[i].row;
      }

      memcpy((handler.get())->values.data(), values,
             sizeof(double) * max(num_coords, 1));
      memcpy((handler.get())->col_idx.data(), col_idx,
             sizeof(MKL_INT) * max(num_coords, 1));
      memcpy((handler.get())->rowStart.data(), rows_start,
             sizeof(MKL_INT) * this->rows);
      (handler.get())->rowStart[this->rows] = max(num_coords, 1);

      mkl_sparse_d_create_csr(
          &((handler.get())->mkl_handle), SPARSE_INDEX_BASE_ZERO, this->rows,
          this->cols, (handler.get())->rowStart.data(),
          (handler.get())->rowStart.data() + 1, (handler.get())->col_idx.data(),
          (handler.get())->values.data());
      cout<<" rank "<<rank<<"mkl_sparse_d_create_csr  completed "<<num_coords<<" cols "<<cols<<endl;

      mkl_sparse_destroy(tempCSR);
    }
  }

  CSRLocal(vector<vector<Tuple<VALUE_TYPE>>> *sparse_data_collector) {
     handler = unique_ptr<CSRHandle>(new CSRHandle());
     handler->rowStart.resize(sparse_data_collector->size() + 1, 0);
    for (auto i = 0; i < sparse_data_collector->size(); i++) {

      std::vector<MKL_INT> firstValues;
      std::vector<double> Values;
      vector<Tuple<VALUE_TYPE>> filtered;
      std::copy_if((*sparse_data_collector)[i].begin(), (*sparse_data_collector)[i].end(),
                     std::back_inserter(filtered),
                     [](const Tuple<VALUE_TYPE> &tuple) {
                         return tuple.col>=0;
                     });

      std::transform(filtered.begin(), filtered.end(),
                     std::back_inserter(firstValues),
                     [](const Tuple<VALUE_TYPE> &tuple) {
                       return tuple.col;
                     });

      std::transform(filtered.begin(), filtered.end(),
                     std::back_inserter(Values),
                     [](const Tuple<VALUE_TYPE> &tuple) {
                       return tuple.value;
                     });

      // Insert the first values into ((handler.get())->col_idx)
       handler->col_idx.insert((handler->col_idx).end(), firstValues.begin(), firstValues.end());
       handler->values.insert((handler->values).end(), Values.begin(), Values.end());
       handler->rowStart[i+1]=filtered.size()+handler->rowStart[i];
    }
  }

  CSRLocal<VALUE_TYPE>& CSRLocal<VALUE_TYPE>::operator=(const CSRLocal<VALUE_TYPE>& other) {
    if (this != &other) {
      // Copy all necessary data members
      rows = other.rows;
      cols = other.cols;
      max_nnz = other.max_nnz;
      num_coords = other.num_coords;
      transpose = other.transpose;

      // Copy the CSRHandle using its copy assignment operator or copy constructor
      handler = make_unique<CSRHandle>(*other.handler);
    }
    return *this;
  }

  ~CSRLocal() {
    //    mkl_sparse_destroy((handler.get())->mkl_handle);
  }
};

} // namespace distblas::core
