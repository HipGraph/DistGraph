#pragma once

#include "../net/process_3D_grid.hpp"
#include "common.h"
#include "distributed_mat.hpp"
#include "sparse_mat.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <random>
#include <unordered_map>

using namespace std;

using namespace distblas::net;

namespace distblas::core {

/**
 * This class represents  the dense matrix.
 */
    template<typename INDEX_TYPE, typename VALUE_TYPE>
    class DenseMat : public DistributedMat<INDEX_TYPE,VALUE_TYPE> {

    private:
    public:
//        uint64_t rows;
//        uint64_t cols;
        unique_ptr <vector<unordered_map < INDEX_TYPE, CacheEntry<VALUE_TYPE>>>> cachePtr;
        unique_ptr <vector<unordered_map < INDEX_TYPE, CacheEntry<VALUE_TYPE>>>> tempCachePtr;
//        Process3DGrid *grid;

        unique_ptr<vector<VALUE_TYPE>> nCoordinatePtr;
        VALUE_TYPE * nCoordinates=nullptr;

        /**
         *
         * @param rows Number of rows of the matrix
         * @param cols  Number of cols of the matrix
         * @param init_mean  initialize with normal distribution with given mean
         * @param std  initialize with normal distribution with given standard
         * deviation
         */
        DenseMat(Process3DGrid *grid, INDEX_TYPE rows, INDEX_TYPE cols, bool lazy = false) : DistributedMat<INDEX_TYPE,VALUE_TYPE>(grid,rows,cols) {
            this->nCoordinatePtr = make_unique < vector < VALUE_TYPE >> (rows * cols);
            this->nnz_count = make_unique < vector < INDEX_TYPE >> (rows, 0);
            this->state_metadata = make_unique < vector < vector < VALUE_TYPE>>>(rows, vector<VALUE_TYPE>(cols, 0));
            this->nCoordinates= this->nCoordinatePtr->data();
            if (!lazy) {
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        VALUE_TYPE val = -1.0 + 2.0 * rand() / (RAND_MAX + 1.0);
                        this->nCoordinates[i * cols + j] = val;
                    }
                }
            }
            bootstrap();
        }


        DenseMat(Process3DGrid *grid, INDEX_TYPE rows, INDEX_TYPE cols, VALUE_TYPE* data) : DistributedMat<INDEX_TYPE,VALUE_TYPE>(grid, rows,cols) {
            this->nCoordinatePtr = make_unique < vector < VALUE_TYPE >> (rows * cols);
            this->nnz_count = make_unique < vector < INDEX_TYPE >> (rows, 0);
            this->state_metadata = make_unique < vector < vector < VALUE_TYPE>>>(rows, vector<VALUE_TYPE>(cols, 0));
            this->nCoordinates= data;
            bootstrap();
        }

        DenseMat(Process3DGrid *grid, string input_file) : DistributedMat<INDEX_TYPE,VALUE_TYPE>() {
            this->nCoordinatePtr = make_unique < vector < VALUE_TYPE >> (this->rows * this->cols);
            this->nnz_count = make_unique < vector < INDEX_TYPE >> (this->rows, 0);
            this->state_metadata = make_unique < vector < vector < VALUE_TYPE>>>(this->rows, vector<VALUE_TYPE>(this->cols, 0));
            this->nCoordinates= data;
            bootstrap();
        }



        ~DenseMat() {}

        void bootstrap() override {
            this->cachePtr =
                    make_unique < vector < unordered_map < INDEX_TYPE, CacheEntry<VALUE_TYPE>>>>(this->grid->col_world_size);
            this->tempCachePtr =
                    make_unique < vector < unordered_map < INDEX_TYPE, CacheEntry<VALUE_TYPE>>>>(this->grid->col_world_size);
        }

        void fetch_local_data(VALUE_TYPE *stdArray, int local_key) {
            int base_index = local_key * this->cols;
            std::copy(nCoordinates + base_index,nCoordinates + base_index + this->cols, stdArray);
        }

        void multiply(DenseMat<INDEX_TYPE, VALUE_TYPE> *other, DenseMat<INDEX_TYPE, VALUE_TYPE> *output) {

            assert(this->cols == other->rows);
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < this->rows; ++i) {
                for (int j = 0; j < other->cols; ++j) {
                    VALUE_TYPE value = 0;
                    for (int k = 0; k < this->cols; ++k) {
                        value += this->nCoordinates[i * this->cols + k] * other->nCoordinates[k * other->cols + j];
                    }
                    output->nCoordinates[i * other->cols + j] = value;
                }
            }
        }

        void invalidate_cache(int current_itr, int current_batch, bool temp) {
            if (temp) {
                purge_temp_cache();
            } else {
                for (int i = 0; i < this->grid->col_world_size; i++) {
                    auto &arrayMap = (*cachePtr)[i];
                    for (auto it = arrayMap.begin(); it != arrayMap.end();) {
                        distblas::core::CacheEntry<VALUE_TYPE> cache_ent =
                                it->second;
                        if (cache_ent.inserted_itr < current_itr and
                            cache_ent.inserted_batch_id <= current_batch) {
                            it = arrayMap.erase(it);
                        } else {
                            // Move to the next item
                            ++it;
                        }
                    }
                }
            }
        }

        void purge_temp_cache() {
            for (int i = 0; i < this->grid->col_world_size; i++) {
                (*this->tempCachePtr)[i].clear();
                std::unordered_map<INDEX_TYPE, CacheEntry<VALUE_TYPE>>().swap((*this->tempCachePtr)[i]);
            }
        }

        //******************** Utility methods for debugging ************************
        void print_matrix() {
            int rank = this->grid->rank_in_col;
            string output_path = "embedding" + to_string(rank) + ".txt";
            char stats[500];
            strcpy(stats, output_path.c_str());
            ofstream fout(stats, std::ios_base::app);
            for (int i = 0; i < this->rows; ++i) {
                fout << (i + 1) << " ";
                for (int j = 0; j < this->cols; ++j) {
                    fout << this->nCoordinates[i * this->cols + j] << " ";
                }
                fout << endl;
            }
        }

        void print_matrix_rowptr(int iter) {
            int rank = this->grid->rank_in_col;
            string output_path =
                    "rank_" + to_string(rank) + "itr_" + to_string(iter) + "_embedding.txt";
            char stats[500];
            strcpy(stats, output_path.c_str());
            ofstream fout(stats, std::ios_base::app);
            //    fout << (*this->matrixPtr).rows() << " " << (*this->matrixPtr).cols()
            //         << endl;
            for (int i = 0; i < rows; ++i) {
                fout << i + rank * this->rows << " ";
                for (int j = 0; j < this->cols; ++j) {
                    fout << this->nCoordinates[i * this->cols + j] << " ";
                }
                fout << endl;
            }
        }

        void print_cache(int iter) {
            int rank = grid->rank_in_col;

            for (int i = 0; i < (*this->cachePtr).size(); i++) {
                unordered_map <INDEX_TYPE, CacheEntry<VALUE_TYPE>> map =
                        (*this->cachePtr)[i];
//      (*this->tempCachePtr)[i];

                string output_path = "rank_" + to_string(rank) + "remote_rank_" +
                                     to_string(i) + " itr_" + to_string(iter) + ".txt";
                char stats[500];
                strcpy(stats, output_path.c_str());
                ofstream fout(stats, std::ios_base::app);

                for (const auto &kvp: map) {
                    INDEX_TYPE key = kvp.first;
                    vector<VALUE_TYPE> value = kvp.second.value;
                    fout << key << " ";
                    for (int i = 0; i < this->cols; ++i) {
                        fout << value[i] << " ";
                    }
                    fout << std::endl;
                }
            }
        }


    };

} // namespace distblas::core
