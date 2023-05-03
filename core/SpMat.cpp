#include "../include/DistBLAS/SpMat.hpp"
#include "../include/DistBLAS/Common.hpp"

using namespace distblas::core;

SpMat::SpMat(vector<Tuple<T>> &coords, int gRows, int gCols, int gNNz) {
  this->gRows = gRows;
  this->gCols = gCols;
  this->gNNz = gNNz;
  this->coords = coords;
}

SpMat::~SpMat() {

}

void SpMat::initialize_csr() {

}
