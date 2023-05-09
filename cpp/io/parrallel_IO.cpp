/**
 * This implementation contains the CombBLAS based parallel IO Implementation.
 */
#include "CombBLAS/CombBLAS.h"
#include "parrallel_IO.hpp"
using namespace combblas;
using namespace distblas::io;

typedef SpParMat<int64_t, int, SpDCCols<int32_t, int>> PSpMat_s32p64_Int;


ParallelIO<T>::ParallelIO() {}



ParallelIO<T>::~ParallelIO() {}
