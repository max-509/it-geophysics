#ifndef _COH_SUMM
#define _COH_SUMM

#include <cstddef>

void compute(const float* rec_samples, const float* rec_coords, const float* area_coords, ptrdiff_t n_samples, ptrdiff_t n_rec, ptrdiff_t n_xyz, float dt, float vv, float* result_data);

#endif /*_COH_SUMM*/