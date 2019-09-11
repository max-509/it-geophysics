#ifndef _COH_SUMM
#define _COH_SUMM

#include <cstdint>
#include <cstddef>

void compute(const float* rec_samples, const float* rec_coords, const int32_t* sources_times, ptrdiff_t n_samples, ptrdiff_t n_rec, ptrdiff_t n_xyz, float* result_data);

#endif /*_COH_SUMM*/