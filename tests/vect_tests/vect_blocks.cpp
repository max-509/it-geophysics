#include <fstream>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <omp.h>
#include <immintrin.h>
#include <functional>
#include <cstddef>

inline float calc_radius(float dx, float dy, float dz) {
    return sqrt(dx*dx+dy*dy+dz*dz);
}

inline __m256 vect_calc_radius(__m256 dx, __m256 dy, __m256 dz) {
    return _mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)), _mm256_mul_ps(dz, dz)));
}

int main(int argc, char const *argv[]) {
    std::ifstream data_file, receivers_file;
    data_file.open("../Data_noise_free.bin", std::ios::binary);
    if (!data_file.is_open()) {
        std::cerr << "Can't open Data_noise_free.bin" << std::endl;
        return 1;
    }
    receivers_file.open("../Receivers_Array.bin", std::ios::binary);
    if (!receivers_file.is_open()) {
        std::cerr << "Can't open Receivers_Array.bin" << std::endl;
        return 1;
    }

    ptrdiff_t rec_count = 2000;
    ptrdiff_t times = 10000;

    std::unique_ptr<float[]> rec_times{new float[times*rec_count]()};
    data_file.read(reinterpret_cast<char*>(rec_times.get()), rec_count*times*sizeof(float));

    std::unique_ptr<float[]> rec_coords{new float[3*rec_count]()};
    receivers_file.read(reinterpret_cast<char*>(rec_coords.get()), rec_count*3*sizeof(float));

    float dt = 2e-3;

    ptrdiff_t nx = 140;
    ptrdiff_t ny = 140;
    ptrdiff_t nz = 140;

    float vv = 3000;

    long long x0 = -1000;
    long long x1 = 1000;
    long long y0 = -1000;
    long long y1 = 1000;
    long long z0 = 500;
    long long z1 = 2500;

    std::unique_ptr<float[]> area_discr{new float[times]};

    float dx, dy, dz;
    if (1 < nx) {
        dx = ((float)(x1-x0))/(nx-1);
    } else {
        dx = 0;
    }
    if (1 < ny) {
        dy = ((float)(y1-y0))/(ny-1);
    } else {
        dy = 0;
    }
    if (1 < nz) {
        dz = ((float)(z1-z0))/(nz-1);
    } else {
        dz = 0;
    }

    double t1, t2;

    ptrdiff_t rec_block_size = rec_count;
    ptrdiff_t times_block_size = times;
    if (argc >= 2) {
        rec_block_size = atoi(argv[1]);
    }
    if (argc >= 3) {
        times_block_size = atoi(argv[2]);
    }
    std::unique_ptr<ptrdiff_t[]> min_ind_arr{new ptrdiff_t[nx*ny*nz]};
    std::unique_ptr<ptrdiff_t[]> ind_arr{new ptrdiff_t[nx*ny*nz*rec_count]};
    __m256 vect_rec_coord[(rec_count/8)*3];

    //**********************************************************//
    t1 = omp_get_wtime();
    if (rec_count >= 8) {
        for (ptrdiff_t m = 0; m < rec_count; m+=8) {
            for (ptrdiff_t i = 0; i < 3; ++i) {
                vect_rec_coord[(m/8)*3+i] = _mm256_set_ps(rec_coords[(m+7)*3+i], rec_coords[(m+6)*3+i], rec_coords[(m+5)*3+i], rec_coords[(m+4)*3+i],
                                                          rec_coords[(m+3)*3+i], rec_coords[(m+2)*3+i], rec_coords[(m+1)*3+i], rec_coords[m*3+i]);
            }
        }
        __m256 vect_ind_min;
        for (ptrdiff_t i = 0; i < nz; ++i) {
            for (ptrdiff_t j = 0; j < nx; ++j) {
                for (ptrdiff_t k = 0; k < ny; ++k) {
                    for (ptrdiff_t m = 0; m < rec_count; m+=8) {
                        __m256 vect_ind = _mm256_add_ps(_mm256_set1_ps(1.0f),
                                                        _mm256_round_ps(_mm256_div_ps(vect_calc_radius(
                                                        _mm256_sub_ps(_mm256_set1_ps(x0+j*dx), vect_rec_coord[(m/8)*3]),
                                                        _mm256_sub_ps(_mm256_set1_ps(y0+k*dy), vect_rec_coord[(m/8)*3+1]),
                                                        _mm256_sub_ps(_mm256_set1_ps(z0+i*dz), vect_rec_coord[(m/8)*3+2])),
                                                        _mm256_mul_ps(_mm256_set1_ps(vv), _mm256_set1_ps(dt))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)));
                        if (0 != m) {
                            vect_ind_min = _mm256_min_ps(vect_ind_min, vect_ind);
                        } else {
                            vect_ind_min = vect_ind;
                        }
                        float temp_ind[8];
                        _mm256_store_ps(temp_ind, vect_ind);
                        for (ptrdiff_t v = 0; v < 8; ++v) {
                            ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m+v] = temp_ind[v];
                        }
                    }

                    float min_indexes[8];
                    _mm256_store_ps(min_indexes, vect_ind_min);
                    min_ind_arr[i*nx*ny+j*ny+k] = *std::min_element(min_indexes, min_indexes+8);

                    for (ptrdiff_t m = rec_count-(rec_count%8); m < rec_count; ++m) {
                        ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m] = round(calc_radius((x0+j*dx)-rec_coords[m*3],
                                                                                                    (y0+k*dy)-rec_coords[m*3+1],
                                                                                                    (z0+i*dz)-rec_coords[m*3+2])
                                                                                                    /(vv*dt)) + 1;
                        min_ind_arr[i*nx*ny+j*ny+k] = std::min(min_ind_arr[i*nx*ny+j*ny+k],
                                                               ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m]);
                    }
                }
            }
        }
    } else {
        for (ptrdiff_t i = 0; i < nz; ++i) {
            for (ptrdiff_t j = 0; j < nx; ++j) {
                for (ptrdiff_t k = 0; k < ny; ++k) {
                    ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count] = round(calc_radius((x0+j*dx)-rec_coords[0],
                                                                                              (y0+k*dy)-rec_coords[1],
                                                                                              (z0+i*dz)-rec_coords[2])
                                                                                              /(vv*dt)) + 1;
                    min_ind_arr[i*nx*ny+j*ny+k] = ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count];

                    for (ptrdiff_t m = 1; m < rec_count; ++m) {
                        ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m] = round(calc_radius((x0+j*dx)-rec_coords[m*3],
                                                                                                    (y0+k*dy)-rec_coords[m*3+1],
                                                                                                    (z0+i*dz)-rec_coords[m*3+2])
                                                                                                    /(vv*dt)) + 1;
                        min_ind_arr[i*nx*ny+j*ny+k] = std::min(min_ind_arr[i*nx*ny+j*ny+k],
                                                               ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m]);
                    }
                }
            }
        }
    }
    t2 = omp_get_wtime();
    //**********************************************************//

    std::cout << "vect Time: " << t2-t1 << std::endl;
    return 0;

    t1 = omp_get_wtime();
    //algorithm
    //******************************************************//
    #pragma omp parallel
    {
        if (rec_count >= 8) {
            #pragma omp for collapse(2)
            for (ptrdiff_t m = 0; m < rec_count; m+=8) {
                for (ptrdiff_t i = 0; i < 3; ++i) {
                    vect_rec_coord[(m/8)*3+i] = _mm256_set_ps(rec_coords[(m+7)*3+i], rec_coords[(m+6)*3+i], rec_coords[(m+5)*3+i], rec_coords[(m+4)*3+i],
                                                              rec_coords[(m+3)*3+i], rec_coords[(m+2)*3+i], rec_coords[(m+1)*3+i], rec_coords[m*3+i]);
                }
            }
            __m256 vect_ind_min;
            #pragma omp for collapse(3)
            for (ptrdiff_t i = 0; i < nz; ++i) {
                for (ptrdiff_t j = 0; j < nx; ++j) {
                    for (ptrdiff_t k = 0; k < ny; ++k) {
                        for (ptrdiff_t m = 0; m < rec_count; m+=8) {
                            __m256 vect_ind = _mm256_add_ps(_mm256_set1_ps(1.0f),
                                                            _mm256_round_ps(_mm256_div_ps(vect_calc_radius(
                                                            _mm256_sub_ps(_mm256_set1_ps(x0+j*dx), vect_rec_coord[(m/8)*3]),
                                                            _mm256_sub_ps(_mm256_set1_ps(y0+k*dy), vect_rec_coord[(m/8)*3+1]),
                                                            _mm256_sub_ps(_mm256_set1_ps(z0+i*dz), vect_rec_coord[(m/8)*3+2])),
                                                            _mm256_mul_ps(_mm256_set1_ps(vv), _mm256_set1_ps(dt))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)));
                            if (0 != m) {
                                vect_ind_min = _mm256_min_ps(vect_ind_min, vect_ind);
                            } else {
                                vect_ind_min = vect_ind;
                            }
                            float temp_ind[8];
                            _mm256_store_ps(temp_ind, vect_ind);
                            for (ptrdiff_t v = 0; v < 8; ++v) {
                                ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m+v] = temp_ind[v];
                            }
                        }

                        float min_indexes[8];
                        _mm256_store_ps(min_indexes, vect_ind_min);
                        min_ind_arr[i*nx*ny+j*ny+k] = *std::min_element(min_indexes, min_indexes+8);

                        for (ptrdiff_t m = rec_count-(rec_count%8); m < rec_count; ++m) {
                            ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m] = round(calc_radius((x0+j*dx)-rec_coords[m*3],
                                                                                                        (y0+k*dy)-rec_coords[m*3+1],
                                                                                                        (z0+i*dz)-rec_coords[m*3+2])
                                                                                                        /(vv*dt)) + 1;
                            min_ind_arr[i*nx*ny+j*ny+k] = std::min(min_ind_arr[i*nx*ny+j*ny+k],
                                                                   ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m]);
                        }
                    }
                }
            }
        } else {
            #pragma omp for collapse(3)
            for (ptrdiff_t i = 0; i < nz; ++i) {
                for (ptrdiff_t j = 0; j < nx; ++j) {
                    for (ptrdiff_t k = 0; k < ny; ++k) {
                        ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count] = round(calc_radius((x0+j*dx)-rec_coords[0],
                                                                                                  (y0+k*dy)-rec_coords[1],
                                                                                                  (z0+i*dz)-rec_coords[2])
                                                                                                  /(vv*dt)) + 1;
                        min_ind_arr[i*nx*ny+j*ny+k] = ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count];

                        for (ptrdiff_t m = 1; m < rec_count; ++m) {
                            ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m] = round(calc_radius((x0+j*dx)-rec_coords[m*3],
                                                                                                        (y0+k*dy)-rec_coords[m*3+1],
                                                                                                        (z0+i*dz)-rec_coords[m*3+2])
                                                                                                        /(vv*dt)) + 1;
                            min_ind_arr[i*nx*ny+j*ny+k] = std::min(min_ind_arr[i*nx*ny+j*ny+k],
                                                                   ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m]);
                        }
                    }
                }
            }
        }
        #pragma omp for 
    	for (ptrdiff_t c_t = 0; c_t < times; c_t += times_block_size) {
            for (ptrdiff_t c_r = 0; c_r < rec_count; c_r += rec_block_size) {
    		    for (ptrdiff_t i = 0; i < nz; ++i) {
                    for (ptrdiff_t j = 0; j < nx; ++j) {
                        for (ptrdiff_t k = 0; k < ny; ++k) {
                            ptrdiff_t min_ind = min_ind_arr[i*nx*ny+j*ny+k];
                            for (ptrdiff_t m = c_r; m < std::min(c_r+rec_block_size, rec_count); ++m) {
                                ptrdiff_t ind = ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m]-min_ind;
                                for (ptrdiff_t l = c_t; l < std::min(c_t+times_block_size, times-ind); ++l) {
                                    area_discr[i*nx*ny*times+j*ny*times+k*times+l] += rec_times[m*times+ind+l];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    //******************************************************//
    t2 = omp_get_wtime();

    std::cout << "Time: " << t2-t1 << std::endl;

    return 0;
}