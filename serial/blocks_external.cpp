#include <fstream>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <omp.h>
#include <immintrin.h>

//(15, 1000) - best variant

//Laptop
//Test times_rec_blocks, where pair-numbers - (rec_block_size, times_block_size)
//r27-32 : (10, 10000); (10, 5000); (10, 2500); (10, 2000); (10, 1000); (10, 500);
//TODO: не забыть попробовать векторизовать цикл для приемников

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

    size_t rec_count = 2000;
    size_t times = 10000;

    std::unique_ptr<float[]> rec_times{new float[rec_count*times]()};
    data_file.read(reinterpret_cast<char*>(rec_times.get()), rec_count*times*sizeof(float));

    std::unique_ptr<float[]> rec_coords{new float[rec_count*3]()};
    receivers_file.read(reinterpret_cast<char*>(rec_coords.get()), rec_count*3*sizeof(float));

    float dt = 2e-3;

    size_t nx = 50;
    size_t ny = 50;
    size_t nz = 50;

    float vv = 3000;

    long long x0 = -1000;
    long long x1 = 1000;
    long long y0 = -1000;
    long long y1 = 1000;
    long long z0 = 500;
    long long z1 = 2500;

    std::unique_ptr<float[]> area_discr{new float[times*nx*ny*nz]()};

    float dx, dy, dz;
    if (1 < nx) dx = ((float)(x1-x0))/(nx-1);
    if (1 < ny) dy = ((float)(y1-y0))/(ny-1);
    if (1 < nz) dz = ((float)(z1-z0))/(nz-1);

    double t1, t2;

    size_t rec_block_size = rec_count;
    size_t times_block_size = times;
    if (argc >= 2) {
        rec_block_size = atoi(argv[1]);
    }
    if (argc >= 3) {
        times_block_size = atoi(argv[2]);
    }

    std::unique_ptr<size_t[]> min_ind_arr{new size_t[nx*ny*nz]()};
    std::unique_ptr<size_t[]> ind_arr{new size_t[nx*ny*nz*rec_count]()};
    __m256 vect_rec_coord[(rec_count/8)*3];
    t1 = omp_get_wtime();
    //algorithm
    //******************************************************//
    #pragma omp parallel
    {
        if (rec_count >= 8) {
            #pragma omp for collapse(2)
            for (size_t m = 0; m < rec_count; m+=8) {
                for (size_t i = 0; i < 3; ++i) {
                    vect_rec_coord[(m/8)*3+i] = _mm256_set_ps(rec_coords[(m+7)*3+i], rec_coords[(m+6)*3+i], rec_coords[(m+5)*3+i], rec_coords[(m+4)*3+i],
                                                              rec_coords[(m+3)*3+i], rec_coords[(m+2)*3+i], rec_coords[(m+1)*3+i], rec_coords[m*3+i]);
                }
            }
        }
        #pragma omp for collapse(4)
        for (size_t i = 0; i < nz; ++i) {
            for (size_t j = 0; j < nx; ++j) {
                for (size_t k = 0; k < ny; ++k) {
                    for (size_t m = 0; m < rec_count; m+=8) {
                        if (m != rec_count-(rec_count%8)) {
                            __m256 vect_ind = _mm256_add_ps(_mm256_set1_ps(1.0f),
                                                            _mm256_round_ps(_mm256_div_ps(vect_calc_radius(
                                                            _mm256_sub_ps(_mm256_set1_ps(x0+j*dx), vect_rec_coord[(m/8)*3]),
                                                            _mm256_sub_ps(_mm256_set1_ps(y0+k*dy), vect_rec_coord[(m/8)*3+1]),
                                                            _mm256_sub_ps(_mm256_set1_ps(z0+i*dz), vect_rec_coord[(m/8)*3+2])),
                                                            _mm256_mul_ps(_mm256_set1_ps(vv), _mm256_set1_ps(dt))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)));
                            float temp_ind[8];
                            _mm256_store_ps(temp_ind, vect_ind);
                            for (size_t v = 0; v < 8; ++v) {
                                ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m+v] = temp_ind[v];
                                if (m != 0 || v != 0) {
                                    min_ind_arr[i*nx*ny+j*ny+k] = std::min(min_ind_arr[i*nx*ny+j*ny+k],
                                                                       ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m+v]);
                                } else {
                                    min_ind_arr[i*nx*ny+j*ny+k] = ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count];    
                                }
                            }
                        } else {
                            for (size_t n_m = m; n_m < rec_count; ++n_m) {
                                ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+n_m] = round(calc_radius((x0+j*dx)-rec_coords[n_m*3],
                                                                                                    (y0+k*dy)-rec_coords[n_m*3+1],
                                                                                                    (z0+i*dz)-rec_coords[n_m*3+2])
                                                                                                    /(vv*dt)) + 1;
                                if (0 != m) {
                                    min_ind_arr[i*nx*ny+j*ny+k] = std::min(min_ind_arr[i*nx*ny+j*ny+k],
                                                                           ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+n_m]);
                                } else {
                                    min_ind_arr[i*nx*ny+j*ny+k] = ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count];
                                }        
                            }   
                        }

                    }
                }
            }
        }

        #pragma omp for collapse(2)
    	for (size_t c_t = 0; c_t < times; c_t += times_block_size) {
            for (size_t c_r = 0; c_r < rec_count; c_r += rec_block_size) {
        		for (size_t i = 0; i < nz; ++i) {
                    for (size_t j = 0; j < nx; ++j) {
                        for (size_t k = 0; k < ny; ++k) {
                            size_t min_ind = min_ind_arr[i*nx*ny+j*ny+k];
                            for (size_t m = c_r; m < std::min(c_r+rec_block_size, rec_count); ++m) {
                                size_t ind = ind_arr[i*nx*ny*rec_count+j*ny*rec_count+k*rec_count+m]-min_ind;
                                for (size_t l = c_t; l < std::min(c_t+times_block_size, times-ind); ++l) {
                                    // for (size_t m = 0; m < rec_count; ++m) {
                                    //  std::cout << rec_times[m*times+ind_arr[m]+l-min_ind] << std::endl;
                                    // }
                                    // return 0;
                                    area_discr[i*nx*ny*times+j*ny*times+k*times+l] += rec_times[m*times+ind+l];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    //#pragma omp collapse с выносом цикла по x наверх
    //распараллелить по c_r с использованием локальных массивов
    //******************************************************//
    t2 = omp_get_wtime();

    // std::ifstream results_file;
    // results_file.open("../Summation_Results2.bin", std::ios::binary);
    // if (!results_file.is_open()) {
    //     std::cerr << "Can't open Summation_Results.bin" << std::endl;
    //     return 1;
    // }

    // std::unique_ptr<float[]> real_results{new float[nx*ny*nz*times]};
    // results_file.read(reinterpret_cast<char*>(real_results.get()), nx*ny*nz*times*sizeof(float));

    // float result = 0;
    // float temp1 = 0, temp2 = 0;
    // for (size_t i = 0; i < nz; ++i) {
    //     for (size_t j = 0; j < nx; ++j) {
    //         for (size_t k = 0; k < ny; ++k) {
    //             for (size_t l = 0; l < times; ++l) {
    //                 temp1 += (real_results[i*nx*ny*times+j*ny*times+k*times+l]-area_discr[i*nx*ny*times+j*ny*times+k*times+l])*
    //                          (real_results[i*nx*ny*times+j*ny*times+k*times+l]-area_discr[i*nx*ny*times+j*ny*times+k*times+l]);
    //                 // std::cout << real_results[i*nx*ny*times+j*ny*times+k*times+l] << " " << area_discr[i*nx*ny*times+j*ny*times+k*times+l] << std::endl;
    //                 temp2 += real_results[i*nx*ny*times+j*ny*times+k*times+l]*real_results[i*nx*ny*times+j*ny*times+k*times+l];
    //             }
    //         }
    //     }
    // }
    // result = sqrt(temp1)/sqrt(temp2);

    // std::cout << "Result == " << result << std::endl;

    std::cout << "Time: " << t2-t1 << std::endl;

    return 0;
}