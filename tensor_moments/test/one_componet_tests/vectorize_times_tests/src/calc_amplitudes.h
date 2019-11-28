#ifndef _CALC_AMPLITUDES
#define _CALC_AMPLITUDES

#include "array3D.h"
#include "array2D.h"

#include <x86intrin.h>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <iostream>

/*Define functions for calculate amplitudes*/
template <typename T>
T vect_calc_norm(T x, T y, T z);

template <typename T>
inline T calc_norm(T x, T y, T z) {
    return sqrt(x*x+y*y+z*z)+1e-36;
}

template <typename T, size_t N>
inline void vect_sgn(T *val);

template <typename T, size_t N>
inline void transpose_green_tensor(const T* G_P_trans, ptrdiff_t G_P_size, ptrdiff_t vector_dim, T* G_P);

template <typename T>
void non_vector_calc_amplitudes(ptrdiff_t ind_first_rec, const Array2D<T> &sources_coords, const Array2D<T> &rec_coords, const Array2D<T> &tensor_matrix, Array3D<T> &amplitudes) {
    ptrdiff_t n_rec = rec_coords.get_y_dim();
    ptrdiff_t sources_count = sources_coords.get_y_dim();
    ptrdiff_t matrix_dim = tensor_matrix.get_x_dim();
    ptrdiff_t ampl_dim = amplitudes.get_x_dim();

    #pragma omp parallel
    {
        T coord_vect[3];
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = ind_first_rec; r_ind < n_rec; ++r_ind) {
                for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                    coord_vect[crd] = rec_coords(r_ind, crd)-sources_coords(i, crd);
                }
                T norm = calc_norm(coord_vect[0], coord_vect[1], coord_vect[2]);
                coord_vect[0] /= (norm*norm);
                coord_vect[1] /= (norm*norm);
                coord_vect[2] /= (norm*norm);

                for (ptrdiff_t ampl_i = 0; ampl_i < ampl_dim; ++ampl_i) {
                    for (ptrdiff_t n = 0; n < matrix_dim; ++n) {
                        for (ptrdiff_t m = 0; m < matrix_dim; ++m) {
                            amplitudes(i, r_ind, ampl_i) += coord_vect[ampl_i]*coord_vect[n]*coord_vect[m]*tensor_matrix(n, m);
                        }
                    }
                    amplitudes(i, r_ind, ampl_i) /= (fabs(amplitudes(i, r_ind, ampl_i))+1e-36);
                }
            }
        }         
    }
}

template <typename T, size_t N>
void instrinsic_calc_amplitudes(const Array2D<T> &sources_coords, const Array2D<T> &rec_coords, const Array2D<T> &tensor_matrix, Array3D<T> &amplitudes) {
    non_vector_calc_amplitudes<T>(0, sources_coords, rec_coords, tensor_matrix, amplitudes);
}

/*Selection of SIMD instructions*/
#if defined(__AVX512F__)
template <typename T> 
void calc_amplitudes(const Array2D<T> &src_crds, const Array2D<T> &rec_crds, const Array2D<T> &tnsr_mtrx, Array3D<T> &amplitudes) {
    instrinsic_calc_amplitudes<T, 512>(src_crds, rec_crds, tnsr_mtrx, amplitudes);
}

/*Implement functions for AVX512F*/
template <>
__m512d vect_calc_norm<__m512d>(__m512d x, __m512d y, __m512d z) {
    return _mm512_add_pd(_mm512_sqrt_pd(_mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(x, x), _mm512_mul_pd(y, y)), _mm512_mul_pd(z, z))), _mm512_set1_pd(1e-300));
}

template <>
inline void transpose_green_tensor<double, 512>(const double* G_trans, ptrdiff_t G_size, ptrdiff_t vector_dim, double* G) {
    for (ptrdiff_t g_ind = 0; g_ind < G_size; ++g_ind) {
        __m512d new_row = _mm512_set_pd(G_trans[((vector_dim*(g_ind*vector_dim+7))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+7))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+6))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+6))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+5))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+5))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+4))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+4))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+3))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+3))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+2))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+2))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+1))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+1))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+0))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+0))/(G_size*vector_dim)]);
        _mm512_storeu_pd(G+g_ind*vector_dim, new_row);    
    }
}

template <>
inline void vect_sgn<double, 512>(double *arr) {
    ptrdiff_t vector_dim = sizeof(__m512d)/sizeof(double);
    double tmp[vector_dim];
    for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {  
        tmp[v_s] = (fabs(arr[v_s])+1e-300);
    }
    _mm512_storeu_pd(arr, _mm512_div_pd(_mm512_loadu_pd(arr), _mm512_loadu_pd(tmp)));
}

template <>
__m512 vect_calc_norm<__m512>(__m512 x, __m512 y, __m512 z) {
    return _mm512_add_ps(_mm512_sqrt_ps(_mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(x, x), _mm512_mul_ps(y, y)), _mm512_mul_ps(z, z))), _mm512_set1_ps(1e-36));
}

template <>
inline void transpose_green_tensor<float, 512>(const float* G_trans, ptrdiff_t G_size, ptrdiff_t vector_dim, float* G) {
    for (ptrdiff_t g_ind = 0; g_ind < G_size; ++g_ind) {
        __m512 new_row = _mm512_set_ps(G_trans[((vector_dim*(g_ind*vector_dim+15))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+15))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+14))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+14))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+13))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+13))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+12))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+12))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+11))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+11))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+10))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+10))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+9))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+9))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+8))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+8))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+7))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+7))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+6))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+6))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+5))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+5))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+4))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+4))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+3))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+3))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+2))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+2))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+1))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+1))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+0))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+0))/(G_size*vector_dim)]);
        _mm512_storeu_ps(G+g_ind*vector_dim, new_row);    
    }
}

template <>
inline void vect_sgn<float, 512>(float *arr) {
    ptrdiff_t vector_dim = sizeof(__m512)/sizeof(float);
    float tmp[vector_dim];
    for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {  
        tmp[v_s] = (fabs(arr[v_s])+1e-36);
    }
    _mm512_storeu_ps(arr, _mm512_div_ps(_mm512_loadu_ps(arr), _mm512_loadu_ps(tmp)));
}

template <>
void instrinsic_calc_amplitudes<double, 512>(const Array2D<double> &sources_coords, const Array2D<double> &rec_coords, const Array2D<double> &tensor_matrix,  Array3D<double> &amplitudes) {
    ptrdiff_t n_rec = rec_coords.get_y_dim();
    ptrdiff_t sources_count = sources_coords.get_y_dim();
    ptrdiff_t matrix_dim = tensor_matrix.get_x_dim();
    ptrdiff_t vector_dim = sizeof(__m512d)/sizeof(double);
    ptrdiff_t ampl_dim = amplitudes.get_x_dim();
    ptrdiff_t G_size = ampl_dim*matrix_dim*matrix_dim;
    ptrdiff_t G_vect_size = vector_dim*G_size;
    std::unique_ptr<__m512d[], decltype(free)*> vect_rec_coord{reinterpret_cast<__m512d*>(aligned_alloc(sizeof(__m512d), sizeof(__m512d)*(n_rec/vector_dim)*3)), free};

    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
            for (ptrdiff_t i = 0; i < 3; ++i) {
                vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm512_set_pd(rec_coords(r_ind+7, i), rec_coords(r_ind+6, i), rec_coords(r_ind+5, i), rec_coords(r_ind+4, i),
                                                                       rec_coords(r_ind+3, i), rec_coords(r_ind+2, i), rec_coords(r_ind+1, i), rec_coords(r_ind+0, i));
            }
        }

        __m512d coord_vect[3];
        double G_trans[G_vect_size];
        double G_data[G_vect_size];
        Array2D<double> G{G_data, G_vect_size, G_size};
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (ptrdiff_t crd = 0; crd < matrix_dim; ++crd) {
                    coord_vect[crd] = _mm512_sub_pd(vect_rec_coord[(r_ind/vector_dim)*3+crd], _mm512_set1_pd(sources_coords(i, crd)));
                }

                __m512d norm = vect_calc_norm(coord_vect[0], coord_vect[1], coord_vect[2]);

                for (ptrdiff_t crd = 0; crd < matrix_dim; ++crd) {
                    coord_vect[crd] = _mm512_div_pd(_mm512_div_pd(coord_vect[crd], norm), norm);
                }

                for (ptrdiff_t g_ind = 0; g_ind < G_size; ++g_ind) {
                    ptrdiff_t i_ind = g_ind/9;
                    ptrdiff_t j_ind = (g_ind/3)%3;
                    ptrdiff_t k_ind = g_ind%3;
                    _mm512_storeu_pd(G_trans+g_ind*vector_dim, _mm512_mul_pd(_mm512_mul_pd(coord_vect[i_ind], coord_vect[j_ind]), coord_vect[k_ind]));
                }

                transpose_green_tensor<double, 512>(G_trans, G_size, vector_dim, G_data);

                for (ptrdiff_t ampl_i = 0; ampl_i < ampl_dim*vector_dim; ++ampl_i) {
                    ptrdiff_t k = ampl_i%ampl_dim;
                    ptrdiff_t v_s = ampl_i/ampl_dim;
                    for (ptrdiff_t n = 0; n < matrix_dim; ++n) {
                        for (ptrdiff_t m = 0; m < matrix_dim; ++m) {
                            amplitudes(i, r_ind, ampl_i) += G(v_s, k*matrix_dim*matrix_dim+n*matrix_dim+n)*tensor_matrix(n, m);
                        }
                    }
                    amplitudes(i, r_ind, ampl_i) /= (fabs(amplitudes(i, r_ind, ampl_i))+1e-300);
                }
                // for (ptrdiff_t ampl_i = 0; ampl_i < ampl_dim*vector_dim; ampl_i+=vector_dim) {
                //     vect_sgn<double, 512>(&amplitudes(i, r_ind, ampl_i));
                // } 
            }
        }         
    }
    non_vector_calc_amplitudes<double>(n_rec-(n_rec%vector_dim), sources_coords, rec_coords, tensor_matrix, amplitudes);
}

template <>
void instrinsic_calc_amplitudes<float, 512>(const Array2D<float> &sources_coords, const Array2D<float> &rec_coords, const Array2D<float> &tensor_matrix, Array3D<float> &amplitudes) {
    ptrdiff_t n_rec = rec_coords.get_y_dim();
    ptrdiff_t sources_count = sources_coords.get_y_dim();
    ptrdiff_t matrix_dim = tensor_matrix.get_x_dim();
    ptrdiff_t vector_dim = sizeof(__m512)/sizeof(float);
    ptrdiff_t ampl_dim = amplitudes.get_x_dim();
    ptrdiff_t G_size = ampl_dim*matrix_dim*matrix_dim;
    ptrdiff_t G_vect_size = vector_dim*G_size;
    std::unique_ptr<__m512[], decltype(free)*> vect_rec_coord{reinterpret_cast<__m512*>(aligned_alloc(sizeof(__m512), sizeof(__m512)*(n_rec/vector_dim)*3)), free};

    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
            for (ptrdiff_t i = 0; i < 3; ++i) {
                vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm512_set_ps(rec_coords(r_ind+15, i), rec_coords(r_ind+14, i), rec_coords(r_ind+13, i), rec_coords(r_ind+12, i),
                                                                       rec_coords(r_ind+11, i), rec_coords(r_ind+10, i), rec_coords(r_ind+9, i), rec_coords(r_ind+8, i),
                                                                       rec_coords(r_ind+7, i), rec_coords(r_ind+6, i), rec_coords(r_ind+5, i), rec_coords(r_ind+4, i),
                                                                       rec_coords(r_ind+3, i), rec_coords(r_ind+2, i), rec_coords(r_ind+1, i), rec_coords(r_ind+0, i));
            }
        }

        __m512 coord_vect[3];
        float G_trans[G_vect_size];
        float G_data[G_vect_size];
        Array2D<float> G{G_data, G_vect_size, G_size};
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (ptrdiff_t crd = 0; crd < matrix_dim; ++crd) {
                    coord_vect[crd] = _mm512_sub_ps(vect_rec_coord[(r_ind/vector_dim)*3+crd], _mm512_set1_ps(sources_coords(i, crd)));
                }

                __m512 norm = vect_calc_norm(coord_vect[0], coord_vect[1], coord_vect[2]);

                for (ptrdiff_t crd = 0; crd < matrix_dim; ++crd) {
                    coord_vect[crd] = _mm512_div_ps(_mm512_div_ps(coord_vect[crd], norm), norm);
                }

                for (ptrdiff_t g_ind = 0; g_ind < G_size; ++g_ind) {
                    ptrdiff_t i_ind = g_ind/9;
                    ptrdiff_t j_ind = (g_ind/3)%3;
                    ptrdiff_t k_ind = g_ind%3;
                    _mm512_storeu_ps(G_trans+g_ind*vector_dim, _mm512_mul_ps(_mm512_mul_ps(coord_vect[i_ind], coord_vect[j_ind]), coord_vect[k_ind]));
                }

                transpose_green_tensor<float, 512>(G_trans, G_size, vector_dim, G_data);

                for (ptrdiff_t ampl_i = 0; ampl_i < ampl_dim*vector_dim; ++ampl_i) {
                    ptrdiff_t k = ampl_i%ampl_dim;
                    ptrdiff_t v_s = ampl_i/ampl_dim;
                    for (ptrdiff_t n = 0; n < matrix_dim; ++n) {
                        for (ptrdiff_t m = 0; m < matrix_dim; ++m) {
                            amplitudes(i, r_ind, ampl_i) += G(v_s, k*matrix_dim*matrix_dim+n*matrix_dim+n)*tensor_matrix(n, m);
                        }
                    }
                    amplitudes(i, r_ind, ampl_i) /= (fabs(amplitudes(i, r_ind, ampl_i))+1e-36);
                }
                // for (ptrdiff_t ampl_i = 0; ampl_i < ampl_dim*vector_dim; ampl_i+=vector_dim) {
                //     vect_sgn<float, 512>(&amplitudes(i, r_ind, ampl_i));
                // } 
            }
        }     
    }
    non_vector_calc_amplitudes<float>(n_rec-(n_rec%vector_dim), sources_coords, rec_coords, tensor_matrix, amplitudes);
}

/*__AVX512F__*/

#elif defined(__AVX__)
template <typename T> 
void calc_amplitudes(const Array2D<T> &src_crds, const Array2D<T> &rec_crds, const Array2D<T> &tnsr_mtrx, Array3D<T> &amplitudes) {
    printf("avx\n");
    instrinsic_calc_amplitudes<T, 256>(src_crds, rec_crds, tnsr_mtrx, amplitudes);
}

/*Implement functions for AVX*/

template <>
__m256d vect_calc_norm<__m256d>(__m256d x, __m256d y, __m256d z) {
    return _mm256_add_pd(_mm256_sqrt_pd(_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(x, x), _mm256_mul_pd(y, y)), _mm256_mul_pd(z, z))), _mm256_set1_pd(1e-300));
}

template <>
inline void transpose_green_tensor<double, 256>(const double* G_trans, ptrdiff_t G_size, ptrdiff_t vector_dim, double* G) {
    for (ptrdiff_t g_ind = 0; g_ind < G_size; ++g_ind) {
        __m256d new_row = _mm256_set_pd(G_trans[((vector_dim*(g_ind*vector_dim+3))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+3))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+2))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+2))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+1))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+1))/(G_size*vector_dim)],
                                        G_trans[((vector_dim*(g_ind*vector_dim+0))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+0))/(G_size*vector_dim)]);
        _mm256_storeu_pd(G+g_ind*vector_dim, new_row);    
    }
}

template <>
inline void vect_sgn<double, 256>(double *arr) {
    ptrdiff_t vector_dim = sizeof(__m256d)/sizeof(double);
    double tmp[vector_dim];
    for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {  
        tmp[v_s] = (fabs(arr[v_s])+1e-300);
    }
    _mm256_storeu_pd(arr, _mm256_div_pd(_mm256_loadu_pd(arr), _mm256_loadu_pd(tmp)));
}

template <>
__m256 vect_calc_norm<__m256>(__m256 x, __m256 y, __m256 z) {
    return _mm256_add_ps(_mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y)), _mm256_mul_ps(z, z))), _mm256_set1_ps(1e-36));
}

template <>
inline void transpose_green_tensor<float, 256>(const float* G_trans, ptrdiff_t G_size, ptrdiff_t vector_dim, float* G) {
    for (ptrdiff_t g_ind = 0; g_ind < G_size; ++g_ind) {
        __m256 new_row = _mm256_set_ps(G_trans[((vector_dim*(g_ind*vector_dim+7))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+7))/(G_size*vector_dim)],
                                       G_trans[((vector_dim*(g_ind*vector_dim+6))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+6))/(G_size*vector_dim)],
                                       G_trans[((vector_dim*(g_ind*vector_dim+5))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+5))/(G_size*vector_dim)],
                                       G_trans[((vector_dim*(g_ind*vector_dim+4))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+4))/(G_size*vector_dim)],
                                       G_trans[((vector_dim*(g_ind*vector_dim+3))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+3))/(G_size*vector_dim)],
                                       G_trans[((vector_dim*(g_ind*vector_dim+2))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+2))/(G_size*vector_dim)],
                                       G_trans[((vector_dim*(g_ind*vector_dim+1))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+1))/(G_size*vector_dim)],
                                       G_trans[((vector_dim*(g_ind*vector_dim+0))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+0))/(G_size*vector_dim)]);
        _mm256_storeu_ps(G+g_ind*vector_dim, new_row);    
    }
}

template <>
inline void vect_sgn<float, 256>(float *arr) {
    ptrdiff_t vector_dim = sizeof(__m256)/sizeof(float);
    float tmp[vector_dim];
    for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {  
        tmp[v_s] = (fabs(arr[v_s])+1e-36);
    }
    _mm256_storeu_ps(arr, _mm256_div_ps(_mm256_loadu_ps(arr), _mm256_loadu_ps(tmp)));
}

template <>
void instrinsic_calc_amplitudes<double, 256>(const Array2D<double> &sources_coords, const Array2D<double> &rec_coords, const Array2D<double> &tensor_matrix,  Array3D<double> &amplitudes) {
    ptrdiff_t n_rec = rec_coords.get_y_dim();
    ptrdiff_t sources_count = sources_coords.get_y_dim();
    ptrdiff_t matrix_dim = tensor_matrix.get_x_dim();
    ptrdiff_t vector_dim = sizeof(__m256d)/sizeof(double);
    ptrdiff_t ampl_dim = amplitudes.get_x_dim(); 
    ptrdiff_t G_size = ampl_dim*matrix_dim*matrix_dim;
    ptrdiff_t G_vect_size = vector_dim*G_size;
    std::unique_ptr<__m256d[], decltype(free)*> vect_rec_coord{reinterpret_cast<__m256d*>(aligned_alloc(sizeof(__m256d), sizeof(__m256d)*(n_rec/vector_dim)*3)), free};

    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
            for (ptrdiff_t crd = 0; crd < 3; ++crd) {
                vect_rec_coord[(r_ind/vector_dim)*3+crd] = _mm256_set_pd(rec_coords(r_ind+3, crd), rec_coords(r_ind+2, crd), rec_coords(r_ind+1, crd), rec_coords(r_ind+0, crd));
            }
        }

        __m256d coord_vect[3];
        double G_trans[G_vect_size];
        double G_data[G_vect_size];
        Array2D<double> G{G_data, G_vect_size, G_size};
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (ptrdiff_t crd = 0; crd < matrix_dim; ++crd) {
                    coord_vect[crd] = _mm256_sub_pd(vect_rec_coord[(r_ind/vector_dim)*3+crd], _mm256_set1_pd(sources_coords(i, crd)));
                }

                __m256d norm = vect_calc_norm(coord_vect[0], coord_vect[1], coord_vect[2]);

                for (ptrdiff_t crd = 0; crd < matrix_dim; ++crd) {
                    coord_vect[crd] = _mm256_div_pd(_mm256_div_pd(coord_vect[crd], norm), norm);
                }

                for (ptrdiff_t g_ind = 0; g_ind < G_size; ++g_ind) {
                    ptrdiff_t i_ind = g_ind/9;
                    ptrdiff_t j_ind = (g_ind/3)%3;
                    ptrdiff_t k_ind = g_ind%3;
                    _mm256_storeu_pd(G_trans+g_ind*vector_dim, _mm256_mul_pd(_mm256_mul_pd(coord_vect[i_ind], coord_vect[j_ind]), coord_vect[k_ind]));
                }

                transpose_green_tensor<double, 256>(G_trans, G_size, vector_dim, G_data);

                for (ptrdiff_t ampl_i = 0; ampl_i < ampl_dim*vector_dim; ++ampl_i) {
                    ptrdiff_t k = ampl_i%ampl_dim;
                    ptrdiff_t v_s = ampl_i/ampl_dim;
                    for (ptrdiff_t n = 0; n < matrix_dim; ++n) {
                        for (ptrdiff_t m = 0; m < matrix_dim; ++m) {
                            amplitudes(i, r_ind, ampl_i) += G(v_s, k*matrix_dim*matrix_dim+n*matrix_dim+n)*tensor_matrix(n, m);
                        }
                    }
                    amplitudes(i, r_ind, ampl_i) /= (fabs(amplitudes(i, r_ind, ampl_i))+1e-300);
                }
                // for (ptrdiff_t ampl_i = 0; ampl_i < ampl_dim*vector_dim; ampl_i+=vector_dim) {
                //     vect_sgn<double, 256>(&amplitudes(i, r_ind, ampl_i));
                // } 
            }
        }         
    }
    non_vector_calc_amplitudes<double>(n_rec-(n_rec%vector_dim), sources_coords, rec_coords, tensor_matrix, amplitudes);
}

template <>
void instrinsic_calc_amplitudes<float, 256>(const Array2D<float> &sources_coords, const Array2D<float> &rec_coords, const Array2D<float> &tensor_matrix, Array3D<float> &amplitudes) {
    ptrdiff_t n_rec = rec_coords.get_y_dim();
    ptrdiff_t sources_count = sources_coords.get_y_dim();
    ptrdiff_t matrix_dim = tensor_matrix.get_x_dim();
    ptrdiff_t vector_dim = sizeof(__m256)/sizeof(float);
    ptrdiff_t ampl_dim = amplitudes.get_x_dim();
    ptrdiff_t G_size = ampl_dim*matrix_dim*matrix_dim;
    ptrdiff_t G_vect_size = vector_dim*G_size;
    std::unique_ptr<__m256[], decltype(free)*> vect_rec_coord{reinterpret_cast<__m256*>(aligned_alloc(sizeof(__m256), sizeof(__m256)*(n_rec/vector_dim)*3)), free};

    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
            for (ptrdiff_t i = 0; i < 3; ++i) {
                vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm256_set_ps(rec_coords(r_ind+7, i), rec_coords(r_ind+6, i), rec_coords(r_ind+5, i), rec_coords(r_ind+4, i),
                                                                       rec_coords(r_ind+3, i), rec_coords(r_ind+2, i), rec_coords(r_ind+1, i), rec_coords(r_ind+0, i));
            }
        }

        __m256 coord_vect[3];
        float G_trans[G_vect_size];
        float G_data[G_vect_size];
        Array2D<float> G{G_data, G_vect_size, G_size};
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (ptrdiff_t crd = 0; crd < matrix_dim; ++crd) {
                    coord_vect[crd] = _mm256_sub_ps(vect_rec_coord[(r_ind/vector_dim)*3+crd], _mm256_set1_ps(sources_coords(i, crd)));
                }

                __m256 norm = vect_calc_norm(coord_vect[0], coord_vect[1], coord_vect[2]);

                for (ptrdiff_t crd = 0; crd < matrix_dim; ++crd) {
                    coord_vect[crd] = _mm256_div_ps(_mm256_div_ps(coord_vect[crd], norm), norm);
                }

                for (ptrdiff_t g_ind = 0; g_ind < G_size; ++g_ind) {
                    ptrdiff_t i_ind = g_ind/9;
                    ptrdiff_t j_ind = (g_ind/3)%3;
                    ptrdiff_t k_ind = g_ind%3;
                    _mm256_storeu_ps(G_trans+g_ind*vector_dim, _mm256_mul_ps(_mm256_mul_ps(coord_vect[i_ind], coord_vect[j_ind]), coord_vect[k_ind]));
                }

                transpose_green_tensor<float, 256>(G_trans, G_size, vector_dim, G_data);

                for (ptrdiff_t ampl_i = 0; ampl_i < ampl_dim*vector_dim; ++ampl_i) {
                    ptrdiff_t k = ampl_i%ampl_dim;
                    ptrdiff_t v_s = ampl_i/ampl_dim;
                    for (ptrdiff_t n = 0; n < matrix_dim; ++n) {
                        for (ptrdiff_t m = 0; m < matrix_dim; ++m) {
                            amplitudes(i, r_ind, ampl_i) += G(v_s, k*matrix_dim*matrix_dim+n*matrix_dim+n)*tensor_matrix(n, m);
                        }
                    }
                    amplitudes(i, r_ind, ampl_i) /= (fabs(amplitudes(i, r_ind, ampl_i))+1e-36);
                }
                // for (ptrdiff_t ampl_i = 0; ampl_i < ampl_dim*vector_dim; ampl_i+=vector_dim) {
                //     vect_sgn<float, 256>(&amplitudes(i, r_ind, ampl_i));
                // } 
            }
        }
    }
    non_vector_calc_amplitudes<float>(n_rec-(n_rec%vector_dim), sources_coords, rec_coords, tensor_matrix, amplitudes);
}

/*__AVX__*/

#elif defined(__SSE2__)
template <typename T> 
void calc_amplitudes(const Array2D<T> &src_crds, const Array2D<T> &rec_crds, const Array2D<T> &tnsr_mtrx, Array3D<T> &amplitudes) {    
    printf("sse2\n");
    instrinsic_calc_amplitudes<T, 128>(src_crds, rec_crds, tnsr_mtrx, amplitudes);
}

/*Implement functions for SSE2*/

template <>
__m128d vect_calc_norm<__m128d>(__m128d x, __m128d y, __m128d z) {
    return _mm_add_pd(_mm_sqrt_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y)), _mm_mul_pd(z, z))), _mm_set1_pd(1e-300));
}

template <>
inline void transpose_green_tensor<double, 128>(const double* G_trans, ptrdiff_t G_size, ptrdiff_t vector_dim, double* G) {
    for (ptrdiff_t g_ind = 0; g_ind < G_size; ++g_ind) {
        __m128d new_row = _mm_set_pd(G_trans[((vector_dim*(g_ind*vector_dim+1))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+1))/(G_size*vector_dim)],
                                     G_trans[((vector_dim*(g_ind*vector_dim+0))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+0))/(G_size*vector_dim)]);
        _mm_storeu_pd(G+g_ind*vector_dim, new_row);    
    }
}

template <>
inline void vect_sgn<double, 128>(double *arr) {
    ptrdiff_t vector_dim = sizeof(__m128d)/sizeof(double);
    double tmp[vector_dim];
    for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {  
        tmp[v_s] = (fabs(arr[v_s])+1e-300);
    }
    _mm_storeu_pd(arr, _mm_div_pd(_mm_loadu_pd(arr), _mm_loadu_pd(tmp)));
}

template <>
__m128 vect_calc_norm<__m128>(__m128 x, __m128 y, __m128 z) {
    return _mm_add_ps(_mm_sqrt_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(x, x), _mm_mul_ps(y, y)), _mm_mul_ps(z, z))), _mm_set1_ps(1e-36));
}

template <>
inline void transpose_green_tensor<float, 128>(const float* G_trans, ptrdiff_t G_size, ptrdiff_t vector_dim, float* G) {
    for (ptrdiff_t g_ind = 0; g_ind < G_size; ++g_ind) {
        __m128 new_row = _mm_set_ps(G_trans[((vector_dim*(g_ind*vector_dim+3))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+3))/(G_size*vector_dim)],
                                    G_trans[((vector_dim*(g_ind*vector_dim+2))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+2))/(G_size*vector_dim)],
                                    G_trans[((vector_dim*(g_ind*vector_dim+1))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+1))/(G_size*vector_dim)],
                                    G_trans[((vector_dim*(g_ind*vector_dim+0))%(G_size*vector_dim))+(vector_dim*(g_ind*vector_dim+0))/(G_size*vector_dim)]);
        _mm_storeu_ps(G+g_ind*vector_dim, new_row);    
    }
}

template <>
inline void vect_sgn<float, 128>(float *arr) {
    ptrdiff_t vector_dim = sizeof(__m128)/sizeof(float);
    float tmp[vector_dim];
    for (ptrdiff_t v_s = 0; v_s < vector_dim; ++v_s) {  
        tmp[v_s] = (fabs(arr[v_s])+1e-36);
    }
    _mm_storeu_ps(arr, _mm_div_ps(_mm_loadu_ps(arr), _mm_loadu_ps(tmp)));
}

template <>
void instrinsic_calc_amplitudes<double, 128>(const Array2D<double> &sources_coords, const Array2D<double> &rec_coords, const Array2D<double> &tensor_matrix, Array3D<double> &amplitudes) {
    ptrdiff_t n_rec = rec_coords.get_y_dim();
    ptrdiff_t sources_count = sources_coords.get_y_dim();
    ptrdiff_t matrix_dim = tensor_matrix.get_x_dim();
    ptrdiff_t vector_dim = sizeof(__m128d)/sizeof(double);
    ptrdiff_t ampl_dim = amplitudes.get_x_dim();
    ptrdiff_t G_size = ampl_dim*matrix_dim*matrix_dim;
    ptrdiff_t G_vect_size = vector_dim*G_size;
    std::unique_ptr<__m128d[], decltype(free)*> vect_rec_coord{reinterpret_cast<__m128d*>(aligned_alloc(sizeof(__m128d), sizeof(__m128d)*(n_rec/vector_dim)*3)), free};

    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
            for (ptrdiff_t i = 0; i < 3; ++i) {
                vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm_set_pd(rec_coords(r_ind+1, i), rec_coords(r_ind+0, i));
            }
        }

        __m128d coord_vect[3];
        double G_trans[G_vect_size];
        double G_data[G_vect_size];
        Array2D<double> G{G_data, G_vect_size, G_size};
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (ptrdiff_t crd = 0; crd < matrix_dim; ++crd) {
                    coord_vect[crd] = _mm_sub_pd(vect_rec_coord[(r_ind/vector_dim)*3+crd], _mm_set1_pd(sources_coords(i, crd)));
                }

                __m128d norm = vect_calc_norm(coord_vect[0], coord_vect[1], coord_vect[2]);

                for (ptrdiff_t crd = 0; crd < matrix_dim; ++crd) {
                    coord_vect[crd] = _mm_div_pd(_mm_div_pd(coord_vect[crd], norm), norm);
                }

                for (ptrdiff_t g_ind = 0; g_ind < G_size; ++g_ind) {
                    ptrdiff_t i_ind = g_ind/9;
                    ptrdiff_t j_ind = (g_ind/3)%3;
                    ptrdiff_t k_ind = g_ind%3;
                    _mm_storeu_pd(G_trans+g_ind*vector_dim, _mm_mul_pd(_mm_mul_pd(coord_vect[i_ind], coord_vect[j_ind]), coord_vect[k_ind]));
                }

                transpose_green_tensor<double, 128>(G_trans, G_size, vector_dim, G_data);

                for (ptrdiff_t ampl_i = 0; ampl_i < ampl_dim*vector_dim; ++ampl_i) {
                    ptrdiff_t k = ampl_i%ampl_dim;
                    ptrdiff_t v_s = ampl_i/ampl_dim;
                    for (ptrdiff_t n = 0; n < matrix_dim; ++n) {
                        for (ptrdiff_t m = 0; m < matrix_dim; ++m) {
                            amplitudes(i, r_ind, ampl_i) += G(v_s, k*matrix_dim*matrix_dim+n*matrix_dim+n)*tensor_matrix(n, m);
                        }
                    }
                    amplitudes(i, r_ind, ampl_i) /= (fabs(amplitudes(i, r_ind, ampl_i))+1e-300);
                }
                // for (ptrdiff_t ampl_i = 0; ampl_i < ampl_dim*vector_dim; ampl_i+=vector_dim) {
                //     vect_sgn<double, 128>(&amplitudes(i, r_ind, ampl_i));
                // } 
            }
        }         
    }
    non_vector_calc_amplitudes<double>(n_rec-(n_rec%vector_dim), sources_coords, rec_coords, tensor_matrix, amplitudes);
}

template <>
void instrinsic_calc_amplitudes<float, 128>(const Array2D<float> &sources_coords, const Array2D<float> &rec_coords, const Array2D<float> &tensor_matrix, Array3D<float> &amplitudes) {
    ptrdiff_t n_rec = rec_coords.get_y_dim();
    ptrdiff_t sources_count = sources_coords.get_y_dim();
    ptrdiff_t matrix_dim = tensor_matrix.get_x_dim();
    ptrdiff_t vector_dim = sizeof(__m128)/sizeof(float);
    ptrdiff_t ampl_dim = amplitudes.get_x_dim();
    ptrdiff_t G_size = ampl_dim*matrix_dim*matrix_dim;
    ptrdiff_t G_vect_size = vector_dim*G_size;
    std::unique_ptr<__m128[], decltype(free)*> vect_rec_coord{reinterpret_cast<__m128*>(aligned_alloc(sizeof(__m128), sizeof(__m128)*(n_rec/vector_dim)*3)), free};

    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
            for (ptrdiff_t i = 0; i < 3; ++i) {
                vect_rec_coord[(r_ind/vector_dim)*3+i] = _mm_set_ps(rec_coords(r_ind+3, i), rec_coords(r_ind+2, i), rec_coords(r_ind+1, i), rec_coords(r_ind+0, i));
            }
        }

       __m128 coord_vect[3];
        float G_trans[G_vect_size];
        float G_data[G_vect_size];
        Array2D<float> G{G_data, G_vect_size, G_size};
        #pragma omp for collapse(2)
        for (ptrdiff_t i = 0; i < sources_count; ++i) {
            for (ptrdiff_t r_ind = 0; r_ind < n_rec-(n_rec%vector_dim); r_ind+=vector_dim) {
                for (ptrdiff_t crd = 0; crd < matrix_dim; ++crd) {
                    coord_vect[crd] = _mm_sub_ps(vect_rec_coord[(r_ind/vector_dim)*3+crd], _mm_set1_ps(sources_coords(i, crd)));
                }

                __m128 norm = vect_calc_norm(coord_vect[0], coord_vect[1], coord_vect[2]);

                for (ptrdiff_t crd = 0; crd < matrix_dim; ++crd) {
                    coord_vect[crd] = _mm_div_ps(_mm_div_ps(coord_vect[crd], norm), norm);
                }

                for (ptrdiff_t g_ind = 0; g_ind < G_size; ++g_ind) {
                    ptrdiff_t i_ind = g_ind/9;
                    ptrdiff_t j_ind = (g_ind/3)%3;
                    ptrdiff_t k_ind = g_ind%3;
                    _mm_storeu_ps(G_trans+g_ind*vector_dim, _mm_mul_ps(_mm_mul_ps(coord_vect[i_ind], coord_vect[j_ind]), coord_vect[k_ind]));
                }

                transpose_green_tensor<float, 128>(G_trans, G_size, vector_dim, G_data);

                for (ptrdiff_t ampl_i = 0; ampl_i < ampl_dim*vector_dim; ++ampl_i) {
                    ptrdiff_t k = ampl_i%ampl_dim;
                    ptrdiff_t v_s = ampl_i/ampl_dim;
                    for (ptrdiff_t n = 0; n < matrix_dim; ++n) {
                        for (ptrdiff_t m = 0; m < matrix_dim; ++m) {
                            amplitudes(i, r_ind, ampl_i) += G(v_s, k*matrix_dim*matrix_dim+n*matrix_dim+n)*tensor_matrix(n, m);
                        }
                    }
                    amplitudes(i, r_ind, ampl_i) /= (fabs(amplitudes(i, r_ind, ampl_i))+1e-36);
                }
                // for (ptrdiff_t ampl_i = 0; ampl_i < ampl_dim*vector_dim; ampl_i+=vector_dim) {
                //     vect_sgn<float, 128>(&amplitudes(i, r_ind, ampl_i));
                // } 
            }
        }         
    }
    non_vector_calc_amplitudes<float>(n_rec-(n_rec%vector_dim), sources_coords, rec_coords, tensor_matrix, amplitudes);
}

/*__SSE2__*/

#else /*Without SIMD instructions*/

template <typename T>
void calc_amplitudes(const Array2D<T> &src_crds, const Array2D<T> &rec_crds, const Array2D<T> &tnsr_mtrx, Array3D<T> &amplitudes) {
    printf("no simd\n");
    instrinsic_calc_amplitudes<T, 0>(src_crds, rec_crds, tnsr_mtrx, amplitudes);
}

#endif /*End selection of SIMD instructions*/

#endif /*_CALC_AMPLITUDES*/