#!/bin/sh
#first arg - n_rec, second arg - n_srcs
mkdir results_times
export OMP_NUM_THREADS=1

g++ -O3 -pthread -march=native -mavx2 -std=c++11 -fopenmp vector_time_test.cpp -DVECT_NO
./a.out 10000 20000 >> tests.txt
# ./a.out 10000 40000 >> results_times/no_vect_times.txt
# ./a.out 10000 80000 >> results_times/no_vect_times.txt
# ./a.out 10000 120000 >> results_times/no_vect_times.txt
# ./a.out 10000 160000 >> results_times/no_vect_times.txt
# ./a.out 10000 190000 >> results_times/no_vect_times.txt
# ./a.out 20000 40000 >> results_times/no_vect_times.txt
# ./a.out 20000 80000 >> results_times/no_vect_times.txt
# ./a.out 20000 120000 >> results_times/no_vect_times.txt
# ./a.out 20000 160000 >> results_times/no_vect_times.txt
# ./a.out 20000 190000 >> results_times/no_vect_times.txt

g++ -O3 -pthread -march=native -msse2 -std=c++11 -fopenmp vector_time_test.cpp -DVECT_128
./a.out 10000 20000 >> tests.txt
# ./a.out 10000 40000 >> results_times/128_vect_times.txt
# ./a.out 10000 80000 >> results_times/128_vect_times.txt
# ./a.out 10000 120000 >> results_times/128_vect_times.txt
# ./a.out 10000 160000 >> results_times/128_vect_times.txt
# ./a.out 10000 190000 >> results_times/128_vect_times.txt
# ./a.out 20000 40000 >> results_times/128_vect_times.txt
# ./a.out 20000 80000 >> results_times/128_vect_times.txt
# ./a.out 20000 120000 >> results_times/128_vect_times.txt
# ./a.out 20000 160000 >> results_times/128_vect_times.txt
# ./a.out 20000 190000 >> results_times/128_vect_times.txt

g++ -O3 -pthread -march=native -mavx2 -std=c++11 -fopenmp vector_time_test.cpp -DVECT_256
./a.out 10000 20000 >> tests.txt
# ./a.out 10000 40000 >> results_times/256_vect_times.txt
# ./a.out 10000 80000 >> results_times/256_vect_times.txt
# ./a.out 10000 120000 >> results_times/256_vect_times.txt
# ./a.out 10000 160000 >> results_times/256_vect_times.txt
# ./a.out 10000 190000 >> results_times/256_vect_times.txt
# ./a.out 20000 40000 >> results_times/256_vect_times.txt
# ./a.out 20000 80000 >> results_times/256_vect_times.txt
# ./a.out 20000 120000 >> results_times/256_vect_times.txt
# ./a.out 20000 160000 >> results_times/256_vect_times.txt
# ./a.out 20000 190000 >> results_times/256_vect_times.txt

# g++ -O3 -pthread -mavx512dq -mavx512f -std=c++11 -fopenmp vector_time_test.cpp -DVECT_512
# ./a.out 10000 20000 >> tests.txt

