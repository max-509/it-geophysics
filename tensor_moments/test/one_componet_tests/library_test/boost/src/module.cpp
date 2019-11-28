#include "coherent_summation.h"

// #define BOOST_MULTI_ARRAY_NO_GENERATORS
#define BOOST_DISABLE_ASSERTS

#include <boost/multi_array.hpp>
#include <cstdint>
#include <numpy/arrayobject.h>
#include <Python.h> 

static PyObject *CompError;

template <typename T>
using Array1D = boost::multi_array_ref<T, 1>;
template <typename T>
using Array2D = boost::multi_array_ref<T, 2>;

static PyObject* compute_coherent_summation(PyObject* self, PyObject* args) {
	PyObject *arg1 = nullptr, *arg2 = nullptr, *arg3 = nullptr, *arg4 = nullptr, *arg5 = nullptr;

	if (!PyArg_ParseTuple(args, "OOOOO", &arg1, &arg2, &arg3, &arg4, &arg5)) {
		return nullptr;
	}

	PyObject *rec_samples = nullptr, *rec_coords = nullptr, *sources_coords = nullptr, *sources_times = nullptr, *tensor_matrix = nullptr;
	rec_samples = PyArray_FROM_OTF(arg1, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
	if (rec_samples == nullptr) {
		return nullptr;
	}
	rec_coords = PyArray_FROM_OTF(arg2, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
	if (rec_coords == nullptr) {
		return nullptr;
	}
	sources_coords = PyArray_FROM_OTF(arg3, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
	if (sources_coords == nullptr) {
		return nullptr;
	}
	sources_times = PyArray_FROM_OTF(arg4, NPY_INT64, NPY_ARRAY_IN_ARRAY);
	if (sources_times == nullptr) {
		return nullptr;
	}
	tensor_matrix = PyArray_FROM_OTF(arg5, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
	if (tensor_matrix == nullptr) {
		return nullptr;
	}

	const npy_intp n_rec = PyArray_DIMS(rec_samples)[0];
	const npy_intp n_samples = PyArray_DIMS(rec_samples)[1];
	const npy_intp sources_count = PyArray_DIMS(sources_times)[0];

	if (PyArray_DIMS(rec_coords)[1] != 3) {
		PyErr_SetString(CompError, "Error: Incorrect receivers coordinates data");
		return nullptr;
	}
	if (PyArray_DIMS(sources_coords)[1] != 3 || PyArray_DIMS(sources_coords)[0] != sources_count) {
		PyErr_SetString(CompError, "Error: Incorrect sources coordinates data");
		return nullptr;
	}
	if (n_rec != PyArray_DIMS(rec_coords)[0]) {
		PyErr_SetString(CompError, "Error: The number of receivers in the array of samples and in the array of coordinates does not match");
		return nullptr;
	}
	if (PyArray_DIMS(sources_times)[1] != n_rec) {
		PyErr_SetString(CompError, "Error: Incorrect count times");
		return nullptr;
	}
	if (PyArray_DIMS(tensor_matrix)[0] != 6) {
		PyErr_SetString(CompError, "Error: Incorrect tensor matrix");
		return nullptr;	
	}

	double *rec_samples_data = (double*)PyArray_DATA(rec_samples);
	double *rec_coords_data = (double*)PyArray_DATA(rec_coords);
	double *sources_coords_data = (double*)PyArray_DATA(sources_coords);
	int64_t *sources_times_data = (int64_t*)PyArray_DATA(sources_times);
	double *tensor_matrix_data = (double*)PyArray_DATA(tensor_matrix);

	Array2D<double> rec_samples_data2D{rec_samples_data, boost::extents[n_rec][n_samples], boost::c_storage_order()};
	Array2D<double> rec_coords_data2D{rec_coords_data, boost::extents[n_rec][3], boost::c_storage_order()};
	Array2D<double> sources_coords_data2D{sources_coords_data, boost::extents[sources_count][3], boost::c_storage_order()};
	Array2D<int64_t> sources_times_data2D{sources_times_data, boost::extents[sources_count][n_rec], boost::c_storage_order()};
	Array1D<double> tensor_matrix_data1D{tensor_matrix_data, boost::extents[6], boost::c_storage_order()};

	npy_intp result_dims[2] = {sources_count, n_samples};
	PyObject *result_arr = PyArray_ZEROS(2, result_dims, NPY_FLOAT64, 0);

	if (result_arr == nullptr) {
		return nullptr;
	}

	double* result_arr_data = (double*)PyArray_DATA(result_arr);

	compute(rec_samples_data2D, rec_coords_data2D, sources_coords_data2D, sources_times_data2D, tensor_matrix_data1D, result_arr_data);

	return result_arr;

}

static PyMethodDef methods[] = 
{
	{"computeCoherentSummation", compute_coherent_summation, METH_VARARGS, "Coherent Summation"},
	{nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef module = 
{
	PyModuleDef_HEAD_INIT,
	"coherentSummation",
	"Effective parallel coherent summation with vectorization",
	0,
	methods
};

PyMODINIT_FUNC 
PyInit_CoherentSumModule() {

	import_array();

	PyObject* m = nullptr;
	m = PyModule_Create(&module);
	if (m == nullptr) return nullptr;
	CompError = PyErr_NewException("compute.error", nullptr, nullptr);
	Py_INCREF(CompError);
	PyModule_AddObject(m, "error", CompError);

	return m;
}