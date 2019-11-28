#define EIGEN_NO_DEBUG

#include "coherent_summation.h"

#include <Eigen/Dense>
#include <cstdint>
#include <numpy/arrayobject.h>
#include <Python.h> 

static PyObject *CompError;

using namespace Eigen;

template <typename T>
using Array2D = Matrix<T, Dynamic, Dynamic, RowMajor>;
template <typename T>
using MapArray2D = Map<const Array2D<T>, Aligned>;

template <typename T>
using ResArrType = Map<Array2D<T>, Aligned>;

static PyObject* compute_coherent_summation(PyObject* self, PyObject* args) {
	PyObject *arg1 = nullptr, *arg2 = nullptr, *arg3 = nullptr;

	if (!PyArg_ParseTuple(args, "OOO", &arg1, &arg2, &arg3)) {
		return nullptr;
	}

	PyObject *rec_samples = nullptr, *amplitudes = nullptr, *sources_times = nullptr;
	rec_samples = PyArray_FROM_OTF(arg1, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
	if (rec_samples == nullptr) {
		return nullptr;
	}
	amplitudes = PyArray_FROM_OTF(arg2, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
	if (amplitudes == nullptr) {
		return nullptr;
	}
	sources_times = PyArray_FROM_OTF(arg3, NPY_INT64, NPY_ARRAY_IN_ARRAY);
	if (sources_times == nullptr) {
		return nullptr;
	}

	const npy_intp n_rec = PyArray_DIMS(rec_samples)[0];
	const npy_intp n_samples = PyArray_DIMS(rec_samples)[1];
	const npy_intp sources_count = PyArray_DIMS(sources_times)[0];

	if (n_rec != PyArray_DIMS(sources_times)[1]) {
		PyErr_SetString(CompError, "The number of receivers in the array of samples and in the array of coordinates does not match");
		return nullptr;
	}

	if (PyArray_DIMS(amplitudes)[1] != n_rec) {
		PyErr_SetString(CompError, "Incorrect area coordinates data");
		return nullptr;
	}
	if (PyArray_DIMS(amplitudes)[0] != sources_count) {
		PyErr_SetString(CompError, "Incorrect count of sources");
		return nullptr;
	} 

	double *rec_samples_data = (double*)PyArray_DATA(rec_samples);
	double *amplitudes_data = (double*)PyArray_DATA(amplitudes);
	int64_t *sources_times_data = (int64_t*)PyArray_DATA(sources_times);

	MapArray2D<double> rec_samples_data2D{rec_samples_data, n_rec, n_samples};
	MapArray2D<double> amplitudes_data2D{amplitudes_data, sources_count, n_rec};
	MapArray2D<int64_t> sources_times_data2D{sources_times_data, sources_count, n_rec};

	npy_intp result_dims[2] = {sources_count, n_samples};
	PyObject *result_arr = PyArray_ZEROS(2, result_dims, NPY_FLOAT64, 0);

	if (result_arr == nullptr) {
		return nullptr;
	}

	double* result_arr_data = (double*)PyArray_DATA(result_arr);

	ResArrType<double> result_arr_data2D{result_arr_data, sources_count, n_samples};

	compute(rec_samples_data2D, amplitudes_data2D, sources_times_data2D, result_arr_data2D);

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