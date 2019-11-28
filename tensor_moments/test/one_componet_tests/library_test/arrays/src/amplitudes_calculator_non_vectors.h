#ifndef _AMPLITUDES_CALCULATOR_NON_VECTORS
#define _AMPLITUDES_CALCULATOR_NON_VECTORS

#include "amplitudes_calculator.h"

template <typename T>
class AmplitudesCalculatorNonVectors : public AmplitudesCalculator<T, AmplitudesCalculatorNonVectors<T>> {
public:
	AmplitudesCalculatorNonVectors(const T *sources_coords,
						 	  const T *rec_coords,
						 	  const T *tensor_matrix,
                              ptrdiff_t sources_count,
                              ptrdiff_t n_rec,
						 	  T *amplitudes) : 
		sources_coords_(sources_coords),
		rec_coords_(rec_coords),
		tensor_matrix_(tensor_matrix),
		sources_count(sources_count),
        n_rec(n_rec),
		amplitudes_(amplitudes) 
	{ }

	friend AmplitudesCalculator<T, AmplitudesCalculatorNonVectors<T>>;

private:
	const T *sources_coords_;
    const T *rec_coords_;
    const T *tensor_matrix_;
    ptrdiff_t sources_count;
    ptrdiff_t n_rec;
    T *amplitudes_;	

	void realize_calculate() {
		this->non_vector_calculate_amplitudes(0, sources_coords_, rec_coords_, tensor_matrix_, sources_count, n_rec, amplitudes_);
	}
};

#endif /*_AMPLITUDES_CALCULATOR_NON_VECTORS*/