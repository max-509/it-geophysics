#ifndef _AMPLITUDES_CALCULATOR_NON_VECTORS
#define _AMPLITUDES_CALCULATOR_NON_VECTORS

// #define BOOST_MULTI_ARRAY_NO_GENERATORS
#define BOOST_DISABLE_ASSERTS

#include "amplitudes_calculator.h"

#include <boost/multi_array.hpp>
#include <cstddef>
#include <cmath>

template <typename T>
class AmplitudesCalculatorNonVectors : public AmplitudesCalculator<T, AmplitudesCalculatorNonVectors<T>> {
public:

	using Array1D = typename boost::multi_array_ref<T, 1>;
	using Array2D = typename boost::multi_array_ref<T, 2>;

	AmplitudesCalculatorNonVectors(const Array2D &sources_coords,
						 	  	  const Array2D &rec_coords,
						 	  	  const Array1D &tensor_matrix,
						 	  	  Array2D &amplitudes) : 
		sources_coords_(sources_coords),
		rec_coords_(rec_coords),
		tensor_matrix_(tensor_matrix),
		amplitudes_(amplitudes) 
	{ }

	friend AmplitudesCalculator<T, AmplitudesCalculatorNonVectors<T>>;

private:
	const Array2D &sources_coords_;
	const Array2D &rec_coords_;
	const Array1D &tensor_matrix_;
	Array2D &amplitudes_;	

	void realize_calculate() {
		this->non_vector_calculate_amplitudes(0, sources_coords_, rec_coords_, tensor_matrix_, amplitudes_);
	}
};

#endif /*_AMPLITUDES_CALCULATOR_NON_VECTORS*/