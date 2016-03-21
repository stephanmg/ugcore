/*!
 * \file idr.h
 *
 *  Created on: Mar 18, 2016
 *      Author: Stephan Grein
 */

#ifndef __H__UG__LIB_DISC__OPERATOR__LINEAR_OPERATOR__IDR__
#define __H__UG__LIB_DISC__OPERATOR__LINEAR_OPERATOR__IDR__

#include <iostream>
#include <string>
#include <sstream>

#include "lib_algebra/operator/interface/operator.h"
 #include "lib_algebra/operator/interface/linear_solver_profiling.h"
#ifdef UG_PARALLEL
	#include "lib_algebra/parallelization/parallelization.h"
#endif

namespace ug {
///	the IDR(s) method as a solver for linear operators
/**
 * This class implements the IDR(s) method for the solution of linear
 * operator problems like \f$A*x = b\f$, where the solution \f$x = A^{-1}*b\f$
 * is computed. The parameter s refers to the dimension of the shadow space.
 *
 * For detailed description of the algorithm, please refer to:
 *
 * -  Peter Sonneveld and Martin B. van Gijzen, IDR(s): a family of
 *   simple and fast algorithms for solving large nonsymmetric linear systems.
 *   SIAM J. Sci. Comput. Vol. 31, No. 2, pp. 1035-1062, 2008 (copyright SIAM)
 *
 * \tparam 	TVector		vector type
 */
template <typename TVector>
class IDR
	: public IPreconditionedLinearOperatorInverse<TVector>
{
	public:
	///	Vector type
		typedef TVector vector_type;

	///	Base type
		typedef IPreconditionedLinearOperatorInverse<vector_type> base_type;

	protected:
		using base_type::convergence_check;
		using base_type::linear_operator;
		using base_type::preconditioner;
		using base_type::write_debug;

	private:
		size_t m_s;

	public:
	///	constructors
		IDR() :
			m_s(1)
		{};

		IDR(SmartPtr<ILinearIterator<vector_type,vector_type> > spPrecond)
			: base_type ( spPrecond ), m_s(1)
		{}

		IDR( SmartPtr<ILinearIterator<vector_type> > spPrecond,
		          SmartPtr<IConvergenceCheck<vector_type> > spConvCheck)
			: base_type(spPrecond, spConvCheck), m_s(1)
		{};


	///	name of solver
		virtual const char* name() const {return "IDR";}

	///	returns if parallel solving is supported
		virtual bool supports_parallel() const
		{
			if(preconditioner().valid())
				return preconditioner()->supports_parallel();
			return true;
		}

	// 	Solve J(u)*x = b, such that x = J(u)^{-1} b
		virtual bool apply_return_defect(vector_type& x, vector_type& b)
		{
			LS_PROFILE_BEGIN(LS_ApplyReturnDefect);

		//	check correct storage type in parallel
			#ifdef UG_PARALLEL
			if(!b.has_storage_type(PST_ADDITIVE) || !x.has_storage_type(PST_CONSISTENT))
				UG_THROW("IDR: Inadequate storage format of Vectors.");
			#endif

		// 	build defect:  r := b - A*x
			linear_operator()->apply_sub(b, x);
			vector_type& r = b;

		// 	create vectors
			SmartPtr<vector_type> spR = r.clone_without_values(); vector_type& r0 = *spR;
			SmartPtr<vector_type> spP = r.clone_without_values(); vector_type& p = *spP;
			SmartPtr<vector_type> spV = r.clone_without_values(); vector_type& v = *spV;
			SmartPtr<vector_type> spT = r.clone_without_values(); vector_type& t = *spT;
			SmartPtr<vector_type> spS = r.clone_without_values(); vector_type& s = *spS;
			SmartPtr<vector_type> spQ = x.clone_without_values(); vector_type& q = *spQ;

		//	prepare convergence check
			prepare_conv_check();

		//	compute start defect norm
			convergence_check()->start(r);

		//	convert b to unique (should already be unique due to norm calculation)
			#ifdef UG_PARALLEL
			if(!r.change_storage_type(PST_UNIQUE))
				UG_THROW("IDR: Cannot convert b to unique vector.");
			#endif

			/// TODO implement remaining

			//	print ending output
			return convergence_check()->post();
		}

	protected:
	///	prepares the output of the convergence check
		void prepare_conv_check()
		{
		//	set iteration symbol and name
			convergence_check()->set_name(name());
			convergence_check()->set_symbol('%');

		//	set preconditioner string
			std::string s;
			if(preconditioner().valid())
			  s = std::string(" (Precond: ") + preconditioner()->name() + ")";
			else
				s = " (No Preconditioner) ";
			convergence_check()->set_info(s);
		}

	public:
		virtual std::string config_string() const
		{
			return "IDR";
		}
};
} // end namespace ug

#endif /// __H__UG__LIB_DISC__OPERATOR__LINEAR_OPERATOR__IDR__
