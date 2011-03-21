/*
 * lib_algebra_bridge.cpp
 *
 *  Created on: 11.10.2010
 *      Author: andreasvogel
 */

// extern headers
#include <iostream>
#include <sstream>

//#define UG_USE_AMG // temporary switch until AMG for systems works again


// bridge
#include "../ug_bridge.h"

// algebra includes
#include "lib_algebra/algebra_chooser.h"

// \todo: extract only really needed includes
// all parts of lib algebra
#include "lib_algebra/lib_algebra.h"


// user data (temporarily for Kosta Update)
//#include "ug_script/user_data/user_data.h" // kostaupdate is disabled?

// \todo: remove this dependency
// WARNING: To use AMG, please define UG_USE_AMG __and__ (!!) uncomment
// 			the include of lib_discretization. It is uncommented here, since
//          else we get a build of this file on each change of lib_disc (Maybe
//			cmake does not recognize the undef (?) )

//#undef UG_USE_AMG

namespace ug
{
extern enum_AlgebraType g_AlgebraType;
namespace bridge
{

template <typename TAlgebra>
void RegisterAlgebraType(Registry& reg, const char* parentGroup)
{
//	get group string (use same as parent)
	std::string grp = std::string(parentGroup);

//	typedefs for this algebra
	typedef TAlgebra algebra_type;
	typedef typename algebra_type::vector_type vector_type;
	typedef typename algebra_type::matrix_type matrix_type;


	//	Vector
	{
			reg.add_class_<vector_type>("Vector", grp.c_str())
			.add_constructor()
			.add_method("set|hide=true", (bool (vector_type::*)(number))&vector_type::set,
									"Success", "Number")
			.add_method("size|hide=true", (size_t (vector_type::*)())&vector_type::size,
									"Size", "")
			.add_method("set_random|hide=true", (bool (vector_type::*)(number))&vector_type::set_random,
									"Success", "Number")
			.add_method("print|hide=true", &vector_type::p);

		reg.add_function("VecScaleAdd2", &VecScaleAdd2<vector_type>, "", "alpha1*vec1 + alpha2*vec2",
				"dest#alpha1#vec1#alpha2#vec2");
		reg.add_function("VecScaleAdd3", &VecScaleAdd3<vector_type>, "", "alpha1*vec1 + alpha2*vec2 + alpha3*vec3",
				"dest#alpha1#vec1#alpha2#vec2#alpha3#vec3");
	}

	// Vector copy
	{
		reg.add_function("VecScaleAssign", (void (*)(vector_type&, number, const vector_type &))&VecScaleAssign<vector_type>);
	}

	//	Matrix
	{
		reg.add_class_<matrix_type>("Matrix", grp.c_str())
			.add_constructor()
			.add_method("print|hide=true", &matrix_type::p);
	}


	//	ApplyLinearSolver
	{
		reg.add_function( "ApplyLinearSolver",
						  &ApplyLinearSolver<vector_type>, grp.c_str());
	}


	// Debug Writer (abstract base class)
		{
			typedef IDebugWriter<algebra_type> T;
			reg.add_class_<T>("IDebugWriter", grp.c_str());
		}

	// IPositionProvider (abstract base class)
	{
		reg.add_class_<IPositionProvider<1> >("IPositionProvider1d", grp.c_str());
		reg.add_class_<IPositionProvider<2> >("IPositionProvider2d", grp.c_str());
		reg.add_class_<IPositionProvider<3> >("IPositionProvider3d", grp.c_str());
	}

	// IVectorWriter (abstract base class)
	{
		typedef IVectorWriter<vector_type> T;
		reg.add_class_<T>("IVectorWriter", grp.c_str());
	}


	// Base Classes
	{
		{
		//	ILinearOperator
			typedef ILinearOperator<vector_type, vector_type> T;
			reg.add_class_<T>("ILinearOperator", grp.c_str())
				.add_method("init", (bool(T::*)())&T::init);
		}

		{
		//	IMatrixOperator
			typedef ILinearOperator<vector_type, vector_type> TBase;
			typedef IMatrixOperator<vector_type, vector_type, matrix_type> T;
			reg.add_class_<T, TBase>("IMatrixOperator", grp.c_str())
				.add_method("resize", &T::resize)
				.add_method("num_rows", &T::num_rows, "rows")
				.add_method("num_cols", &T::num_cols, "cols");
		}

		{
		//	PureMatrixOperator
			typedef IMatrixOperator<vector_type, vector_type, matrix_type> TBase;
			typedef PureMatrixOperator<vector_type, vector_type, matrix_type> T;
			reg.add_class_<T, TBase>("PureMatrixOperator", grp.c_str())
				.add_constructor();
		}

		{
		//	ILinearIterator
			typedef ILinearIterator<vector_type, vector_type> T;
			reg.add_class_<T>("ILinearIterator", grp.c_str());
		}

		{
		//	IPreconditioner
			typedef ILinearIterator<vector_type, vector_type>  TBase;
			typedef IPreconditioner<algebra_type> T;
			reg.add_class_<T, TBase>("IPreconditioner", grp.c_str());
		}

		{
		//	ILinearOperatorInverse
			typedef ILinearOperatorInverse<vector_type, vector_type> T;
			reg.add_class_<T>("ILinearOperatorInverse", grp.c_str())
				.add_method("init", (bool(T::*)(ILinearOperator<vector_type,vector_type>&))&T::init)
				.add_method("apply_return_defect", &T::apply_return_defect, "Success", "u#f",
						"Solve A*u = f, such that u = A^{-1} f by iterating u := u + B(f - A*u),  f := f - A*u becomes new defect")
				.add_method("apply", &T::apply, "Success", "u#f", "Solve A*u = f, such that u = A^{-1} f by iterating u := u + B(f - A*u), f remains constant");
		}

		{
		//	IMatrixOperatorInverse
			typedef ILinearOperatorInverse<vector_type, vector_type>  TBase;
			typedef IMatrixOperatorInverse<vector_type, vector_type, matrix_type> T;
			reg.add_class_<T, TBase>("IMatrixOperatorInverse", grp.c_str());
		}

		{
		//	IOperator
			typedef IOperator<vector_type, vector_type> T;
			reg.add_class_<T>("IOperator", grp.c_str());
		}

		{
		//	IOperatorInverse
			typedef IOperatorInverse<vector_type, vector_type> T;
			reg.add_class_<T>("IOperatorInverse", grp.c_str());
		}

		{
		//	IProlongationOperator
			typedef IProlongationOperator<vector_type, vector_type> T;
			reg.add_class_<T>("IProlongationOperator", grp.c_str());
		}

		{
		//	IProjectionOperator
			typedef IProjectionOperator<vector_type, vector_type> T;
			reg.add_class_<T>("IProjectionOperator", grp.c_str());
		}
	}

	// Preconditioner
	{
	//	get group string
		std::stringstream grpSS2; grpSS2 << grp << "/Preconditioner";
		std::string grp2 = grpSS2.str();

	//	Jacobi
		reg.add_class_<	JacobiPreconditioner<algebra_type>,
						IPreconditioner<algebra_type> >("Jacobi", grp2.c_str())
			.add_constructor()
			.add_method("set_damp", &JacobiPreconditioner<algebra_type>::set_damp, "", "damp");

	//	GaussSeidel
		reg.add_class_<	GSPreconditioner<algebra_type>,
						IPreconditioner<algebra_type> >("GaussSeidel", grp2.c_str())
			.add_constructor();

	//	Symmetric GaussSeidel
		reg.add_class_<	SGSPreconditioner<algebra_type>,
						IPreconditioner<algebra_type> >("SymmetricGaussSeidel", grp2.c_str())
			.add_constructor();

	//	Backward GaussSeidel
		reg.add_class_<	BGSPreconditioner<algebra_type>,
						IPreconditioner<algebra_type> >("BackwardGaussSeidel", grp2.c_str())
			.add_constructor();

	//	ILU
		reg.add_class_<	ILUPreconditioner<algebra_type>,
						IPreconditioner<algebra_type> >("ILU", grp2.c_str())
			.add_constructor()
			.add_method("set_debug", &ILUPreconditioner<algebra_type>::set_debug, "", "d");


	//	ILU Threshold
		reg.add_class_<	ILUTPreconditioner<algebra_type>,
						IPreconditioner<algebra_type> >("ILUT", grp2.c_str())
			.add_constructor()
			.add_method("set_threshold", &ILUTPreconditioner<algebra_type>::set_threshold,
						"", "threshold", "sets threshold of incomplete LU factorisation"); // added 01122010ih






#ifdef UG_USE_AMG
	//	AMG
		reg.add_class_< typename amg_base<algebra_type>::LevelInformation > ("AMGLevelInformation", grp2.c_str())
			.add_method("get_creation_time_ms", &amg_base<algebra_type>::LevelInformation::get_creation_time_ms, "creation time of this level (in ms)")
			.add_method("get_nr_of_nodes", &amg_base<algebra_type>::LevelInformation::get_nr_of_nodes, "nr of nodes of this level")
			.add_method("is_valid", &amg_base<algebra_type>::LevelInformation::is_valid, "true if this is a valid level information");

//todo: existance of AMGPreconditioner class should not depend on defines.
		reg.add_class_<	amg_base<algebra_type>, IPreconditioner<algebra_type> > ("AMGBase", grp2.c_str())
			.add_method("set_num_presmooth", &amg_base<algebra_type>::set_num_presmooth, "", "nu1", "sets nr. of presmoothing steps (nu1)")
			.add_method("get_num_presmooth", &amg_base<algebra_type>::get_num_presmooth, "nr. of presmoothing steps (nu1)")

			.add_method("set_num_postsmooth", &amg_base<algebra_type>::set_num_postsmooth, "", "nu2", "sets nr. of postsmoothing steps (nu2)")
			.add_method("get_num_postsmooth", &amg_base<algebra_type>::get_num_postsmooth, "nr. of postsmoothing steps (nu2)")

			.add_method("set_cycle_type", &amg_base<algebra_type>::set_cycle_type, "", "gamma", "sets cycle type in multigrid cycle (gamma)")
			.add_method("get_cycle_type", &amg_base<algebra_type>::get_cycle_type, "cycle type in multigrid cycle (gamma)")

			.add_method("set_max_levels", &amg_base<algebra_type>::set_max_levels, "", "max_levels", "sets max nr of AMG levels")
			.add_method("get_max_levels", &amg_base<algebra_type>::get_max_levels,  "max nr of AMG levels")

			.add_method("set_max_nodes_for_base", &amg_base<algebra_type>::set_max_nodes_for_base, "", "maxNrOfNodes", "sets the maximal nr of nodes for base solver")
			.add_method("get_max_nodes_for_base", &amg_base<algebra_type>::get_max_nodes_for_base, "maximal nr of nodes for base solver")

			.add_method("set_max_fill_before_base", &amg_base<algebra_type>::set_max_fill_before_base, "", "fillrate", "sets maximal fill rate before base solver is used")
			.add_method("get_max_fill_before_base", &amg_base<algebra_type>::get_max_fill_before_base, "maximal fill rate before base solver is used", "")

			.add_method("get_operator_complexity", &amg_base<algebra_type>::get_operator_complexity, "", "")
			.add_method("get_nodes_complexity", &amg_base<algebra_type>::get_nodes_complexity, "", "")
			.add_method("get_timing_whole_setup_ms", &amg_base<algebra_type>::get_timing_whole_setup_ms, "", "")
			.add_method("get_timing_coarse_solver_ms", &amg_base<algebra_type>::get_timing_coarse_solver_ms, "", "")
			.add_method("get_level_information", &amg_base<algebra_type>::get_level_information, "", "")

			.add_method("set_presmoother", &amg_base<algebra_type>::set_presmoother, "", "presmoother")
			.add_method("set_postsmoother", &amg_base<algebra_type>::set_postsmoother, "", "postsmoother")
			.add_method("set_base_solver", &amg_base<algebra_type>::set_base_solver, "", "basesmoother")
			.add_method("check", &amg_base<algebra_type>::check, "", "")
			.add_method("set_matrix_write_path", &amg_base<algebra_type>::set_matrix_write_path, "", "")
			.add_method("set_fsmoothing", &amg_base<algebra_type>::set_fsmoothing, "", "")

			.add_method("set_position_provider2d", &amg_base<algebra_type>::set_position_provider2d)
			.add_method("set_position_provider3d", &amg_base<algebra_type>::set_position_provider3d)
			;

		reg.add_class_<	amg<algebra_type>, amg_base<algebra_type> > ("AMGPreconditioner", grp2.c_str())
			.add_constructor()
			.add_method("set_theta", &amg<algebra_type>::set_theta, "", "theta", "sets theta, a measure for strong connectivity")
			.add_method("get_theta", &amg<algebra_type>::get_theta, "theta", "")
			.add_method("set_sigma", &amg<algebra_type>::set_sigma, "", "sigma", "sets theta, a parameter used for truncation of interpolation")
			.add_method("get_sigma", &amg<algebra_type>::get_sigma, "", "sigma", "sets theta, a parameter used for truncation of interpolation")

			.add_method("tostring", &amg<algebra_type>::tostring)
			.add_method("enable_aggressive_coarsening_A", &amg<algebra_type>::enable_aggressive_coarsening_A)
			.add_method("disable_aggressive_coarsening", &amg<algebra_type>::disable_aggressive_coarsening)
			.add_method("is_aggressive_coarsening", &amg<algebra_type>::is_aggressive_coarsening)
			.add_method("is_aggressive_coarsening_A", &amg<algebra_type>::is_aggressive_coarsening_A);

		reg.add_class_<	famg<algebra_type>, amg_base<algebra_type> > ("FAMGPreconditioner", grp2.c_str())
			.add_constructor()
			.add_method("tostring", &famg<algebra_type>::tostring)
			.add_method("set_aggressive_coarsening", &famg<algebra_type>::set_aggressive_coarsening)
			.add_method("set_delta", &famg<algebra_type>::set_delta, "", "delta", "\"Interpolation quality\" F may not be worse than this (F < m_delta)")
			.add_method("set_theta", &famg<algebra_type>::set_theta, "" , "theta", "clip all interpolations with m_theta * F > min F.")
			.add_method("set_damping_for_smoother_in_interpolation_calculation",
					&famg<algebra_type>::set_damping_for_smoother_in_interpolation_calculation)
			.add_method("set_testvector_damps", &famg<algebra_type>::set_testvector_damps)
			.add_method("reset_testvectors", &famg<algebra_type>::reset_testvectors)
			.add_method("add_testvector", &famg<algebra_type>::add_testvector)
			.add_method("add_vector_writer", &famg<algebra_type>::add_vector_writer)
			;
#endif
	}

	//	LinearIteratorProduct
					reg.add_class_<	LinearIteratorProduct<vector_type, vector_type>,
									ILinearIterator<vector_type, vector_type> >("LinearIteratorProduct", grp.c_str())
						.add_constructor()
						.add_method("add_iteration", &LinearIteratorProduct<vector_type, vector_type>::add_iterator, "Add an iterator");
/*
	{
#ifdef LAPACK_AVAILABLE
		string subgroup = grp; // + string("/Preconditioner");

		reg.add_class_<	PINVIT<algebra_type> >("EigenSolver", subgroup.c_str())
			.add_constructor()
			.add_method("add_vector", &PINVIT<algebra_type>::add_vector,
						"", "vector")
			.add_method("set_preconditioner|interactive=false", &PINVIT<algebra_type>::set_preconditioner,
						"", "Preconditioner")
			.add_method("set_linear_operator_A|interactive=false", &PINVIT<algebra_type>::set_linear_operator_A,
						"", "LinearOperatorA")
			.add_method("set_linear_operator_B|interactive=false", &PINVIT<algebra_type>::set_linear_operator_B,
						"", "LinearOperatorB")
			.add_method("set_max_iterations|interactive=false", &PINVIT<algebra_type>::set_max_iterations,
							"", "precision")
			.add_method("set_precision|interactive=false", &PINVIT<algebra_type>::set_precision,
							"", "precision")
			.add_method("apply", &PINVIT<algebra_type>::apply);
#endif
	}
*/
	// todo: Solvers should be independent of type and placed in general section
	{
	//	get group string
		std::stringstream grpSS3; grpSS3 << grp << "/Solver";
		std::string grp3 = grpSS3.str();

	// 	LinearSolver
		reg.add_class_<	LinearSolver<algebra_type>,
						ILinearOperatorInverse<vector_type, vector_type> >("LinearSolver", grp3.c_str())
			.add_constructor()
			.add_method("set_preconditioner|interactive=false", &LinearSolver<algebra_type>::set_preconditioner,
						"", "Preconditioner")
			.add_method("set_convergence_check|interactive=false", &LinearSolver<algebra_type>::set_convergence_check,
						"", "Check");

	// 	CG Solver
		reg.add_class_<	CGSolver<algebra_type>,
						ILinearOperatorInverse<vector_type, vector_type> >("CG", grp3.c_str())
			.add_constructor()
			.add_method("set_preconditioner|interactive=false", &CGSolver<algebra_type>::set_preconditioner,
						"", "Preconditioner")
			.add_method("set_convergence_check|interactive=false", &CGSolver<algebra_type>::set_convergence_check,
						"", "Check");

	// 	BiCGStab Solver
		reg.add_class_<	BiCGStabSolver<algebra_type>,
						ILinearOperatorInverse<vector_type, vector_type> >("BiCGStab", grp3.c_str())
			.add_constructor()
			.add_method("set_preconditioner|interactive=false", &BiCGStabSolver<algebra_type>::set_preconditioner,
						"", "Preconditioner")
			.add_method("set_convergence_check|interactive=false", &BiCGStabSolver<algebra_type>::set_convergence_check,
						"", "Check");

	// 	LUSolver
		reg.add_class_<	LUSolver<algebra_type>,
						ILinearOperatorInverse<vector_type, vector_type> >("LU", grp3.c_str())
			.add_constructor()
			.add_method("set_convergence_check|interactive=false", &LUSolver<algebra_type>::set_convergence_check,
						"", "Check");

	// 	DirichletDirichletSolver
#ifdef UG_PARALLEL
		{
			typedef DirichletDirichletSolver<algebra_type> T;
			typedef ILinearOperatorInverse<vector_type, vector_type> BaseT;
			reg.add_class_<	T, BaseT >("DirichletDirichlet", grp3.c_str())
			.add_constructor()
			.add_method("set_convergence_check|interactive=false", &T::set_convergence_check,
						"", "Check")
			.add_method("set_theta|interactive=false", &T::set_theta,
						"", "Theta", "set damping factor theta")
			.add_method("set_neumann_solver|interactive=false", &T::set_neumann_solver,
						"", "Neumann Solver")
			.add_method("set_dirichlet_solver|interactive=false", &T::set_dirichlet_solver,
						"", "Dirichlet Solver")
			.add_method("set_debug", &T::set_debug);
		}
#endif
	// 	LocalSchurComplement
#ifdef UG_PARALLEL
		{
			typedef LocalSchurComplement<algebra_type> T;
			reg.add_class_<	T, ILinearOperator<vector_type, vector_type> >("LocalSchurComplement", grp3.c_str())
			.add_constructor()
			.add_method("set_matrix|interactive=false", &T::set_matrix,
						"", "Matrix")
			.add_method("set_dirichlet_solver|interactive=false", &T::set_dirichlet_solver,
						"", "Dirichlet Solver")
			.add_method("set_debug", &T::set_debug, "", "d")
			// the following functions would normally not be executed from script
			.add_method("init", (bool (T::*)())&T::init)
			.add_method("apply", &T::apply,
						"Success", "local SC times Vector", "Vector");
		}
#endif
	// 	FETISolver
#ifdef UG_PARALLEL
		{
			typedef FETISolver<algebra_type> T;
			typedef IMatrixOperatorInverse<vector_type, vector_type, matrix_type> BaseT;
			reg.add_class_<	T, BaseT >("FETI", grp3.c_str())
			.add_constructor()
			.add_method("set_convergence_check|interactive=false", &T::set_convergence_check,
						"", "Check")
			.add_method("set_neumann_solver|interactive=false", &T::set_neumann_solver,
						"", "Neumann Solver")
			.add_method("set_dirichlet_solver|interactive=false", &T::set_dirichlet_solver,
						"", "Dirichlet Solver")
			.add_method("set_coarse_problem_solver|interactive=false", &T::set_coarse_problem_solver,
						"", "Coarse Problem Solver")
			.add_method("set_domain_decomp_info", &T::set_domain_decomp_info)
			.add_method("print_statistic_of_inner_solver", &T::print_statistic_of_inner_solver)
			.add_method("set_debug", &T::set_debug);
		}
#endif

	// 	HLIBSolver
#ifdef USE_HLIBPRO
		{
			typedef HLIBSolver<algebra_type> T;
			reg.add_class_<	T, ILinearOperatorInverse<vector_type, vector_type> >("HLIBSolver", grp3.c_str())
			.add_constructor()
			.add_method("set_convergence_check|interactive=false", &T::set_convergence_check,
						"", "Check")
			.add_method("set_hlib_nmin|interactive=false",         &T::set_hlib_nmin,
						"", "HLIB nmin")
			.add_method("set_hlib_accuracy_H|interactive=false",   &T::set_hlib_accuracy_H,
						"", "HLIB accuracy_H")
			.add_method("set_hlib_accuracy_LU|interactive=false",  &T::set_hlib_accuracy_LU,
						"", "HLIB accuracy_LU")
			.add_method("set_hlib_verbosity|interactive=false",    &T::set_hlib_verbosity,
						"", "HLIB verbosity")
			.add_method("set_clustering_method|interactive=false", &T::set_clustering_method,
						"", "Clustering")
			.add_method("set_ps_basename|interactive=false",       &T::set_ps_basename,
						"", "PostScript basename")
			.add_method("check_crs_matrix|interactive=false",      &T::check_crs_matrix,
						"", "Check CR matrix")
/*			*/
			.add_method("set_debug", &T::set_debug);
		}
#endif

	}

}



bool RegisterStaticLibAlgebraInterface(Registry& reg, const char* parentGroup)
{
	try
	{
	//	get group string
		std::stringstream groupString; groupString << parentGroup << "/Algebra";
		std::string grp = groupString.str();

		// Chooser Interface
		reg.add_class_<	AlgebraTypeChooserInterface >("AlgebraTypeChooserInterface", grp.c_str());
		reg.add_class_<	CPUAlgebraChooser,AlgebraTypeChooserInterface >("CPUAlgebraChooser", grp.c_str())
			.add_constructor()
			.add_method("set_fixed_blocksize", &CPUAlgebraChooser::set_fixed_blocksize, "", "blocksize")
			.add_method("set_variable_blocksize", &CPUAlgebraChooser::set_variable_blocksize);

		// StandardConvCheck
		reg.add_class_<IConvergenceCheck>("IConvergenceCheck", grp.c_str());

		reg.add_class_<StandardConvCheck, IConvergenceCheck>("StandardConvergenceCheck", grp.c_str())
			.add_constructor()
			.add_method("set_maximum_steps|interactive=false", &StandardConvCheck::set_maximum_steps,
					"", "Maximum Steps")
			.add_method("set_minimum_defect|interactive=false", &StandardConvCheck::set_minimum_defect,
					"", "Minimum Defect")
			.add_method("set_reduction|interactive=false", &StandardConvCheck::set_reduction,
					"", "Reduction")
			.add_method("set_verbose_level|interactive=false", &StandardConvCheck::set_verbose_level,
					"", "Verbose");

//		reg.add_function("KostaUpdate", &KostaUpdate<CPUAlgebra::vector_type>);
	}

	catch(UG_REGISTRY_ERROR_RegistrationFailed ex)
	{
		UG_LOG("### ERROR in RegisterLibAlgebraInterface: "
				"Registration failed (using name " << ex.name << ").\n");
		return false;
	}

	return true;
}

bool RegisterDynamicLibAlgebraInterface(Registry& reg, int algebra_type, const char* parentGroup)
{
	try
	{
		//	get group string
		std::stringstream groupString; groupString << parentGroup << "/Algebra";
		std::string grp = groupString.str();

		// register algebra
		switch(algebra_type)
		{
		case eCPUAlgebra:		 		RegisterAlgebraType<CPUAlgebra >(reg, grp.c_str()); break;
//		case eCPUBlockAlgebra2x2: 		RegisterAlgebraType<CPUBlockAlgebra<2> >(reg, grp.c_str()); break;
		case eCPUBlockAlgebra3x3: 		RegisterAlgebraType<CPUBlockAlgebra<3> >(reg, grp.c_str()); break;
//		case eCPUBlockAlgebra4x4: 		RegisterAlgebraType<CPUBlockAlgebra<4> >(reg, grp.c_str()); break;
//		case eCPUVariableBlockAlgebra: 	RegisterAlgebraType<CPUVariableBlockAlgebra>(reg, grp.c_str()); break;
		default: UG_ASSERT(0, "In RegisterDynamicLibAlgebraInterface: " << algebra_type << " is unsupported algebra type");
					UG_LOG("In RegisterDynamicLibAlgebraInterface: " << algebra_type << " is unsupported algebra type");
					return false;
		}
	}
	catch(UG_REGISTRY_ERROR_RegistrationFailed ex)
	{
		UG_LOG("### ERROR in RegisterLibAlgebraInterface: "
				"Registration failed (using name " << ex.name << ").\n");
		return false;
	}
	return true;
}


} // end namespace bridge
} // end namespace ug
