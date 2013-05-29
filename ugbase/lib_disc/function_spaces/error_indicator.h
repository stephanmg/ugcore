/*
 * error_indicator.h
 *
 *  Created on: 30.03.2011
 *      Author: andreasvogel
 */

#ifndef __H__UG__LIB_DISC__FUNCTION_SPACE__ERROR_INDICATOR__
#define __H__UG__LIB_DISC__FUNCTION_SPACE__ERROR_INDICATOR__

#include <vector>

#include "common/common.h"
#include "common/util/provider.h"
#include "lib_grid/algorithms/refinement/refiner_interface.h"
#include "lib_disc/common/geometry_util.h"
#include "lib_disc/reference_element/reference_element_util.h"
#include "lib_disc/reference_element/reference_mapping_provider.h"
#include "lib_disc/spatial_disc/disc_util/fvcr_geom.h"

namespace ug{

template <typename TFunction>
void ComputeGradientLagrange1(TFunction& u, size_t fct,
                     MultiGrid::AttachmentAccessor<
                     typename TFunction::element_type,
                     ug::Attachment<number> >& aaError)
{
	static const int dim = TFunction::dim;
	typedef typename TFunction::const_element_iterator const_iterator;
	typedef typename TFunction::element_type element_type;

//	get position accessor
	typename TFunction::domain_type::position_accessor_type& aaPos
			= u.domain()->position_accessor();

//	some storage
	MathMatrix<dim, dim> JTInv;
	std::vector<MathVector<dim> > vLocalGrad;
	std::vector<MathVector<dim> > vGlobalGrad;
	std::vector<MathVector<dim> > vCorner;

//	get iterator over elements
	const_iterator iter = u.template begin<element_type>();
	const_iterator iterEnd = u.template end<element_type>();

//	loop elements
	for(; iter != iterEnd; ++iter)
	{
	//	get the element
		element_type* elem = *iter;

	//	reference object type
		ReferenceObjectID roid = elem->reference_object_id();

	//	get trial space
		const LocalShapeFunctionSet<dim>& lsfs =
				LocalShapeFunctionSetProvider::get<dim>(roid, LFEID(LFEID::LAGRANGE, 1));

	//	create a reference mapping
		DimReferenceMapping<dim, dim>& map
			= ReferenceMappingProvider::get<dim, dim>(roid);

	//	get local Mid Point
		MathVector<dim> localIP = ReferenceElementCenter<dim>(roid);

	//	number of shape functions
		const size_t numSH = lsfs.num_sh();
		vLocalGrad.resize(numSH);
		vGlobalGrad.resize(numSH);

	//	evaluate reference gradient at local midpoint
		lsfs.grads(&vLocalGrad[0], localIP);

	//	get corners of element
		CollectCornerCoordinates(vCorner, *elem, aaPos);

	//	update mapping
		map.update(&vCorner[0]);

	//	compute jacobian
		map.jacobian_transposed_inverse(JTInv, localIP);

	//	compute size (volume) of element
		const number elemSize = ElementSize<dim>(roid, &vCorner[0]);

	//	compute gradient at mid point by summing contributions of all shape fct
		MathVector<dim> MidGrad; VecSet(MidGrad, 0.0);
		for(size_t sh = 0 ; sh < numSH; ++sh)
		{
		//	get global Gradient
			MatVecMult(vGlobalGrad[sh], JTInv, vLocalGrad[sh]);

		//	get vertex
			VertexBase* vert = elem->vertex(sh);

		//	get of of vertex
			std::vector<MultiIndex<2> > ind;
			u.inner_multi_indices(vert, fct, ind);

		//	scale global gradient
			vGlobalGrad[sh] *= BlockRef(u[ind[0][0]], ind[0][1]);

		//	sum up
			MidGrad += vGlobalGrad[sh];
		}

	//	write result in array storage
		aaError[elem] = VecTwoNorm(MidGrad) * pow(elemSize, 2./dim);
	}
}

template <typename TFunction>
void ComputeGradientCrouzeixRaviart(TFunction& u, size_t fct,
                     MultiGrid::AttachmentAccessor<
                     typename TFunction::element_type,
                     ug::Attachment<number> >& aaError)
{
	static const int dim = TFunction::dim;
	typedef typename TFunction::const_element_iterator const_iterator;
	typedef typename TFunction::domain_type domain_type;
	typedef typename domain_type::grid_type grid_type;
	typedef typename TFunction::element_type element_type;
	typedef typename element_type::side side_type;
	
	typename grid_type::template traits<side_type>::secure_container sides;

//	get position accessor
	typename TFunction::domain_type::position_accessor_type& aaPos
			= u.domain()->position_accessor();
			
	grid_type& grid = *u.domain()->grid();

//	some storage
	MathMatrix<dim, dim> JTInv;
	std::vector<MathVector<dim> > vLocalGrad;
	std::vector<MathVector<dim> > vGlobalGrad;
	std::vector<MathVector<dim> > vCorner;

//	get iterator over elements
	const_iterator iter = u.template begin<element_type>();
	const_iterator iterEnd = u.template end<element_type>();

//	loop elements
	for(; iter != iterEnd; ++iter)
	{
	//	get the element
		element_type* elem = *iter;
	
	//  get sides of element		
		grid.associated_elements_sorted(sides, elem );

	//	reference object type
		ReferenceObjectID roid = elem->reference_object_id();

	//	get trial space
		const LocalShapeFunctionSet<dim>& lsfs =
				LocalShapeFunctionSetProvider::get<dim>(roid, LFEID(LFEID::CROUZEIX_RAVIART, 1));

	//	create a reference mapping
		DimReferenceMapping<dim, dim>& map
			= ReferenceMappingProvider::get<dim, dim>(roid);

	//	get local Mid Point
		MathVector<dim> localIP = ReferenceElementCenter<dim>(roid);

	//	number of shape functions
		const size_t numSH = lsfs.num_sh();
		vLocalGrad.resize(numSH);
		vGlobalGrad.resize(numSH);

	//	evaluate reference gradient at local midpoint
		lsfs.grads(&vLocalGrad[0], localIP);

	//	get corners of element
		CollectCornerCoordinates(vCorner, *elem, aaPos);

	//	update mapping
		map.update(&vCorner[0]);

	//	compute jacobian
		map.jacobian_transposed_inverse(JTInv, localIP);

	//	compute size (volume) of element
		const number elemSize = ElementSize<dim>(roid, &vCorner[0]);

	//	compute gradient at mid point by summing contributions of all shape fct
		MathVector<dim> MidGrad; VecSet(MidGrad, 0.0);
		for(size_t sh = 0 ; sh < numSH; ++sh)
		{
		//	get global Gradient
			MatVecMult(vGlobalGrad[sh], JTInv, vLocalGrad[sh]);

		//	get of of vertex
			std::vector<MultiIndex<2> > ind;
			u.inner_multi_indices(sides[sh], fct, ind);

		//	scale global gradient
			vGlobalGrad[sh] *= BlockRef(u[ind[0][0]], ind[0][1]);

		//	sum up
			MidGrad += vGlobalGrad[sh];
		}

	//	write result in array storage
		aaError[elem] = VecTwoNorm(MidGrad) * pow(elemSize, 2./dim);
	}
}

template <typename TFunction>
void ComputeGradientPiecewiseConstant(TFunction& u, size_t fct,
                     MultiGrid::AttachmentAccessor<
                     typename TFunction::element_type,
                     ug::Attachment<number> >& aaError)
{
	static const int dim = TFunction::dim;
	typedef typename TFunction::const_element_iterator const_iterator;
	typedef typename TFunction::domain_type domain_type;
	typedef typename domain_type::grid_type grid_type;
	typedef typename TFunction::element_type element_type;
	typedef typename element_type::side side_type;
	
	typename grid_type::template traits<side_type>::secure_container sides;

//	get position accessor
	typename TFunction::domain_type::position_accessor_type& aaPos
			= u.domain()->position_accessor();
			
	grid_type& grid = *u.domain()->grid();

	domain_type& domain = *u.domain().get();

//	some storage
	MathMatrix<dim, dim> JTInv;
	
	std::vector<MathVector<dim> > vCorner;

//	get iterator over elements
	const_iterator iter = u.template begin<element_type>();
	const_iterator iterEnd = u.template end<element_type>();
	
	DimCRFVGeometry<dim> geo;

//	loop elements
	for(; iter != iterEnd; ++iter)
	{
	//	get the element
		element_type* elem = *iter;
		
		MathVector<dim> vGlobalGrad=0;
	
	//  get sides of element		
		grid.associated_elements_sorted(sides, elem );

	//	reference object type
		ReferenceObjectID roid = elem->reference_object_id();

	//	get corners of element
		CollectCornerCoordinates(vCorner, *elem, aaPos);
		
		//	evaluate finite volume geometry
		geo.update(elem, &(vCorner[0]), domain.subset_handler().get());

	//	compute size (volume) of element
		const number elemSize = ElementSize<dim>(roid, &vCorner[0]);
		
		typename grid_type::template traits<element_type>::secure_container assoElements;
		
	// assemble element-wise finite volume gradient
		for (size_t s=0;s<sides.size();s++){
			grid.associated_elements(assoElements,sides[s]);
			// face value is average of associated elements
			number faceValue = 0;
			size_t numOfAsso = assoElements.size();
			for (size_t i=0;i<numOfAsso;i++){
				std::vector<MultiIndex<2> > ind;
				u.inner_multi_indices(assoElements[i], fct, ind);
				faceValue+=BlockRef(u[ind[0][0]], ind[0][1]);
			}
			faceValue/=(number)numOfAsso;
			const typename DimCRFVGeometry<dim>::SCV& scv = geo.scv(s);
			for (int d=0;d<dim;d++){
				vGlobalGrad[d] += faceValue * scv.normal()[d]; 
			}
		}
		vGlobalGrad/=(number)elemSize;

	//	write result in array storage
		aaError[elem] = VecTwoNorm(vGlobalGrad) * pow(elemSize, 2./dim);
	}
}


/// marks elements according to an attached error value field
/**
 * This function marks elements for refinement. The passed error attachment
 * is used as a weight for the amount of the error an each element. All elements
 * that have an indicated error with s* max <= err <= max are marked for refinement.
 * Here, max is the maximum error measured, s is a scaling quantity chosen by
 * the user. In addition, all elements with an error smaller than TOL
 * (user defined) are not refined.
 *
 * \param[in, out]	refiner		Refiner, elements marked on exit
 * \param[in]		u			Grid Function
 * \param[in]		TOL			Minimum error, such that an element is marked
 * \param[in]		scale		scaling factor indicating lower bound for marking
 * \param[in]		aaError		Error value attachment to elements
 */
template <typename TFunction>
void MarkElements(MultiGrid::AttachmentAccessor<typename TFunction::element_type,
                  ug::Attachment<number> >& aaError,
                  IRefiner& refiner,
                  TFunction& u,
                  number TOL,
                  number refineFrac, number coarseFrac,
				  int maxLevel)
{
	typedef typename TFunction::element_type element_type;
	typedef typename TFunction::const_element_iterator const_iterator;

//	reset maximum of error
	number max = 0.0, min = std::numeric_limits<number>::max();
	number totalErr = 0.0;
	int numElem = 0;

//	get element iterator
	const_iterator iter = u.template begin<element_type>();
	const_iterator iterEnd = u.template end<element_type>();

//	loop all elements to find the maximum of the error
	for( ;iter != iterEnd; ++iter)
	{
	//	get element
		element_type* elem = *iter;

	//	search for maximum and minimum
		if(aaError[elem] > max)	max = aaError[elem];
		if(aaError[elem] < min)	min = aaError[elem];

	//	sum up total error
		totalErr += aaError[elem];
		numElem += 1;
	}

#ifdef UG_PARALLEL
	number maxLocal = max, minLocal = min, totalErrLocal = totalErr;
	int numElemLocal = numElem;
	if(pcl::GetNumProcesses() > 1){
		pcl::ProcessCommunicator com;
		com.allreduce(&maxLocal, &max, 1, PCL_DT_DOUBLE, PCL_RO_MAX);
		com.allreduce(&minLocal, &min, 1, PCL_DT_DOUBLE, PCL_RO_MIN);
		com.allreduce(&totalErrLocal, &totalErr, 1, PCL_DT_DOUBLE, PCL_RO_SUM);
		com.allreduce(&numElemLocal, &numElem, 1, PCL_DT_INT, PCL_RO_SUM);
	}
#endif
	UG_LOG("  +++++  Gradient Error Indicator on "<<numElem<<" Elements +++++\n");
#ifdef UG_PARALLEL
	if(pcl::GetNumProcesses() > 1){
		UG_LOG("  +++ Element Errors on Proc " << pcl::GetProcRank() <<
			   ": maximum=" << max << ", minimum="<<min<<", sum="<<totalErr<<".\n");
	}
#endif

	UG_LOG("  +++ Element Errors: maximum=" << max << ", minimum="<<min<<
	       ", sum="<<totalErr<<".\n");

//	check if total error is smaller than tolerance. If that is the case we're done
	if(totalErr < TOL)
	{
		UG_LOG("  +++ Total error "<<totalErr<<" smaller than TOL ("<<TOL<<"). done.");
		return;
	}

//	Compute minimum
	number minErrToRefine = max * refineFrac;
	UG_LOG("  +++ Refining elements if error greater " << refineFrac << "*" <<max<<
			" = "<< minErrToRefine << ".\n");
	number maxErrToCoarse = min * (1+coarseFrac);
	if(maxErrToCoarse < TOL/numElem) maxErrToCoarse = TOL/numElem;
	UG_LOG("  +++ Coarsening elements if error smaller "<< maxErrToCoarse << ".\n");

//	reset counter
	int numMarkedRefine = 0, numMarkedCoarse = 0;

	iter = u.template begin<element_type>();
	iterEnd = u.template end<element_type>();

//	loop elements for marking
	for(; iter != iterEnd; ++iter)
	{
	//	get element
		element_type* elem = *iter;

	//	marks for refinement
		if(aaError[elem] >= minErrToRefine)
			if(u.domain()->grid()->get_level(elem) <= maxLevel)
			{
				refiner.mark(elem, RM_REFINE);
				numMarkedRefine++;
			}

	//	marks for coarsening
		if(aaError[elem] <= maxErrToCoarse)
		{
			refiner.mark(elem, RM_COARSEN);
			numMarkedCoarse++;
		}
	}

#ifdef UG_PARALLEL
	if(pcl::GetNumProcesses() > 1){
		UG_LOG("  +++ Marked for refinement on Proc "<<pcl::GetProcRank()<<": " << numMarkedRefine << " Elements.\n");
		UG_LOG("  +++ Marked for coarsening on Proc "<<pcl::GetProcRank()<<": " << numMarkedCoarse << " Elements.\n");
		pcl::ProcessCommunicator com;
		int numMarkedRefineLocal = numMarkedRefine, numMarkedCoarseLocal = numMarkedCoarse;
		com.allreduce(&numMarkedRefineLocal, &numMarkedRefine, 1, PCL_DT_INT, PCL_RO_SUM);
		com.allreduce(&numMarkedCoarseLocal, &numMarkedCoarse, 1, PCL_DT_INT, PCL_RO_SUM);
	}
#endif

	UG_LOG("  +++ Marked for refinement: " << numMarkedRefine << " Elements.\n");
	UG_LOG("  +++ Marked for coarsening: " << numMarkedCoarse << " Elements.\n");
}

template <typename TDomain, typename TAlgebra>
void MarkForAdaption_GradientIndicator(IRefiner& refiner,
                                       GridFunction<TDomain, TAlgebra>& u,
                                       const char* fctName,
                                       number TOL,
                                       number refineFrac, number coarseFrac,
                                       int maxLevel)
{
//	types
	typedef GridFunction<TDomain, TAlgebra> TFunction;
	typedef typename TFunction::domain_type::grid_type grid_type;
	typedef typename TFunction::element_type element_type;

//	function id
	const size_t fct = u.fct_id_by_name(fctName);

//	get multigrid
	SmartPtr<grid_type> pMG = u.domain()->grid();

// 	attach error field
	typedef Attachment<number> ANumber;
	ANumber aError;
	pMG->template attach_to<element_type>(aError);
	MultiGrid::AttachmentAccessor<element_type, ANumber> aaError(*pMG, aError);

// 	Compute error on elements
	if (u.local_finite_element_id(fct) == LFEID(LFEID::LAGRANGE, 1))
		ComputeGradientLagrange1(u, fct, aaError);
	else{
		if (u.local_finite_element_id(fct) == LFEID(LFEID::CROUZEIX_RAVIART, 1))
			ComputeGradientCrouzeixRaviart(u, fct, aaError);
		else{
			if (u.local_finite_element_id(fct) == LFEID(LFEID::PIECEWISE_CONSTANT, 0)){
				ComputeGradientPiecewiseConstant(u,fct,aaError);
			} else {
				UG_THROW("Non-supported finite element type " << u.local_finite_element_id(fct) << "\n");
			}
		}
	}
	
// 	Mark elements for refinement
	MarkElements(aaError, refiner, u, TOL, refineFrac, coarseFrac, maxLevel);

// 	detach error field
	pMG->template detach_from<element_type>(aError);
};

template <typename TFunction>
void computeGradientJump(TFunction& u,
                     MultiGrid::AttachmentAccessor<
                     typename TFunction::element_type,
                     ug::Attachment<number> >& aaGrad,
					 MultiGrid::AttachmentAccessor<
                     typename TFunction::element_type,
                     ug::Attachment<number> >& aaError)
{
	typedef typename TFunction::domain_type domain_type;
	typedef typename domain_type::grid_type grid_type;
	typedef typename TFunction::element_type element_type;
	typedef typename element_type::side side_type;
	typedef typename TFunction::template traits<side_type>::const_iterator side_iterator;
	
	grid_type& grid = *u.domain()->grid();

	//	get iterator over elements
	side_iterator iter = u.template begin<side_type>();
	side_iterator iterEnd = u.template end<side_type>();
	//	loop elements
	for(; iter != iterEnd; ++iter)
	{
		//	get the element
		side_type* side = *iter;
		typename grid_type::template traits<element_type>::secure_container neighElements;
		grid.associated_elements(neighElements,side);
		if (neighElements.size()!=2) continue;
		number localJump = std::abs(aaGrad[neighElements[0]]-aaGrad[neighElements[1]]);
		for (size_t i=0;i<2;i++)
			if (aaError[neighElements[i]]<localJump) aaError[neighElements[i]]=localJump;
	}
}

// indicator is based on elementwise computation of jump between elementwise gradients
// the value in an element is the maximum jump between the element gradient and gradient
// in elements with common face
template <typename TDomain, typename TAlgebra>
void MarkForAdaption_GradientJumpIndicator(IRefiner& refiner,
                                       GridFunction<TDomain, TAlgebra>& u,
                                       const char* fctName,
                                       number TOL,
                                       number refineFrac, number coarseFrac,
                                       int maxLevel)
{
//	types
	typedef GridFunction<TDomain, TAlgebra> TFunction;
	typedef typename TFunction::domain_type::grid_type grid_type;
	typedef typename TFunction::element_type element_type;

//	function id
	const size_t fct = u.fct_id_by_name(fctName);

//	get multigrid
	SmartPtr<grid_type> pMG = u.domain()->grid();

// 	attach error field
	typedef Attachment<number> ANumber;
	ANumber aGrad;
	pMG->template attach_to<element_type>(aGrad);
	MultiGrid::AttachmentAccessor<element_type, ANumber> aaGrad(*pMG, aGrad);
	
	ANumber aError;
	pMG->template attach_to<element_type>(aError);
	MultiGrid::AttachmentAccessor<element_type, ANumber> aaError(*pMG, aError);

// 	Compute error on elements
	if (u.local_finite_element_id(fct) == LFEID(LFEID::LAGRANGE, 1))
		ComputeGradientLagrange1(u, fct, aaGrad);
	else{
		if (u.local_finite_element_id(fct) == LFEID(LFEID::CROUZEIX_RAVIART, 1))
			ComputeGradientCrouzeixRaviart(u, fct, aaGrad);
		else{
			if (u.local_finite_element_id(fct) == LFEID(LFEID::PIECEWISE_CONSTANT, 0)){
				ComputeGradientPiecewiseConstant(u,fct,aaGrad);
			} else {
				UG_THROW("Non-supported finite element type " << u.local_finite_element_id(fct) << "\n");
			}
		}
	}
	computeGradientJump(u,aaGrad,aaError);
	
// 	Mark elements for refinement
	MarkElements(aaError, refiner, u, TOL, refineFrac, coarseFrac, maxLevel);

// 	detach error field
	pMG->template detach_from<element_type>(aError);
};

} // end namespace ug

#endif /* __H__UG__LIB_DISC__FUNCTION_SPACE__ERROR_INDICATOR__ */
