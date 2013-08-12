// created by Sebastian Reiter
// s.b.reiter@googlemail.com
// 24.11.2011 (m,d,y)
 
#include <sstream>
#include "surface_view.h"
#include "common/assert.h"
#include "lib_grid/parallelization/util/compol_boolmarker.h"
#include "lib_grid/file_io/file_io.h"

#ifdef UG_PARALLEL
	#include "pcl/pcl_interface_communicator.h"
	#include "pcl/pcl_process_communicator.h"
	#include "lib_grid/parallelization/util/compol_copy_attachment.h"
#endif

using namespace std;

namespace ug{

///	adds marking at extracting side
//todo:	change to ComPol_AttachmentBinaryOr
template <class TLayout>
class ComPol_GatherSurfaceStates : public pcl::ICommunicationPolicy<TLayout>
{
	public:
		typedef TLayout								Layout;
		typedef typename Layout::Type				GeomObj;
		typedef typename Layout::Element			Element;
		typedef typename Layout::Interface			Interface;
		typedef typename Interface::const_iterator	InterfaceIter;

	///	Construct the communication policy with a ug::BoolMarker.
		ComPol_GatherSurfaceStates(MultiGrid& mg,
						MultiElementAttachmentAccessor<SurfaceView::ASurfaceState>& aaElemSurfState)
			 :	m_mg(mg), m_aaESS(aaElemSurfState)
		{}

		virtual ~ComPol_GatherSurfaceStates()	{}

		virtual int get_required_buffer_size(const Interface& interface)
		{
			return interface.size() * sizeof(byte);
		}

	///	write surface state for each entry
		virtual bool collect(ug::BinaryBuffer& buff, const Interface& intfc)
		{
		//	write the entry indices of marked elements.
			for(InterfaceIter iter = intfc.begin(); iter != intfc.end(); ++iter)
			{
				Element elem = intfc.get_element(iter);
				byte val = m_aaESS[elem].get();
				buff.write((char*)&val, sizeof(byte));
			}
			return true;
		}

	///	reads marks from the given stream
		virtual bool extract(ug::BinaryBuffer& buff, const Interface& intfc)
		{
			for(InterfaceIter iter = intfc.begin(); iter != intfc.end(); ++iter)
			{
				Element elem = intfc.get_element(iter);
				byte nv;
				buff.read((char*)&nv, sizeof(byte));
				if(nv > m_aaESS[elem])
					m_aaESS[elem] = nv;
			}
			return true;
		}

	protected:
		MultiGrid&															m_mg;
		MultiElementAttachmentAccessor<SurfaceView::ASurfaceState>	m_aaESS;
};

////////////////////////////////////////////////////////////////////////////////
//	Create Surface View
////////////////////////////////////////////////////////////////////////////////
template <class TElem>
bool SurfaceView::
is_local_surface_view_element(TElem* elem)
{
	#ifdef UG_PARALLEL
		return !(m_pMG->has_children(elem)
				 || m_distGridMgr->is_ghost(elem));
				//|| m_distGridMgr->contains_status(elem, ES_V_MASTER));
	#else
		return !m_pMG->has_children(elem);
	#endif
}

void SurfaceView::
refresh_surface_states()
{
//todo	we need a global max-dim!!! (empty processes have to do the right thing, too)
	int maxElem = -1;
	if(m_pMG->num<Volume>() > 0)
		maxElem = VOLUME;
	else if(m_pMG->num<Face>() > 0)
		maxElem = FACE;
	else if(m_pMG->num<EdgeBase>() > 0)
		maxElem = EDGE;
	else if(m_pMG->num<VertexBase>() > 0)
		maxElem = VERTEX;

	#ifdef UG_PARALLEL
		pcl::ProcessCommunicator pc;
		maxElem = pc.allreduce(maxElem, PCL_RO_MAX);
	#endif

	switch(maxElem){
		case VOLUME:
			refresh_surface_states<Volume>();
			break;
		case FACE:
			refresh_surface_states<Face>();
			break;
		case EDGE:
			refresh_surface_states<EdgeBase>();
			break;
		case VERTEX:
			refresh_surface_states<VertexBase>();
			break;
		default: break;
	}
}

//static void DebugSave(MultiGrid& mg, SurfaceView& sv, const char* prefix)
//{
//	stringstream ss;
//	ss << prefix;
//#ifdef UG_PARALLEL
//	ss << "_p" << pcl::GetProcRank();
//#endif
//	ss << ".ugx";
//
//	SaveSurfaceViewTransformed(mg, sv, ss.str().c_str(), 0.1);
//}

template <class TElem>
void SurfaceView::
refresh_surface_states()
{
//	some typedefs
	typedef typename geometry_traits<TElem>::iterator ElemIter;

	MultiGrid& mg = *m_pMG;

//	reset surface states of all elements. Initially, we'll set all states to hidden
	SetAttachmentValues(m_aaSurfState, mg.begin<VertexBase>(), mg.end<VertexBase>(), UNASSIGNED);
	SetAttachmentValues(m_aaSurfState, mg.begin<EdgeBase>(), mg.end<EdgeBase>(), UNASSIGNED);
	SetAttachmentValues(m_aaSurfState, mg.begin<Face>(), mg.end<Face>(), UNASSIGNED);
	SetAttachmentValues(m_aaSurfState, mg.begin<Volume>(), mg.end<Volume>(), UNASSIGNED);

//	iterate through all levels of the mgsh
	for(size_t level = 0; level < mg.num_levels(); ++level){

	//	iterate through all elements on that level
		for(ElemIter iter = mg.begin<TElem>(level);
			iter != mg.end<TElem>(level); ++iter)
		{
			TElem* elem = *iter;

			if(is_local_surface_view_element(elem)){
				surface_state(elem).set(PURE_SURFACE);
				mark_sides_as_surface_or_shadow<TElem, typename TElem::side>(elem);
			}
		}
	}

//	make sure that all constrained elements are surface view members
//	(think e.g. of constrained ghost elements)
	for(ConstrainedTriangleIterator iter = mg.begin<ConstrainedTriangle>();
		iter != mg.end<ConstrainedTriangle>(); ++iter)
	{
		Face* elem = *iter;
		surface_state(elem).set(SHADOWING);
		mark_sides_as_surface_or_shadow<Face, EdgeBase>(elem, SHADOWING);
		if(GeometricObject* p = m_pMG->get_parent(elem))
			surface_state(p).set(SHADOW_NONCOPY);
	}
	for(ConstrainedQuadrilateralIterator iter = mg.begin<ConstrainedQuadrilateral>();
		iter != mg.end<ConstrainedQuadrilateral>(); ++iter)
	{
		Face* elem = *iter;
		surface_state(elem).set(SHADOWING);
		mark_sides_as_surface_or_shadow<Face, EdgeBase>(elem, SHADOWING);
		if(GeometricObject* p = m_pMG->get_parent(elem))
			surface_state(p).set(SHADOW_NONCOPY);
	}
	for(ConstrainedEdgeIterator iter = mg.begin<ConstrainedEdge>();
		iter != mg.end<ConstrainedEdge>(); ++iter)
	{
		EdgeBase* elem = *iter;
		surface_state(elem).set(SHADOWING);
		mark_sides_as_surface_or_shadow<EdgeBase, VertexBase>(elem, SHADOWING);
		if(GeometricObject* p = m_pMG->get_parent(elem))
			surface_state(p).set(SHADOW_NONCOPY);
	}

//	we now have to mark all shadowing elements.
//	Only low dimensional elements can be shadows.
//	Perform assignment on higher dimensional elements first, since lower
//	dimensional elements may shadow higher dimensional elements...
	if(TElem::HAS_SIDES){
		mark_shadowing<typename TElem::side>(true);
	}

//	DebugSave(*m_pMG, *this, "surf_03_shadowings_marked");

//	communicate states between all processes
	adjust_parallel_surface_states<VertexBase>();
	adjust_parallel_surface_states<EdgeBase>();
	adjust_parallel_surface_states<Face>();
	adjust_parallel_surface_states<Volume>();
}

template <class TElem, class TSide>
void SurfaceView::
mark_sides_as_surface_or_shadow(TElem* elem, byte surfaceState)
{
	if(!TElem::HAS_SIDES)
		return;

	typename Grid::traits<TSide>::secure_container	sides;

	m_pMG->associated_elements(sides, elem);
	for(size_t i = 0; i < sides.size(); ++i){
		TSide* s = sides[i];
		if(surface_state(s) == UNASSIGNED){
			size_t numChildren = m_pMG->num_children<TSide>(s);
			if(numChildren == 0)
				surface_state(s).set(surfaceState);
			else if(numChildren == 1)
				surface_state(s).set(SHADOW_COPY);
			else
				surface_state(s).add(SHADOW_NONCOPY);
		}
	}

	if(TSide::HAS_SIDES)
		mark_sides_as_surface_or_shadow<TElem, typename TSide::side>(elem);
}

template <class TElem>
void SurfaceView::
mark_shadowing(bool markSides)
{
	typedef typename Grid::traits<TElem>::iterator TIter;

	MultiGrid& mg = *m_pMG;

	for(size_t lvl = 1; lvl < mg.num_levels(); ++lvl){
		for(TIter iter = mg.begin<TElem>(lvl); iter != mg.end<TElem>(lvl); ++iter)
		{
			TElem* e = *iter;
			if(!surface_state(e).contains(PURE_SURFACE))
				continue;

			GeometricObject* p = mg.get_parent(e);
			if(surface_state(p).contains(SHADOW_COPY) || surface_state(p).contains(SHADOW_NONCOPY)){
				surface_state(e).set(SHADOWING);
			}
		}
	}

	if(markSides && TElem::HAS_SIDES){
		mark_shadowing<typename TElem::side>(markSides);
	}
}

template <class TElem>
void SurfaceView::
adjust_parallel_surface_states()
{
	#ifdef UG_PARALLEL
		typedef typename GridLayoutMap::Types<TElem>::Layout	Layout;

		GridLayoutMap& glm = m_distGridMgr->grid_layout_map();
		ComPol_GatherSurfaceStates<Layout>	cpAdjust(*m_pMG, m_aaSurfState);
		pcl::InterfaceCommunicator<Layout> com;
		com.exchange_data(glm, INT_H_SLAVE, INT_H_MASTER, cpAdjust);
	//	the v-communication is only required if constrained ghosts can be surface elements.
		com.exchange_data(glm, INT_V_MASTER, INT_V_SLAVE, cpAdjust);
		com.communicate();

		ComPol_CopyAttachment<Layout, ASurfaceState> cpCopyStates(*m_pMG, m_aSurfState);
		com.exchange_data(glm, INT_H_MASTER, INT_H_SLAVE, cpCopyStates);
		com.communicate();

	//todo:	communicate final marks to v-masters? (if one wants to iterate over ghosts...)
	#endif
}


////////////////////////////////////////////////////////////////////////////////
// SurfaceView
////////////////////////////////////////////////////////////////////////////////

SurfaceView::SurfaceView(SmartPtr<MGSubsetHandler> spMGSH,
                         bool adaptiveMG) :
	m_spMGSH(spMGSH),
	m_adaptiveMG(adaptiveMG),
	m_pMG(m_spMGSH->multi_grid()),
	m_distGridMgr(m_spMGSH->multi_grid()->distributed_grid_manager())
{
	UG_ASSERT(m_pMG, "A MultiGrid has to be assigned to the given subset handler");

	m_pMG->attach_to_all_dv(m_aSurfState, 0);
	m_aaSurfState.access(*m_pMG, m_aSurfState);

	refresh_surface_states();
}

SurfaceView::~SurfaceView()
{
	m_pMG->detach_from_all(m_aSurfState);
}

}// end of namespace