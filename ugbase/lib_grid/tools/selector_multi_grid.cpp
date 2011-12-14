// created by Sebastian Reiter
// s.b.reiter@googlemail.com
// y10 m02 d15

#include <cassert>
#include "selector_multi_grid.h"
#include "lib_grid/multi_grid.h"

namespace ug
{

////////////////////////////////////////////////////////////////////////
MGSelector::MGSelector(uint supportedElements) :
	ISelector(supportedElements),
	m_aSharedEntryVRT("MGSelector_SharedListEntryVRT"),
	m_aSharedEntryEDGE("MGSelector_SharedListEntryEDGE"),
	m_aSharedEntryFACE("MGSelector_SharedListEntryFACE"),
	m_aSharedEntryVOL("MGSelector_SharedListEntryVOL")
{
	m_pMultiGrid = NULL;
}

MGSelector::MGSelector(MultiGrid& grid, uint supportedElements) :
	ISelector(supportedElements), m_pMultiGrid(NULL),
	m_aSharedEntryVRT("MGSelector_SharedListEntryVRT"),
	m_aSharedEntryEDGE("MGSelector_SharedListEntryEDGE"),
	m_aSharedEntryFACE("MGSelector_SharedListEntryFACE"),
	m_aSharedEntryVOL("MGSelector_SharedListEntryVOL")
{
	assign_grid(&grid);
}

MGSelector::~MGSelector()
{
	cleanup();
}

void MGSelector::cleanup()
{
//	delete all levels
	for(size_t i = 0; i < m_levels.size(); ++i)
		delete m_levels[i];

	m_levels.clear();

//	detach shared entry-attachments of section containers
	if(m_pMultiGrid){
		if(elements_are_supported(SE_VERTEX))
			m_pMultiGrid->detach_from_vertices(m_aSharedEntryVRT);
		if(elements_are_supported(SE_EDGE))
			m_pMultiGrid->detach_from_edges(m_aSharedEntryEDGE);
		if(elements_are_supported(SE_FACE))
			m_pMultiGrid->detach_from_faces(m_aSharedEntryFACE);
		if(elements_are_supported(SE_VOLUME))
			m_pMultiGrid->detach_from_volumes(m_aSharedEntryVOL);

		m_pMultiGrid = NULL;
	}

	BaseClass::set_grid(NULL);
}

void MGSelector::assign_grid(MultiGrid& grid)
{
	assign_grid(&grid);
}

void MGSelector::assign_grid(MultiGrid* grid)
{
//	if a grid already exists, we'll perform cleanup
	if(m_pMultiGrid)
		cleanup();

	m_pMultiGrid = grid;

//	attach shared entry-attachments to section containers
	if(m_pMultiGrid){
		if(elements_are_supported(SE_VERTEX))
			m_pMultiGrid->attach_to_vertices(m_aSharedEntryVRT);
		if(elements_are_supported(SE_EDGE))
			m_pMultiGrid->attach_to_edges(m_aSharedEntryVOL);
		if(elements_are_supported(SE_FACE))
			m_pMultiGrid->attach_to_faces(m_aSharedEntryEDGE);
		if(elements_are_supported(SE_VOLUME))
			m_pMultiGrid->attach_to_volumes(m_aSharedEntryFACE);
	}
	BaseClass::set_grid(grid);
}


void MGSelector::add_level()
{
//	adds a level and and initializes associated section containers.
	Level* pLvl = new Level;
	if(elements_are_supported(SE_VERTEX))
		pLvl->m_vertices.get_container().set_pipe(
				&m_pGrid->get_attachment_pipe<VertexBase>(), m_aSharedEntryVRT);
	if(elements_are_supported(SE_EDGE))
		pLvl->m_edges.get_container().set_pipe(
				&m_pGrid->get_attachment_pipe<EdgeBase>(), m_aSharedEntryEDGE);
	if(elements_are_supported(SE_FACE))
		pLvl->m_faces.get_container().set_pipe(
				&m_pGrid->get_attachment_pipe<Face>(), m_aSharedEntryFACE);
	if(elements_are_supported(SE_VOLUME))
		pLvl->m_volumes.get_container().set_pipe(
				&m_pGrid->get_attachment_pipe<Volume>(), m_aSharedEntryVOL);

	m_levels.push_back(pLvl);
}

void MGSelector::clear_lists()
{
	for(size_t i = 0; i < m_levels.size(); ++i)
		delete m_levels[i];
	m_levels.clear();
}

void MGSelector::clear()
{
	clear<VertexBase>();
	clear<EdgeBase>();
	clear<Face>();
	clear<Volume>();
}

void MGSelector::clear(int level)
{
	clear<VertexBase>(level);
	clear<EdgeBase>(level);
	clear<Face>(level);
	clear<Volume>(level);
}

void MGSelector::add_to_list(VertexBase* elem)
{
	const int level = m_pMultiGrid->get_level(elem);
	section_container<VertexBase>(level).insert(elem,
								elem->shared_pipe_section());
}

void MGSelector::add_to_list(EdgeBase* elem)
{
	const int level = m_pMultiGrid->get_level(elem);
	section_container<EdgeBase>(level).insert(elem,
								elem->shared_pipe_section());
}

void MGSelector::add_to_list(Face* elem)
{
	const int level = m_pMultiGrid->get_level(elem);
	section_container<Face>(level).insert(elem,
								elem->shared_pipe_section());
}

void MGSelector::add_to_list(Volume* elem)
{
	const int level = m_pMultiGrid->get_level(elem);
	section_container<Volume>(level).insert(elem,
								elem->shared_pipe_section());
}	

void MGSelector::erase_from_list(VertexBase* elem)
{
	const int level = m_pMultiGrid->get_level(elem);
	section_container<VertexBase>(level).erase(get_iterator(elem),
						elem->shared_pipe_section());
}
void MGSelector::erase_from_list(EdgeBase* elem)
{
	const int level = m_pMultiGrid->get_level(elem);
	section_container<EdgeBase>(level).erase(get_iterator(elem),
						elem->shared_pipe_section());
}
void MGSelector::erase_from_list(Face* elem)
{
	const int level = m_pMultiGrid->get_level(elem);
	section_container<Face>(level).erase(get_iterator(elem),
						elem->shared_pipe_section());
}

void MGSelector::erase_from_list(Volume* elem)
{
	const int level = m_pMultiGrid->get_level(elem);
	section_container<Volume>(level).erase(get_iterator(elem),
						elem->shared_pipe_section());
}

//	geometric-object-collection
GeometricObjectCollection MGSelector::get_geometric_objects()
{
	uint numLevels = num_levels();
	GeometricObjectCollection goc(numLevels);
	
	for(uint i = 0; i < numLevels; ++i)
	{
		goc.add_level(	&m_levels[i]->m_vertices,
						&m_levels[i]->m_edges,
						&m_levels[i]->m_faces,
						&m_levels[i]->m_volumes);
	}
	
	return goc;
}

/*
void MGSelector::registered_at_grid(Grid* grid)
{
	m_pMultiGrid = dynamic_cast<MultiGrid*>(grid);
	assert(m_pMultiGrid && "bad grid type!");
	ISelector::registered_at_grid(grid);
}

void MGSelector::unregistered_from_grid(Grid* grid)
{
	ISelector::unregistered_from_grid(grid);
	clear_lists();
}
*/
void MGSelector::grid_to_be_destroyed(Grid* grid)
{
	ISelector::grid_to_be_destroyed(grid);
	m_pMultiGrid = NULL;
	clear_lists();
}

}//	end of namespace
