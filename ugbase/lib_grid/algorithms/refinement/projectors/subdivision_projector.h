/*
 * Copyright (c) 2016:  G-CSC, Goethe University Frankfurt
 * Author: Sebastian Reiter
 * 
 * This file is part of UG4.
 * 
 * UG4 is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License version 3 (as published by the
 * Free Software Foundation) with the following additional attribution
 * requirements (according to LGPL/GPL v3 §7):
 * 
 * (1) The following notice must be displayed in the Appropriate Legal Notices
 * of covered and combined works: "Based on UG4 (www.ug4.org/license)".
 * 
 * (2) The following notice must be displayed at a prominent place in the
 * terminal output of covered works: "Based on UG4 (www.ug4.org/license)".
 * 
 * (3) The following bibliography is recommended for citation and must be
 * preserved in all covered files:
 * "Reiter, S., Vogel, A., Heppner, I., Rupp, M., and Wittum, G. A massively
 *   parallel geometric multigrid solver on hierarchically distributed grids.
 *   Computing and visualization in science 16, 4 (2013), 151-164"
 * "Vogel, A., Reiter, S., Rupp, M., Nägel, A., and Wittum, G. UG4 -- a novel
 *   flexible software system for simulating pde based models on high performance
 *   computers. Computing and visualization in science 16, 4 (2013), 165-179"
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 */

#ifndef __H__UG_subdivision_projector
#define __H__UG_subdivision_projector

#include <algorithm>
#include <vector>
#include "../refinement_projector.h"
#include "lib_grid/callbacks/basic_callbacks.h"

namespace ug{

///	Applies piecewise smooth loop subdivision rules
/**
 * \warning	The implementation assumes that vertices which are passed to
 *			'refinement_begins' still exist in the underlying grid when
 *			'refinement_ends' is called.
 */
class SubdivisionProjector : public RefinementProjector {
public:
	SubdivisionProjector ()	:
		m_cbIsCrease (Grid::edge_traits::callback(ConsiderAll()))
	{}
	
	SubdivisionProjector (Grid::edge_traits::callback cbIsCrease) :
		m_cbIsCrease (cbIsCrease)
	{}

	SubdivisionProjector (SPIGeometry3d geometry) :
		RefinementProjector (geometry),
		m_cbIsCrease (Grid::edge_traits::callback(ConsiderAll()))
	{}

	SubdivisionProjector (SPIGeometry3d geometry,
						  Grid::edge_traits::callback cbIsCrease) :
		RefinementProjector (geometry),
		m_cbIsCrease (cbIsCrease)
	{}

	virtual void refinement_begins(const ISubGrid& sg);
	virtual void refinement_ends(const ISubGrid& sg);

	virtual number new_vertex(Vertex* vrt, Edge* parent);
	// virtual number new_vertex(Vertex* vrt, Face* parent);
	// virtual number new_vertex(Vertex* vrt, Volume* parent);

	virtual void set_crease_callback (Grid::edge_traits::callback cbIsCrease)
	{
		m_cbIsCrease = cbIsCrease;
	}

protected:
	virtual bool is_crease_vertex(Vertex* vrt);
	virtual bool is_crease_edge(Edge* edge);

private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize( Archive& ar, const unsigned int version)
	{
		UG_EMPTY_BASE_CLASS_SERIALIZATION(SubdivisionProjector, RefinementProjector);
	}


	typedef std::vector<std::pair<Vertex*, vector3> >	new_pos_vec_t;

	Grid::edge_traits::callback	m_cbIsCrease;
	new_pos_vec_t				m_newPositions;
};

}//	end of namespace

#endif	//__H__UG_subdivision_projector
