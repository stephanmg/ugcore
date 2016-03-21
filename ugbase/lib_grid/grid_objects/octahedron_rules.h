/*
 * Copyright (c) 2014-2015:  G-CSC, Goethe University Frankfurt
 * Author: Martin Stepniewski
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

#ifndef __H__UG__octahedron_rules__
#define __H__UG__octahedron_rules__

#include "common/math/ugmath.h"

namespace ug{
namespace oct_rules
{

////////////////////////////////////////////////////////////////////////////////
//	LOOKUP TABLES

const int NUM_VERTICES	= 6;
const int NUM_EDGES		= 12;
const int NUM_FACES		= 8;
const int NUM_TRIS		= 8;
const int NUM_QUADS		= 0;

/* in case of regular refinement an octahedron is subdivided into 14 elements,
 * 6 octahedrons and 8 tetrahedrons, resulting in 14 type-info plus
 * 6*6 octahedral vertex plus 4*8 tetrahedral vertex indices,
 * thus 82 MAX_NUM_INDS_OUT
 */
const int MAX_NUM_INDS_OUT = 82;//todo: this is just an estimate!

///	the local vertex indices of the given edge
const int EDGE_VRT_INDS[][2] = {	{0, 1}, {0, 2}, {0, 3}, {0, 4},
									{1, 2}, {2, 3}, {3, 4}, {4, 1},
									{1, 5}, {2, 5}, {3, 5}, {4, 5}};

///	the local vertex indices of the given face
const int FACE_VRT_INDS[][4] = {	{0, 1, 2, -1}, {0, 2, 3, -1}, {0, 3, 4, -1}, {0, 4, 1, -1},
									{1, 5, 2, -1}, {2, 5, 3, -1}, {3, 5, 4, -1}, {4, 5, 1, -1}};

///	the octhedrons bottom
const int BOTTOM_VERTEX = 0;

///	the octhedrons top
const int TOP_VERTEX = 5;

/** for each vertex, a pair containing the object type (0: vrt, 1: edge, 2: face)
 * and an index into the associated array, which describe the object which lies
 * on the opposite side of the tetrahedron, to a given vertex.*/
const int OPPOSED_OBJECT[][NUM_VERTICES] = {{0, 5}, {0, 3}, {0, 4}, {0, 1}, {0, 2}, {0, 0}};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//	NOTE: The lists below are all generated automatically

///	returns the j-th edge of the i-th face
const int FACE_EDGE_INDS[8][4] = {	{0, 4, 1, -1}, {1, 5, 2, -1}, {2, 6, 3, -1}, {3, 7, 0, -1},
									{8, 9, 4, -1}, {9, 10, 5, -1}, {10, 11, 6, -1}, {11, 8, 7, -1}};

///	tells whether the i-th face contains the j-th edge
const int FACE_CONTAINS_EDGE[][12] = {	{1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
										{0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0},
										{0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0},
										{1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0},
										{0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0},
										{0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0},
										{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1},
										{0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1}};

///	Associates the index of the connecting edge with each tuple of vertices.
/**	Use two vertex indices to index into this table to retrieve the index
 * of their connecting edge.
 */
const int EDGE_FROM_VRTS[6][6] = {	{-1, 0, 1, 2, 3, -1}, {0, -1, 4, -1, 7, 8}, {1, 4, -1, 5, -1, 9},
									{2, -1, 5, -1, 6, 10}, {3, 7, -1, 6, -1, 11}, {-1, 8, 9, 10, 11, -1}};

///	Associates the index of the connecting face with each triple of vertices.
/**	Use three vertex indices to index into this table to retrieve the index
 * of their connecting face.
 */
const int FACE_FROM_VRTS[6][6][6] = {	{{-1, -1, -1, -1, -1, -1}, {-1, -1, 0, -1, 3, -1}, {-1, 0, -1, 1, -1, -1}, {-1, -1, 1, -1, 2, -1}, {-1, 3, -1, 2, -1, -1}, {-1, -1, -1, -1, -1, -1}},
										{{-1, -1, 0, -1, 3, -1}, {-1, -1, -1, -1, -1, -1}, {0, -1, -1, -1, -1, 4}, {-1, -1, -1, -1, -1, -1}, {3, -1, -1, -1, -1, 7}, {-1, -1, 4, -1, 7, -1}},
										{{-1, 0, -1, 1, -1, -1}, {0, -1, -1, -1, -1, 4}, {-1, -1, -1, -1, -1, -1}, {1, -1, -1, -1, -1, 5}, {-1, -1, -1, -1, -1, -1}, {-1, 4, -1, 5, -1, -1}},
										{{-1, -1, 1, -1, 2, -1}, {-1, -1, -1, -1, -1, -1}, {1, -1, -1, -1, -1, 5}, {-1, -1, -1, -1, -1, -1}, {2, -1, -1, -1, -1, 6}, {-1, -1, 5, -1, 6, -1}},
										{{-1, 3, -1, 2, -1, -1}, {3, -1, -1, -1, -1, 7}, {-1, -1, -1, -1, -1, -1}, {2, -1, -1, -1, -1, 6}, {-1, -1, -1, -1, -1, -1}, {-1, 7, -1, 6, -1, -1}},
										{{-1, -1, -1, -1, -1, -1}, {-1, -1, 4, -1, 7, -1}, {-1, 4, -1, 5, -1, -1}, {-1, -1, 5, -1, 6, -1}, {-1, 7, -1, 6, -1, -1}, {-1, -1, -1, -1, -1, -1}}};

///	given two edges, the table returns the face, which contains both (or -1)
const int FACE_FROM_EDGES[][12] = {	{0, 0, -1, 3, 0, -1, -1, 3, -1, -1, -1, -1}, {0, 0, 1, -1, 0, 1, -1, -1, -1, -1, -1, -1},
									{-1, 1, 1, 2, -1, 1, 2, -1, -1, -1, -1, -1}, {3, -1, 2, 2, -1, -1, 2, 3, -1, -1, -1, -1},
									{0, 0, -1, -1, 0, -1, -1, -1, 4, 4, -1, -1}, {-1, 1, 1, -1, -1, 1, -1, -1, -1, 5, 5, -1},
									{-1, -1, 2, 2, -1, -1, 2, -1, -1, -1, 6, 6}, {3, -1, -1, 3, -1, -1, -1, 3, 7, -1, -1, 7},
									{-1, -1, -1, -1, 4, -1, -1, 7, 4, 4, -1, 7}, {-1, -1, -1, -1, 4, 5, -1, -1, 4, 4, 5, -1},
									{-1, -1, -1, -1, -1, 5, 6, -1, -1, 5, 5, 6}, {-1, -1, -1, -1, -1, -1, 6, 7, 7, -1, 6, 6}};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**	returns an array of integers, which contains the indices of the objects
 * resulting from the refinement of a octahedron.
 *
 *
 * \param newIndsOut	Array which has to be of size MAX_NUM_INDS_OUT.
 * 						When the algorithm is done, the array will contain
 * 						sequences of integers: {{gridObjectID, ind1, ind2, ...}, ...}.
 *						gridObjectID is a constant enumerated in GridObjectID and
 *						describes the type of the grid-object that is
 *						built from the following set of corner indices.
 * 						Old vertices are referenced by their local index. Vertices
 * 						created on an edge are indexed by the index of the edge +
 * 						NUM_VERTICES. If an inner vertex has to be created, it is
 * 						referenced by NUM_VERTICES + NUM_EDGES + NUM_FACES.
 *
 * \param newEdgeVrts	Array of size NUM_EDGES, which has to contain 1 for each
 * 						edge, which shall be refined and 0 for each edge, which
 * 						won't be refined.
 *
 * \param newCenterOut	If the refinement-rule requires a center vertex, then
 * 						this parameter will be set to true. If not, it is set to
 * 						false.
 *
 * \param corners		(optional) List of the six corner positions of the
 * 						octahedron. If it is specified, it is used during full
 * 						refinement (all edges marked), to determine the best (shortest)
 * 						diagonal along which the octahedrons are implicitly subdivided
 * 						into tetrahedrons. Corners are only considered during full
 * 						refinement and are thus irrelevant during recursive refinement
 * 						of other elements.
 *
 * \returns	the number of entries written to newIndsOut or 0, if the refinement
 * 			could not be performed.
 */
int Refine(int* newIndsOut, int* newEdgeVrts, bool& newCenterOut,
		   vector3* corners = NULL);

}//	end of namespace oct_rules
}//	end of namespace ug

#endif
