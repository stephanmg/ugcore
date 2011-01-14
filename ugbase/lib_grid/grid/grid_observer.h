//	created by Sebastian Reiter
//	s.b.reiter@googlemail.com
//	y08 m11 d19

#ifndef __H__LIBGRID__GRID_OBSERVERS__
#define __H__LIBGRID__GRID_OBSERVERS__

#include "common/types.h"

namespace ug
{
////////////////////////////////////////////////////////////////////////
//	predeclarations
class Grid;
class VertexBase;
class EdgeBase;
class Face;
class Volume;

/// \addtogroup lib_grid
/// @{

////////////////////////////////////////////////////////////////////////
//	Observer types
enum ObserverType
{
	OT_NONE = 0,
	OT_GRID_OBSERVER = 1,
	OT_VERTEX_OBSERVER = 2,
	OT_EDGE_OBSERVER = 4,
	OT_FACE_OBSERVER = 8,
	OT_VOLUME_OBSERVER = 16,
	OT_FULL_OBSERVER = OT_GRID_OBSERVER | OT_VERTEX_OBSERVER | OT_EDGE_OBSERVER |
						OT_FACE_OBSERVER | OT_VOLUME_OBSERVER
};

////////////////////////////////////////////////////////////////////////
//	GridObserver
/**
 * The grid observer defines an interface that can be specialized by
 * classes that want to be informed about changes in a grid.
 * If a class derives from GridObserver, it can be registered at a grid.
 * Registration is usually performed through a member function of the
 * observer class itself.
 * Most observers can only be registered at one grid at a time.
 *
 * Please note that methods of different observers are called in the
 * order in which they were registered at the grid. The only exception
 * are the vertex_to_be_erased, edge_to_be_erased, face_to_be_erased and
 * volume_to_be_erased. Those method are called in reverse order of
 * registration.
 */
class GridObserver
{
	public:
		virtual ~GridObserver()	{}

	//	grid callbacks
		virtual void grid_to_be_destroyed(Grid* grid)		{}
		virtual void elements_to_be_cleared(Grid* grid)		{}

	//	creation callbacks
	/**
	 *	\brief	Notified whenever a new element of the given type is created
	 *			in the given grid.
	 *
	 *	Creation callbacks are called in the order in which the GridObservers
	 * 	were registered at the given grid.
	 *
	 * 	if replacesParent is true, then pParent is of the same type as the
	 * 	created object.
	 * 	The method is called with replacesParent == true by
	 * 	Grid::create_and_replace methods.
	 *
	 *	Please note: If replacesParent == true, then a call to
	 * 	OBJECT_to_be_erased(grid, pParent, obj) will follow (OBJECT
	 *  and obj are pseudonyms for the concrete type).*/
	/// \{
		virtual void vertex_created(Grid* grid, VertexBase* vrt,
									GeometricObject* pParent = NULL,
									bool replacesParent = false)			{}

		virtual void edge_created(Grid* grid, EdgeBase* e,
									GeometricObject* pParent = NULL,
									bool replacesParent = false)			{}

		virtual void face_created(Grid* grid, Face* f,
									GeometricObject* pParent = NULL,
									bool replacesParent = false)			{}

		virtual void volume_created(Grid* grid, Volume* vol,
									GeometricObject* pParent = NULL,
									bool replacesParent = false)			{}
	///	\}


	//	erase callbacks
	///	Notified whenever an element of the given type is erased from the given grid.
	/**	Erase callbacks are called in reverse order in which the GridObservers
	 * 	were registered at the given grid.
	 *
	 * 	if replacedBy != NULL is true, then pParent is of the same type as the
	 * 	created object.
	 * 	The method is called with replacesParent == true by
	 * 	Grid::create_and_replace methods.
	 * \{ */
		virtual void vertex_to_be_erased(Grid* grid, VertexBase* vrt,
										 VertexBase* replacedBy = NULL)	{}

		virtual void edge_to_be_erased(Grid* grid, EdgeBase* e,
										 EdgeBase* replacedBy = NULL)	{}

		virtual void face_to_be_erased(Grid* grid, Face* f,
										 Face* replacedBy = NULL)	{}

		virtual void volume_to_be_erased(Grid* grid, Volume* vol,
										 Volume* replacedBy = NULL)	{}

	/**	\}	*/
};

/// @}

}//	end of namespace

#endif
