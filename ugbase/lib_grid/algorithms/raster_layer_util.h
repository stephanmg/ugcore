/*
 * Copyright (c) 2015:  G-CSC, Goethe University Frankfurt
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

#ifndef __H__UG_raster_layer_util
#define __H__UG_raster_layer_util

#include <utility>
#include <string>
#include <vector>
#include "lib_grid/algorithms/heightfield_util.h"

namespace ug{

struct RasterLayerDesc{
	RasterLayerDesc(const std::string& filename, number minHeight) : 
		m_filename(filename), m_minHeight(minHeight) 	{}

	const std::string& filename() const 	{return m_filename;}
	number min_height() const				{return m_minHeight;}

	private:
		std::string m_filename;
		number		m_minHeight;
};

typedef SmartPtr<RasterLayerDesc>	SPRasterLayerDesc;


class RasterLayers{
	public:
		struct layer_t{
			layer_t() : heightfield(), minHeight(SMALL) {}

			Heightfield heightfield;
			number		minHeight;
		};


		typedef RasterLayerDesc		LayerDesc;
		typedef SPRasterLayerDesc	SPLayerDesc;


	///	loads raster data from a list of .asc files.
	/**	layerDescs.front() represents the bottom of the lowest layer.
	 * layerDescs.top() represents the terrain surface. All data inbetween
	 * is interpreted as sorted layer-bottoms from bottom to top.
	 * \{ */
		void load_from_files(const std::vector<LayerDesc>& layerDescs);

		void load_from_files(const std::vector<SPLayerDesc>& layerDescs);
	/** \} */
	
	///	loads raster data from a list of .asc files.
	/**	filenames.front() represents the bottom of the lowest layer.
	 * filenames.top() represents the terrain surface. All data inbetween
	 * is interpreted as sorted layer-bottoms from bottom to top.
	 *
	 *	\param minLayerHeight	If the height of the layer at a given point is
	 * 							smaller than this value, then the layer is considered
	 *							to be non-existant at this point (i.e. has a hole)*/
		void load_from_files(const std::vector<std::string>& filenames,
							 number minLayerHeight);


		void resize(size_t newSize);
		
		size_t size() const			{return m_layers.size();}
		size_t num_layers() const	{return m_layers.size();}
		bool empty() const			{return m_layers.empty();}

		layer_t& operator[] (size_t i)				{return *m_layers[i];}
		const layer_t& operator[] (size_t i) const	{return *m_layers[i];}

		layer_t& 		layer (size_t i)			{return *m_layers[i];}
		const layer_t&	layer (size_t i) const		{return *m_layers[i];}

		const layer_t&	top () const				{return *m_layers.back();}

		Heightfield&		heightfield (size_t i)			{return layer(i).heightfield;}
		const Heightfield&	heightfield (size_t i) const	{return layer(i).heightfield;}

		void	set_min_height (size_t i, number h)		{layer(i).minHeight = h;}
		number	min_height (size_t i) const				{return layer(i).minHeight;}


	///	invalidates cells in lower levels which are too close to valid cells in higher levels
		void invalidate_flat_cells();

	///	invalidates cells that belong to a small lense regarding its horizontal area
		void invalidate_small_lenses(number minArea);

	///	removes small holes by expanding the layer in those regions to the specified height
		void remove_small_holes(number maxArea);

	///	sets invalid or flat cells to the value of the corresponding cell in the level above
	/** This method is somehow antithetical to 'invalidate_flat_cells', since it reassigns
	 * values to invalid cells which are shadowed by valid cells.*/
		void snap_cells_to_higher_layers();

	///	eliminates invalid cells by filling those cells with averages of neighboring valid cells
		void eliminate_invalid_cells();

	///	smoothens the values in each layer by averaging with neighboured values
		void blur_layers(number alpha, size_t numIterations);

	///	finds the first valid value at the given x-y-coordinate starting at the specified layer moving downwards.
	/** returns a pair containing the layer-index in which the valid value was found (first)
	 * and the value at the given coordinate in that layer. Returns -1 if no such layer
	 * was found.*/
		std::pair<int, number> trace_line_down(const vector2& c, size_t firstLayer) const;

	///	finds the first valid value at the given x-y-coordinate starting at the specified layer moving downwards.
	/** returns a pair containing the layer-index in which the valid value was found (first)
	 * and the value at the given coordinate in that layer. Returns -1 if no such layer
	 * was found.*/
		std::pair<int, number> trace_line_up(const vector2& c, size_t firstLayer) const;

	///	returns an index-pair of the layers above and below the specified point
	/** If there is no layer above or below, the associated component of the
	 *	returned is set to -1.*/
		std::pair<int, int> get_layer_indices(const vector3& c) const;

	private:
		std::vector<SmartPtr<layer_t> >	m_layers;
};

}//	end of namespace

#endif	//__H__UG_raster_layer_util
