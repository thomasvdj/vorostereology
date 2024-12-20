/*
 * vpp.h : pyvoro C++ stdlib interface to voro++
 *
 * this file provides pure C++ stdlib wrapper functions for creating, 
 * manipulating, performing computations on and exporting the voro++ complex 
 * C++ classes (the advanced use of virtual functions makes direct cython 
 * manupulation impossible, sadly.)
 *
 * this extension to voro++ is released under the original modified BSD license
 * and constitutes an Extension to the original project.
 *
 * Copyright (c) Joe Jordan 2012
 * contact: <joe.jordan@imperial.ac.uk> or <tehwalrus@h2j9k.org>
 * 
 */
 
#ifndef __VPP_H__
#define __VPP_H__ 1

#include <vector>

void compute_centroids(const std::vector<double> &weights, 
    double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
    const std::vector<double> &x_, const std::vector<double> &y_, const std::vector<double> &z_, int nx_, int ny_, int nz_, double centroids[]);

void compute_volumes(const std::vector<double> &weights, 
    double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
    const std::vector<double> &x_, const std::vector<double> &y_, const std::vector<double> &z_, int nx_, int ny_, int nz_, double volumes[]);

void compute_surface_areas(const std::vector<double> &weights, 
    double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
    const std::vector<double> &x_, const std::vector<double> &y_, const std::vector<double> &z_, int nx_, int ny_, int nz_, double areas[]);

void compute_hessian(const std::vector<double> &weights, 
    double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
    const std::vector<double> &x_, const std::vector<double> &y_, const std::vector<double> &z_, int nx_, int ny_, int nz_,
    std::vector<int> &row_coordinates, std::vector<int> &column_coordinates, std::vector<double> &hessian_entries);

void compute_voronoi_tessellation(const std::vector<double> &weights, 
    double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
    const std::vector<double> &x_, const std::vector<double> &y_, const std::vector<double> &z_, int nx_, int ny_, int nz_,
    std::vector<std::vector< std::vector<int> > > &vertices_indices, std::vector<std::vector <double> > &vertices, std::vector<std::vector<int> > &neighbors, std::vector<bool> &computed_cells);

void compute_voronoi_tessellation_bounded(const std::vector<double> &weights, 
    double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
    const std::vector<double> &x_, const std::vector<double> &y_, const std::vector<double> &z_, int nx_, int ny_, int nz_, 
    const std::vector<bool> &cells_to_compute, const std::vector<bool> &x_outside, const std::vector<bool> &y_outside, const std::vector<bool> &z_outside,
    std::vector<std::vector<std::vector<int> > > &vertices_indices, std::vector<std::vector <double> > &vertices, std::vector<std::vector<int> > &neighbors, std::vector<bool> &computed_cells);

void compute_section(const std::vector<double> &weights, 
    double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
    const std::vector<double> &x_, const std::vector<double> &y_, const std::vector<double> &z_, int nx_, int ny_, int nz_, 
    double a, double b, double c, double x_0, double y_0, double z_0, std::vector<int> &section_indices, std::vector<double> &areas, std::vector<std::vector<double> > &section_vertices);
  
void compute_section_bounded(const std::vector<double> &weights, 
    double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
    const std::vector<double> &x_, const std::vector<double> &y_, const std::vector<double> &z_, int nx_, int ny_, int nz_, 
    double a, double b, double c, double x_0, double y_0, double z_0, 
    const std::vector<bool> &cells_to_compute, const std::vector<bool> &x_outside, const std::vector<bool> &y_outside, const std::vector<bool> &z_outside,
    std::vector<int> &section_indices, std::vector<double> &areas, std::vector<std::vector<double> > &section_vertices);
  
void compute_num_fragments(const std::vector<double> &weights, 
    double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
    const std::vector<double> &x_, const std::vector<double> &y_, const std::vector<double> &z_, int nx_, int ny_, int nz_, std::vector<std::vector<bool> > &cell_sections);

#endif
