/*
 * vpp.cpp : pyvoro C interface to voro++ (implementation)
 *
 * this file provides pure C wrapper functions for creating, manipulating,
 * performing computations on and exporting the voro++ C++ classes.
 *
 * this extension to voro++ is released under the original modified BSD license
 * and constitutes an Extension to the original project.
 *
 * Copyright (c) Joe Jordan 2012
 * contact: <joe.jordan@imperial.ac.uk> or <tehwalrus@h2j9k.org>
 *
 */

#include "vpp.h"
#include "../src/voro++.hh"
#include <numeric>    
#include <limits>
#include <algorithm>  
#include <math.h>
#include <unordered_map>
using namespace voro;
using namespace std;


void compute_hessian(const vector<double> &weights, 
    double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
    const vector<double> &x_, const vector<double> &y_, const vector<double> &z_, int nx_, int ny_, int nz_,
    vector<int> &row_coordinates, vector<int> &column_coordinates, vector<double> &hessian_entries){

    double x_size = bx_ - ax_;
    double y_size = by_ - ay_;
    double z_size = bz_ - az_;
    double x_rsize = 1.0 / x_size;
    double y_rsize = 1.0 / y_size;
    double z_rsize = 1.0 / z_size;
    bool periodic = ((bool)px_ || (bool)py_) || (bool)pz_;

    container_poly con(ax_, bx_, ay_, by_, az_, bz_, nx_, ny_, nz_, (bool)px_, (bool)py_, (bool)pz_, 6);
    c_loop_all cla(con); 
    voronoicell_neighbor cell;

    int i;

    for (i = 0; i < n_; i++) {
        con.put(i, x_[i], y_[i], z_[i], sqrt(weights[i]));
    }
    
    if(cla.start()) do if (con.compute_cell(cell, cla)) {
        i = cla.pid();
            
        vector<int> cell_neighbors;
        cell.neighbors(cell_neighbors);
        vector<double> cell_areas;
        cell.face_areas(cell_areas);
        int num_neighbors = cell_neighbors.size();

        double summed_entries = 0.0;
        double current_entry = 0.0;
        double d1, d2, d3;
        
        // The periodic case is slightly more complicated. In the non-periodic case each cell face corresponds to a unique neighbor,
        // this is no longer true in the periodic case. If multiple faces of a cell correspond to the same neighbor we sum the areas of these faces.
        // an unordered_map is used to keep track of neighbors.
        if (periodic){ 
            unordered_map<int, int> past_neighbors;
            for (int j = 0; j < num_neighbors; j++){
                auto search = past_neighbors.find(cell_neighbors[j]);
                // If we have already had a cell face with this neighbor we add the area of the face to the previous one(s). 
                if (search != past_neighbors.end()){
                    hessian_entries[past_neighbors[cell_neighbors[j]]] += cell_areas[j];
                } else {
                    // If we have not had a cell face with this neighbor we create a new entry in the unordered_map.
                    if (cell_neighbors[j] != i){  // It is even possible that a cell is its own neighbor, if this is the case for a cell face, skip this face.
                        int new_idx = hessian_entries.size();
                        hessian_entries.push_back(cell_areas[j]);
                        row_coordinates.push_back(i);
                        column_coordinates.push_back(cell_neighbors[j]);
                        past_neighbors[cell_neighbors[j]] = new_idx;
                    }
                }
            }
            for(const pair<int, int>& ix : past_neighbors) {
                // In the periodic case we need the periodic distance between generator points instead of the usual Euclidean distance.
                // According to wiki this is an efficient method for computing this distance.
                d1 = x_[ix.first] - x_[i];
                d1 -= x_size * nearbyint(d1 * x_rsize);
                d2 = y_[ix.first] - y_[i];
                d2 -= y_size * nearbyint(d2 * y_rsize);
                d3 = z_[ix.first] - z_[i];
                d3 -= z_size * nearbyint(d3 * z_rsize);
                current_entry = 0.5 * hessian_entries[ix.second] / sqrt(d1*d1 + d2*d2 + d3*d3);

                summed_entries -= current_entry;
                hessian_entries[ix.second] = current_entry;
            }
            past_neighbors.clear();
        } else {
            // The non-periodic case, fairly straightforward.
            for (int j = 0; j < num_neighbors; j++){
                if (cell_neighbors[j] >= 0){
                    row_coordinates.push_back(i);
                    column_coordinates.push_back(cell_neighbors[j]);
                    
                    d1 = x_[i] - x_[cell_neighbors[j]];
                    d2 = y_[i] - y_[cell_neighbors[j]];
                    d3 = z_[i] - z_[cell_neighbors[j]];
                    
                    current_entry = 0.5 * cell_areas[j]/sqrt(d1*d1 + d2*d2 + d3*d3);
                    summed_entries -= current_entry;
                    hessian_entries.push_back(current_entry);
                }
            }
        }
        row_coordinates.push_back(i);
        column_coordinates.push_back(i);
        hessian_entries.push_back(summed_entries);
    } while (cla.inc());

}

void compute_volumes(const vector<double> &weights, 
    double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
    const vector<double> &x_, const vector<double> &y_, const vector<double> &z_, int nx_, int ny_, int nz_, double volumes[]){

    container_poly con(ax_, bx_, ay_, by_, az_, bz_, nx_, ny_, nz_, (bool)px_, (bool)py_, (bool)pz_, 6);
    int i;
    c_loop_all cla(con); 
    voronoicell_neighbor cell;

    for (i = 0; i < n_; i++) {
        con.put(i, x_[i], y_[i], z_[i], sqrt(weights[i]));
    }

    if(cla.start()) do {
        i = cla.pid();
        if (con.compute_cell(cell, cla)) {
            volumes[i] = cell.volume();
        } 
    } while (cla.inc());
}


void compute_surface_areas(const vector<double> &weights, 
    double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
    const vector<double> &x_, const vector<double> &y_, const vector<double> &z_, int nx_, int ny_, int nz_, double areas[]){

    container_poly con(ax_, bx_, ay_, by_, az_, bz_, nx_, ny_, nz_, (bool)px_, (bool)py_, (bool)pz_, 6);
    int i;
    c_loop_all cla(con); 
    voronoicell_neighbor cell;

    for (i = 0; i < n_; i++) {
        con.put(i, x_[i], y_[i], z_[i], sqrt(weights[i]));
    }

    if(cla.start()) do {
        i = cla.pid();
        if (con.compute_cell(cell, cla)) {
            areas[i] = cell.surface_area();
        } 
    } while (cla.inc());
}


void compute_centroids(const vector<double> &weights, 
    double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
    const vector<double> &x_, const vector<double> &y_, const vector<double> &z_, int nx_, int ny_, int nz_, double centroids[]) {

    container_poly con(ax_, bx_, ay_, by_, az_, bz_, nx_, ny_, nz_, (bool)px_, (bool)py_, (bool)pz_, 6);
    int i;
    double x, y, z, r;
    c_loop_all cla(con); 
    voronoicell_neighbor cell;

    for (i = 0; i < n_; i++) {
        con.put(i, x_[i], y_[i], z_[i], sqrt(weights[i]));
    }

    if(cla.start()) do {
        cla.pos(i, x, y, z, r);
        if (con.compute_cell(cell, cla)) {
            cell.centroid(centroids[i], centroids[i + n_], centroids[i + (2 * n_)]);
            centroids[i] += x;
            centroids[i + n_] += y;
            centroids[i + (2 * n_)] += z;
        } else {
            centroids[i] = x; centroids[i + n_] = y; centroids[i + (2 * n_)] = z;
        }
    } while (cla.inc());
}


void compute_voronoi_tessellation(const vector<double> &weights, 
    double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
    const vector<double> &x_, const vector<double> &y_, const vector<double> &z_, int nx_, int ny_, int nz_,
    vector<vector< vector<int> > > &vertices_indices, vector<vector <double> > &vertices, vector<vector<int> > &neighbors, vector<bool> &computed_cells) {
    
    container_poly con(ax_, bx_, ay_, by_, az_, bz_, nx_, ny_, nz_, (bool)px_, (bool)py_, (bool)pz_, 6);
    c_loop_all cla(con); 
    voronoicell_neighbor cell;

    vertices_indices.resize(n_);
    vertices.resize(n_);
    neighbors.resize(n_);
    computed_cells.resize(n_);

    int i;

    for (i = 0; i < n_; i++) {
        con.put(i, x_[i], y_[i], z_[i], sqrt(weights[i]));
    }
    
    if(cla.start()) do {
        i = cla.pid();
        if (con.compute_cell(cell, cla)) {
            vector<int> cell_face_vertices;
            vector<int> cell_neighbors;
            vector<double> cell_vertices;

            cell.neighbors(cell_neighbors);
            cell.face_vertices(cell_face_vertices);
            cell.vertices(x_[i], y_[i], z_[i], cell_vertices);

            vector<vector<int> > cell_vertices_indices(cell_neighbors.size());
            int next_idx = 0;
            for (int j = 0; j < cell_neighbors.size(); j++){
                vector<int> face_vertices_indices(cell_face_vertices[next_idx]);
                for (int k = 0; k < cell_face_vertices[next_idx]; k++){
                    face_vertices_indices[k] = cell_face_vertices[next_idx + 1 + k];
                }
                cell_vertices_indices[j] = face_vertices_indices;
                next_idx += cell_face_vertices[next_idx] + 1;
            }

            neighbors[i] = cell_neighbors;
            vertices[i] = cell_vertices;
            vertices_indices[i] = cell_vertices_indices;
            computed_cells[i] = true;
        } else {
            computed_cells[i] = false;
        }
    } while (cla.inc());
}


void compute_voronoi_tessellation_bounded(const vector<double> &weights, 
    double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
    const vector<double> &x_, const vector<double> &y_, const vector<double> &z_, int nx_, int ny_, int nz_, 
    const vector<bool> &cells_to_compute, const vector<bool> &x_outside, const vector<bool> &y_outside, const vector<bool> &z_outside,
    vector<vector< vector<int> > > &vertices_indices, vector<vector <double> > &vertices, vector<vector<int> > &neighbors, vector<bool> &computed_cells) {

    container_poly con(ax_, bx_, ay_, by_, az_, bz_, nx_, ny_, nz_, (bool)px_, (bool)py_, (bool)pz_, 6);
    c_loop_all cla(con); 
    voronoicell_neighbor cell;

    vertices_indices.resize(n_);
    vertices.resize(n_);
    neighbors.resize(n_);
    computed_cells.resize(n_);

    int i;
    double x,y,z,r;
    bool cut_ax, cut_bx, cut_ay, cut_by, cut_az, cut_bz;
    
    double tmp;
    double c1_ax, c2_ax, c3_ax, c1_bx, c2_bx, c3_bx;
    double c1_ay, c2_ay, c3_ay, c1_by, c2_by, c3_by;
    double c1_az, c2_az, c3_az, c1_bz, c2_bz, c3_bz;
    double rsq_ax, rsq_bx, rsq_ay, rsq_by, rsq_az, rsq_bz;
    int pid_ax, pid_ay, pid_az, pid_bx, pid_by, pid_bz;
    int sgn;

    for (i = 0; i < n_; i++) {
        con.put(i, x_[i], y_[i], z_[i], sqrt(weights[i]));
    }

    if(cla.start()) do {
        cla.pos(i, x, y, z, r);
        computed_cells[i] = false;
        if (cells_to_compute[i]){
            if (con.compute_cell(cell, cla)) {
                pid_ax = -1; pid_bx = -2; pid_ay = -3; pid_by = -4; pid_az = -5; pid_bz = -6;
                bool computed = true;
                cut_ax = false; cut_bx = false; cut_ay = false; cut_by = false; cut_az = false; cut_bz = false;

                // intersect with x = con.ax
                tmp = -2*(x - con.ax);
                sgn = 1;
                if (tmp < 0) sgn = -1;
                c1_ax = tmp; c2_ax = 0; c3_ax = 0; rsq_ax = c1_ax*c1_ax;
                if (rsq_ax < numeric_limits<double>::epsilon()){
                    c1_ax = sgn; rsq_ax = 0;
                }
                if (cell.plane_intersects(c1_ax, c2_ax, c3_ax, rsq_ax)){
                cut_ax = true;
                }

                // intersect with y = con.ay
                tmp = -2*(y - con.ay);
                sgn = 1;
                if (tmp < 0) sgn = -1;
                c1_ay = 0; c2_ay = tmp; c3_ay = 0; rsq_ay = c2_ay*c2_ay;
                if (rsq_ay < numeric_limits<double>::epsilon()){
                    c2_ay = sgn; rsq_ay = 0;
                }
                if (cell.plane_intersects(c1_ay, c2_ay, c3_ay, rsq_ay)){
                cut_ay = true;
                }

                // intersect with z = con.az
                tmp = -2*(z - con.az);
                sgn = 1;
                if (tmp < 0) sgn = -1;
                c1_az = 0; c2_az = 0; c3_az = tmp; rsq_az = c3_az*c3_az;
                if (rsq_az < numeric_limits<double>::epsilon()){
                    c3_az = sgn; rsq_az = 0;
                }
                if (cell.plane_intersects(c1_az, c2_az, c3_az, rsq_az)){
                cut_az = true;
                }

                // intersect with x = con.bx
                tmp = -2*(x-con.bx);
                sgn = 1;
                if (tmp < 0) sgn = -1;
                c1_bx = tmp; c2_bx = 0; c3_bx = 0; rsq_bx = c1_bx*c1_bx;
                if (rsq_bx < numeric_limits<double>::epsilon()){
                    c1_bx = sgn; rsq_bx = 0;
                }
                if (cell.plane_intersects(c1_bx, c2_bx, c3_bx, rsq_bx)){
                cut_bx = true;
                }

                // intersect with y = con.by
                tmp = -2*(y-con.by);
                sgn = 1;
                if (tmp < 0) sgn = -1;
                c1_by = 0; c2_by = tmp; c3_by = 0; rsq_by = c2_by*c2_by;
                if (rsq_by < numeric_limits<double>::epsilon()){
                    c2_by = sgn; rsq_by = 0;
                }
                if (cell.plane_intersects(c1_by, c2_by, c3_by, rsq_by)){
                cut_by = true;
                }

                // intersect with z = con.bz
                tmp = -2*(z-con.bz);
                sgn = 1;
                if (tmp < 0) sgn = -1;
                c1_bz = 0; c2_bz = 0; c3_bz = tmp; rsq_bz = c3_bz*c3_bz;
                if (rsq_bz < numeric_limits<double>::epsilon()){
                    c3_bz = sgn; rsq_bz = 0;
                }
                if (cell.plane_intersects(c1_bz, c2_bz, c3_bz, rsq_bz)){
                cut_bz = true;
                }

                // Now perform all sections with the domain boundaries, if we were to immediately perform the sections
                // after checking that a boundary of the domain intersects, we may accidentaly skip a boundary since
                // the order of cutting with sections may matter
                if (cut_ax){
                    if (x_outside[i]){
                        c1_ax = -1*c1_ax; rsq_ax = -1*rsq_ax;
                        pid_ax = -2; pid_bx = -1;
                    } 
                    if (!(cell.nplane(c1_ax, c2_ax, c3_ax, rsq_ax, pid_ax))){
                        computed = false;
                    } 
                }
                if (cut_ay){
                    if (y_outside[i]){
                        c2_ay = -1*c2_ay; rsq_ay = -1*rsq_ay;
                        pid_ay = -4; pid_by = -3;
                    }
                    if (!(cell.nplane(c1_ay, c2_ay, c3_ay, rsq_ay, pid_ay))){
                        computed = false;
                    }
                }
                if (cut_az) {
                    if (z_outside[i]){
                        c3_az = -1*c3_az; rsq_az = -1*rsq_az;
                        pid_az = -6; pid_bz = -5;
                    }
                    if (!(cell.nplane(c1_az, c2_az, c3_az, rsq_az, pid_az))){
                        computed = false;
                    } 
                }
                if (cut_bx) {
                    if (x_outside[i]){
                        c1_bx = -1*c1_bx; rsq_bx = -1*rsq_bx;
                        pid_ax = -2; pid_bx = -1;
                    }
                    if (!(cell.nplane(c1_bx, c2_bx, c3_bx, rsq_bx, pid_bx))){
                        computed = false;
                    } 
                }
                if (cut_by) {
                    if (y_outside[i]){
                        c2_by = -1*c2_by; rsq_by = -1*rsq_by;
                        pid_ay = -4; pid_by = -3;
                    }
                    if (!(cell.nplane(c1_by, c2_by, c3_by, rsq_by, pid_by))){
                        computed = false;
                    } 
                }
                if (cut_bz) {
                    if (z_outside[i]){
                        c3_bz = -1*c3_bz; rsq_bz = -1*rsq_bz;
                        pid_az = -6; pid_bz = -5;
                    }
                    if (!(cell.nplane(c1_bz, c2_bz, c3_bz, rsq_bz, pid_bz))){
                        computed = false;
                    }  
                }

                if (computed){
                    vector<int> cell_face_vertices;
                    vector<int> cell_neighbors;
                    vector<double> cell_vertices;

                    cell.neighbors(cell_neighbors);
                    cell.face_vertices(cell_face_vertices);
                    cell.vertices(x_[i], y_[i], z_[i], cell_vertices);

                    vector<vector<int> > cell_vertices_indices(cell_neighbors.size());
                    int next_idx = 0;
                    for (int j = 0; j < cell_neighbors.size(); j++){
                        vector<int> face_vertices_indices(cell_face_vertices[next_idx]);
                        for (int k = 0; k < cell_face_vertices[next_idx]; k++){
                            face_vertices_indices[k] = cell_face_vertices[next_idx + 1 + k];
                        }
                        cell_vertices_indices[j] = face_vertices_indices;
                        next_idx += cell_face_vertices[next_idx] + 1;
                    }

                    neighbors[i] = cell_neighbors;
                    vertices[i] = cell_vertices;
                    vertices_indices[i] = cell_vertices_indices;
                    computed_cells[i] = true;
                }
            } 
        } 
    } while (cla.inc());
}


void compute_section(const vector<double> &weights, 
    double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
    const vector<double> &x_, const vector<double> &y_, const vector<double> &z_, int nx_, int ny_, int nz_, 
    double a, double b, double c, double x_0, double y_0, double z_0, vector<int> &section_indices, vector<double> &areas, vector<vector<double> > &section_vertices) {

    container_poly con(ax_, bx_, ay_, by_, az_, bz_, nx_, ny_, nz_, (bool)px_, (bool)py_, (bool)pz_, 6);
    c_loop_all cla(con);
    voronoicell_neighbor cell;

    double x,y,z,r;
    int section_plane_id = n_ + 1;
    double factor, c1, c2, c3, rsq;
    int sgn;
    int i;
    bool skip_cut = false;
    bool successful_cut = false;
    bool cell_in_plane = false;

    if (!(bool)px_) {
        if (abs(b) < numeric_limits<double>::epsilon() && abs(c) < numeric_limits<double>::epsilon()){
            if (abs(x_0 - con.ax) < numeric_limits<double>::epsilon()){
                skip_cut = true;
                section_plane_id = -1;
            }
            if (abs(x_0 - con.bx) < numeric_limits<double>::epsilon()){
                skip_cut = true;
                section_plane_id = -2;
            }
        }
    }
    if (!(bool)py_) {
        if (abs(a) < numeric_limits<double>::epsilon() && abs(c) < numeric_limits<double>::epsilon()){
            if (abs(y_0 - con.ay) < numeric_limits<double>::epsilon()){
                skip_cut = true;
                section_plane_id = -3;
            }
            if (abs(y_0 - con.by) < numeric_limits<double>::epsilon()){
                skip_cut = true;
                section_plane_id = -4;
            }
        }
    }
    if (!(bool)pz_) {
        if (abs(a) < numeric_limits<double>::epsilon() && abs(b) < numeric_limits<double>::epsilon()){
            if (abs(z_0 - con.az) < numeric_limits<double>::epsilon()){
                skip_cut = true;
                section_plane_id = -5;
            }
            if (abs(z_0 - con.bz) < numeric_limits<double>::epsilon()){
                skip_cut = true;
                section_plane_id = -6;
            }
        }
    }

    for (i = 0; i < n_; i++) {
        con.put(i, x_[i], y_[i], z_[i], sqrt(weights[i]));
    }
    
    if (cla.start()) do if (con.compute_cell(cell, cla)){

        successful_cut = false;
        cla.pos(i, x, y, z, r);

        if (!skip_cut){
            factor = -2*(a*(x - x_0) + b*(y - y_0) + c*(z - z_0))/(a*a + b*b + c*c);

            sgn = 1;
            if (factor < 0) sgn = -1;
            c1 = a*factor; c2 = b*factor; c3 = c*factor;
            rsq = c1*c1 + c2*c2 + c3*c3;
            // We check for a boundary case, the generator point may be in the cross section plane.
            // If rsq is zero, this is also the case for c1, c2 and c3. To prevent the cell from disappearing we need to slightly adjust the plane.
            if (rsq < numeric_limits<double>::epsilon()){
                c1 = a*sgn; c2 = b*sgn; c3 = c*sgn;
                rsq = 0;
            }

            // verify that the plane intersects with the cell
            if (cell.plane_intersects(c1, c2, c3, rsq)){
                // Use n_+1 as the id of the cross section plane. This id is guaranteed to not be used yet as the amount of ids
                // that is used is at most the number of generator points n_.
                if (cell.nplane(c1, c2, c3, rsq, section_plane_id)){
                    successful_cut = true;
                } 
            }
        }

        if (successful_cut || skip_cut) {
            vector<int> cell_neighbors;
            vector<int> cell_face_vertices;
            vector<double> cell_vertices;
            vector<double> cell_areas;

            cell.neighbors(cell_neighbors);
            cell.face_areas(cell_areas);
            cell.face_vertices(cell_face_vertices);
            cell.vertices(x, y, z, cell_vertices);

            int plane_face_idx;
            cell_in_plane = false;
            for (int j = 0; j < cell_neighbors.size(); j++){
                if (cell_neighbors[j] == section_plane_id){
                    plane_face_idx = j;
                    cell_in_plane = true;
                    areas.push_back(cell_areas[j]);
                    break;
                }
            }

            if (cell_in_plane){
                section_indices.push_back(i);

                int face_idx = 0;
                int next_idx = 0;
                while (face_idx != plane_face_idx) {
                    next_idx += cell_face_vertices[next_idx] + 1;
                    face_idx++;
                } 
                int num_vertices = cell_face_vertices[next_idx];
                vector<double> section_cell_vertices(3*num_vertices, 0.0);

                for (int j = 0; j < num_vertices; j++){
                    int ix1, ix2;
                    ix1 = j*3; ix2 = cell_face_vertices[next_idx + 1 + j]*3;
                    section_cell_vertices[ix1] = cell_vertices[ix2];
                    section_cell_vertices[ix1 + 1] = cell_vertices[ix2 + 1];
                    section_cell_vertices[ix1 + 2] = cell_vertices[ix2 + 2];
                }
                section_vertices.push_back(section_cell_vertices);
            }
        }
        
    } while (cla.inc());
}

void compute_section_bounded(const vector<double> &weights, 
    double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
    const vector<double> &x_, const vector<double> &y_, const vector<double> &z_, int nx_, int ny_, int nz_, 
    double a, double b, double c, double x_0, double y_0, double z_0, 
    const vector<bool> &cells_to_compute, const vector<bool> &x_outside, const vector<bool> &y_outside, const vector<bool> &z_outside, 
    vector<int> &section_indices, vector<double> &areas, vector<vector<double> > &section_vertices) {
    
    container_poly con(ax_, bx_, ay_, by_, az_, bz_, nx_, ny_, nz_, (bool)px_, (bool)py_, (bool)pz_, 6);
    c_loop_all cla(con);
    voronoicell_neighbor cell;

    int i;
    double x,y,z,r;
    
    double l1, l2, l3;
    l1 = con.bx - con.ax;
    l2 = con.by - con.ay;
    l3 = con.bz - con.az;

    double factor, tmp, c1, c2, c3, rsq;
    bool cut_ax, cut_bx, cut_ay, cut_by, cut_az, cut_bz;
    double c1_ax, c2_ax, c3_ax, c1_bx, c2_bx, c3_bx;
    double c1_ay, c2_ay, c3_ay, c1_by, c2_by, c3_by;
    double c1_az, c2_az, c3_az, c1_bz, c2_bz, c3_bz;
    double rsq_ax, rsq_bx, rsq_ay, rsq_by, rsq_az, rsq_bz;
    int sgn;

    for (i = 0; i < n_; i++) {
        con.put(i, x_[i], y_[i], z_[i], sqrt(weights[i]));
    }

    // Check whether the cross section plane coincides with the domain boundary, if so we only cut with the section plane
    // and therefore we skip cutting with the domain boundary for the appropriate axis.
    bool skip_x, skip_y, skip_z;
    skip_x = false; skip_y = false; skip_z = false;
    if (abs(b) < numeric_limits<double>::epsilon() && abs(c) < numeric_limits<double>::epsilon()){
        if (abs(x_0 - con.ax) < numeric_limits<double>::epsilon() || abs(x_0 - con.bx) < numeric_limits<double>::epsilon()){
            skip_x = true;
        }
    }
    if (abs(a) < numeric_limits<double>::epsilon() && abs(c) < numeric_limits<double>::epsilon()){
        if (abs(y_0 - con.ay) < numeric_limits<double>::epsilon() || abs(y_0 - con.by) < numeric_limits<double>::epsilon()){
            skip_y = true;
        }
    }
    if (abs(a) < numeric_limits<double>::epsilon() && abs(b) < numeric_limits<double>::epsilon()){
        if (abs(z_0 - con.az) < numeric_limits<double>::epsilon() || abs(z_0 - con.bz) < numeric_limits<double>::epsilon()){
            skip_z = true;
        }
    }

    if(cla.start()) do {
        cla.pos(i, x, y, z, r);
        if (cells_to_compute[i]){
            if (con.compute_cell(cell, cla)) {
                
                bool computed = true;
                cut_ax = false; cut_bx = false; cut_ay = false; cut_by = false; cut_az = false; cut_bz = false;

                // intersect with x = con.ax
                tmp = -2*(x - con.ax);
                sgn = 1;
                if (tmp < 0) sgn = -1;
                c1_ax = tmp; c2_ax = 0; c3_ax = 0; rsq_ax = c1_ax*c1_ax;
                if (rsq_ax < numeric_limits<double>::epsilon()){
                    c1_ax = sgn; rsq_ax = 0;
                }
                if (cell.plane_intersects(c1_ax, c2_ax, c3_ax, rsq_ax)){
                cut_ax = true;
                }

                // intersect with y = con.ay
                tmp = -2*(y - con.ay);
                sgn = 1;
                if (tmp < 0) sgn = -1;
                c1_ay = 0; c2_ay = tmp; c3_ay = 0; rsq_ay = c2_ay*c2_ay;
                if (rsq_ay < numeric_limits<double>::epsilon()){
                    c2_ay = sgn; rsq_ay = 0;
                }
                if (cell.plane_intersects(c1_ay, c2_ay, c3_ay, rsq_ay)){
                cut_ay = true;
                }

                // intersect with z = con.az
                tmp = -2*(z - con.az);
                sgn = 1;
                if (tmp < 0) sgn = -1;
                c1_az = 0; c2_az = 0; c3_az = tmp; rsq_az = c3_az*c3_az;
                if (rsq_az < numeric_limits<double>::epsilon()){
                    c3_az = sgn; rsq_az = 0;
                }
                if (cell.plane_intersects(c1_az, c2_az, c3_az, rsq_az)){
                cut_az = true;
                }

                // intersect with x = con.bx
                tmp = -2*(x-con.bx);
                sgn = 1;
                if (tmp < 0) sgn = -1;
                c1_bx = tmp; c2_bx = 0; c3_bx = 0; rsq_bx = c1_bx*c1_bx;
                if (rsq_bx < numeric_limits<double>::epsilon()){
                    c1_bx = sgn; rsq_bx = 0;
                }
                if (cell.plane_intersects(c1_bx, c2_bx, c3_bx, rsq_bx)){
                cut_bx = true;
                }

                // intersect with y = con.by
                tmp = -2*(y-con.by);
                sgn = 1;
                if (tmp < 0) sgn = -1;
                c1_by = 0; c2_by = tmp; c3_by = 0; rsq_by = c2_by*c2_by;
                if (rsq_by < numeric_limits<double>::epsilon()){
                    c2_by = sgn; rsq_by = 0;
                }
                if (cell.plane_intersects(c1_by, c2_by, c3_by, rsq_by)){
                cut_by = true;
                }

                // intersect with z = con.bz
                tmp = -2*(z-con.bz);
                sgn = 1;
                if (tmp < 0) sgn = -1;
                c1_bz = 0; c2_bz = 0; c3_bz = tmp; rsq_bz = c3_bz*c3_bz;
                if (rsq_bz < numeric_limits<double>::epsilon()){
                    c3_bz = sgn; rsq_bz = 0;
                }
                if (cell.plane_intersects(c1_bz, c2_bz, c3_bz, rsq_bz)){
                cut_bz = true;
                }

                // For some reason, voro++ simply puts each point in the domain (has to do with periodicity).
                // If we know the cell is in fact outside we correct for this.

                if (x_outside[i]){
                    if ((x - con.ax) < (con.bx - x)){
                        x += l1;
                    } else {
                        x -= l1;
                    }
                }
                if (y_outside[i]){
                    if ((y - con.ay) < (con.by - y)){
                        y += l2;
                    } else {
                        y -= l2;
                    }
                }
                if (z_outside[i]){
                    if ((z - con.az) < (con.bz - z)){
                        z += l3;
                    } else {
                        z -= l3;
                    }
                }

                // Now perform all sections with the domain boundaries, if we were to immediately perform the sections
                // after checking that a boundary of the domain intersects, we may accidentaly skip a boundary since
                // the order of cutting with sections may matter
                if (cut_ax && (!skip_x)){
                    if (x_outside[i]){
                        c1_ax = -1*c1_ax; rsq_ax = -1*rsq_ax;
                    } 
                    if (!(cell.nplane(c1_ax, c2_ax, c3_ax, rsq_ax, -1))){
                        computed = false;
                    } 
                }
                if (cut_ay && (!skip_y)){
                    if (y_outside[i]){
                        c2_ay = -1*c2_ay; rsq_ay = -1*rsq_ay;
                    }
                    if (!(cell.nplane(c1_ay, c2_ay, c3_ay, rsq_ay, -3))){
                        computed = false;
                    }
                }
                if (cut_az && (!skip_z)) {
                    if (z_outside[i]){
                        c3_az = -1*c3_az; rsq_az = -1*rsq_az;
                    }
                    if (!(cell.nplane(c1_az, c2_az, c3_az, rsq_az, -5))){
                        computed = false;
                    } 
                }
                if (cut_bx && (!skip_x)) {
                    if (x_outside[i]){
                        c1_bx = -1*c1_bx; rsq_bx = -1*rsq_bx;
                    }
                    if (!(cell.nplane(c1_bx, c2_bx, c3_bx, rsq_bx, -2))){
                        computed = false;
                    } 
                }
                if (cut_by && (!skip_y)) {
                    if (y_outside[i]){
                        c2_by = -1*c2_by; rsq_by = -1*rsq_by;
                    }
                    if (!(cell.nplane(c1_by, c2_by, c3_by, rsq_by, -4))){
                        computed = false;
                    } 
                }
                if (cut_bz && (!skip_z)) {
                    if (z_outside[i]){
                        c3_bz = -1*c3_bz; rsq_bz = -1*rsq_bz;
                    }
                    if (!(cell.nplane(c1_bz, c2_bz, c3_bz, rsq_bz, -6))){
                        computed = false;
                    }  
                }

                if (computed){
                    // If the cell has not vanished by cutting the cell with domain boundaries we check if it intersects with 
                    // our cross section plane. 

                    factor = -2*(a*(x - x_0) + b*(y - y_0) + c*(z - z_0))/(a*a + b*b + c*c);

                    sgn = 1;
                    if (factor < 0) sgn = -1;
                    c1 = a*factor; c2 = b*factor; c3 = c*factor;
                    rsq = c1*c1 + c2*c2 + c3*c3;
                    // We check for a boundary case, the generator point may be in the cross section plane.
                    // If rsq is zero, this is also the case for c1, c2 and c3. To prevent the cell from disappearing we need to slightly adjust the plane.
                    if (rsq < numeric_limits<double>::epsilon()){
                        c1 = a*sgn; c2 = b*sgn; c3 = c*sgn;
                        rsq = 0;
                    }

                    // verify that the plane intersects with the cell
                    if (cell.plane_intersects(c1, c2, c3, rsq)){
                        // Use n_+1 as the id of the cross section plane. This id is guaranteed to not be used yet as the amount of ids
                        // that is used is at most the number of generator points n_.

                        if (cell.nplane(c1, c2, c3, rsq, n_+1)){
                            vector<int> cell_neighbors;
                            vector<int> cell_face_vertices;
                            vector<double> cell_vertices;
                            vector<double> cell_areas;

                            cell.neighbors(cell_neighbors);
                            cell.face_areas(cell_areas);
                            cell.face_vertices(cell_face_vertices);
                            cell.vertices(x, y, z, cell_vertices);

                            int plane_face_idx;
                            for (int j = 0; j < cell_neighbors.size(); j++){
                                if (cell_neighbors[j] == n_+1){
                                    plane_face_idx = j;
                                    areas.push_back(cell_areas[j]);
                                    break;
                                }
                            }
                            section_indices.push_back(i);

                            int face_idx = 0;
                            int next_idx = 0;
                            while (face_idx != plane_face_idx) {
                                next_idx += cell_face_vertices[next_idx] + 1;
                                face_idx++;
                            } 
                            int num_vertices = cell_face_vertices[next_idx];
                            vector<double> section_cell_vertices(3*num_vertices, 0.0);

                            for (int j = 0; j < num_vertices; j++){
                                int ix1, ix2;
                                ix1 = j*3; ix2 = cell_face_vertices[next_idx + 1 + j]*3;
                                section_cell_vertices[ix1] = cell_vertices[ix2];
                                section_cell_vertices[ix1 + 1] = cell_vertices[ix2 + 1];
                                section_cell_vertices[ix1 + 2] = cell_vertices[ix2 + 2];
                            }
                            section_vertices.push_back(section_cell_vertices);
                        }
                    }
                }
            }
        }
    } while (cla.inc());
}

void compute_num_fragments(const vector<double> &weights, 
    double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
    const vector<double> &x_, const vector<double> &y_, const vector<double> &z_, int nx_, int ny_, int nz_, vector<vector<bool> > &cell_sections) {

    container_poly con(ax_, bx_, ay_, by_, az_, bz_, nx_, ny_, nz_, (bool)px_, (bool)py_, (bool)pz_, 6);
    c_loop_all cla(con); 
    voronoicell_neighbor cell;

    cell_sections.resize(n_);
    double c1, c2, c3, rsq, tmp;
    int sgn;
    int i;
    double x,y,z,r;

    for (i = 0; i < n_; i++) {
        con.put(i, x_[i], y_[i], z_[i], sqrt(weights[i]));
        cell_sections[i] = vector<bool>(6, false);
    }
    
    if(cla.start()) do if (con.compute_cell(cell, cla)) {

        cla.pos(i, x, y, z, r);

        // intersect with x = con.ax
        sgn = 1;
        tmp = -2*(x - con.ax);
        if (tmp < 0) sgn = -1;
        c1 = tmp; c2 = 0; c3 = 0; rsq = c1*c1;
        if (rsq < numeric_limits<double>::epsilon()){
            c1 =sgn; rsq = 0;
        }
        if (cell.plane_intersects(c1, c2, c3, rsq)){
        cell_sections[i][0] = true;
        }

        // intersect with y = con.ay
        sgn = 1;
        tmp = -2*(y - con.ay);
        if (tmp < 0) sgn = -1;
        c1 = 0; c2 = tmp; c3 = 0; rsq = c2*c2;
        if (rsq < numeric_limits<double>::epsilon()){
            c2 = sgn; rsq = 0;
        }
        if (cell.plane_intersects(c1, c2, c3, rsq)){
        cell_sections[i][2] = true;
        }

        // intersect with z = con.az
        sgn = 1;
        tmp = -2*(z - con.az);
        if (tmp < 0) sgn = -1;
        c1 = 0; c2 = 0; c3 = tmp; rsq = c3*c3;
        if (rsq < numeric_limits<double>::epsilon()){
            c3 = sgn; rsq = 0;
        }
        if (cell.plane_intersects(c1, c2, c3, rsq)){
        cell_sections[i][4] = true;
        }

        // intersect with x = con.bx
        sgn = 1;
        tmp = -2*(x-con.bx);
        if (tmp < 0) sgn = -1;
        c1 = tmp; c2 = 0; c3 = 0; rsq = c1*c1;
        if (rsq < numeric_limits<double>::epsilon()){
            c1 = sgn; rsq = 0;
        }
        if (cell.plane_intersects(c1, c2, c3, rsq)){
        cell_sections[i][1] = true;
        } 

        // intersect with y = con.by
        sgn = 1;
        tmp = -2*(y-con.by);
        if (tmp < 0) sgn = -1;
        c1 = 0; c2 = tmp; c3 = 0; rsq = c2*c2;
        if (rsq < numeric_limits<double>::epsilon()){
            c2 = sgn; rsq = 0;
        }
        if (cell.plane_intersects(c1, c2, c3, rsq)){
        cell_sections[i][3] = true;
        }  

        // intersect with z = con.bz
        sgn = 1;
        tmp = -2*(z-con.bz);
        if (tmp < 0) sgn = -1;
        c1 = 0; c2 = 0; c3 = tmp; rsq = c3*c3;
        if (rsq < numeric_limits<double>::epsilon()){
            c3 = sgn; rsq = 0;
        }
        if (cell.plane_intersects(c1, c2, c3, rsq)){
        cell_sections[i][5] = true;
        } 
        
    } while (cla.inc());
}
