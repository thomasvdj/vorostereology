// Edited by Thomas van der Jagt

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
#include <algorithm>  
#include <math.h>
using namespace voro;
using namespace std;

void* container_poly_create(double ax_, double bx_, double ay_, double by_, 
    double az_, double bz_, int nx_, int ny_, int nz_, int px_, int py_, int pz_) {
  
    return (void*)new container_poly(ax_, bx_, ay_, by_, az_, bz_, nx_, ny_, nz_, (bool)px_, 
        (bool)py_, (bool)pz_, 3);
}

void put_particle(void* container_poly_, int i_, double x_, double y_, double z_, double r_) {
    container_poly* c = (container_poly*)container_poly_;
    c->put(i_, x_, y_, z_, r_);
}

void put_particles(void* container_poly_, int n_, double* x_, double* y_, double* z_, double* r_) {
    container_poly* c = (container_poly*)container_poly_;
    int i;
    for (i = 0; i < n_; i++) {
        c->put(i, x_[i], y_[i], z_[i], r_[i]);
    }
}

vector<double> compute_volumes(const vector<double> &weights, 
    double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
    const vector<double> &x_, const vector<double> &y_, const vector<double> &z_, int num_cpus){
    
    vector<double> volumes(n_, 0.0);

    int num_threads;
    if (RUN_PARALLEL){
        num_threads = num_cpus;
    } else {
        num_threads = 1;
    }
    int block_size = n_/num_threads + (n_ % num_threads != 0);  // ceiling division

    #pragma omp parallel for
    for (int m = 0; m < num_threads; m++){
        vector<bool> cells_to_compute(n_, false);
        for (int k = (m * block_size); k < min(((m + 1) * block_size), n_); k++){
            cells_to_compute[k] = true;
        }
        container_poly* con = new container_poly(ax_, bx_, ay_, by_, az_, bz_, 4, 4, 4, (bool)px_, (bool)py_, (bool)pz_, 3);
        int i;
        c_loop_all* cla = new c_loop_all(*(con)); 
        voronoicell_neighbor cell;
        voronoicell_neighbor* cellptr = new voronoicell_neighbor();

        for (i = 0; i < n_; i++) {
            con->put(i, x_[i], y_[i], z_[i], sqrt(weights[i]));
        }

        if(cla->start()) do {
            i = cla->pid();
            if (cells_to_compute[i]){
                if (con->compute_cell(cell, *(cla))) {
                    *(cellptr) = cell;
                    volumes[i] = cellptr->volume();
                } 
            } 
        } while (cla->inc());

        delete cellptr;
        delete cla;
        delete con;
    }
    return volumes;
}


pair<vector<double>, vector<double> > compute_volumes_centroids(const vector<double> &weights, 
    double ax_, double bx_, double ay_, double by_, double az_, double bz_, int px_, int py_, int pz_, int n_, 
    const vector<double> &x_, const vector<double> &y_, const vector<double> &z_, int num_cpus) {

    vector<double> volumes(n_, 0.0);
    vector<double> centroids((n_) * 3, 0.0);

    int num_threads;
    if (RUN_PARALLEL){
        num_threads = num_cpus;
    } else {
        num_threads = 1;
    }
    int block_size = n_/num_threads + (n_ % num_threads != 0);

    #pragma omp parallel for
    for (int m = 0; m < num_threads; m++){
        vector<bool> cells_to_compute(n_, false);
        for (int k = (m * block_size); k < min(((m + 1) * block_size), n_); k++){
            cells_to_compute[k] = true;
        }
        container_poly* con = new container_poly(ax_, bx_, ay_, by_, az_, bz_, 4, 4, 4, (bool)px_, (bool)py_, (bool)pz_, 3);
        int i;
        double x, y, z, r;
        c_loop_all* cla = new c_loop_all(*(con)); 
        voronoicell_neighbor cell;
        voronoicell_neighbor* cellptr = new voronoicell_neighbor();

        for (i = 0; i < n_; i++) {
            con->put(i, x_[i], y_[i], z_[i], sqrt(weights[i]));
        }

        if(cla->start()) do {
            cla->pos(i, x, y, z, r);
            if (cells_to_compute[i]){
                if (con->compute_cell(cell, *(cla))) {
                    *(cellptr) = cell;
                    volumes[i] = cellptr->volume();
                    cellptr->centroid(centroids[i], centroids[i + n_], centroids[i + (2 * n_)]);
                    centroids[i] += x;
                    centroids[i + n_] += y;
                    centroids[i + (2 * n_)] += z;
                } 
            } 
        } while (cla->inc());

        delete cellptr;
        delete cla;
        delete con;
    }
    return make_pair(volumes, centroids);
}


vector<double> permute_targets(const vector<double> &volumes, const vector<double> &target_volumes, int n_){
    vector<int> volumes_indices(n_);
    iota(volumes_indices.begin(), volumes_indices.end(), 0);
    sort(volumes_indices.begin(), volumes_indices.end(), [&](int i, int j){return volumes[i]<volumes[j];} );
    vector<int> targets_indices(n_);
    iota(targets_indices.begin(), targets_indices.end(), 0);
    sort(targets_indices.begin(), targets_indices.end(), [&](int i, int j){return target_volumes[i]<target_volumes[j];} );
    vector<double> new_targets(n_);
    for (int i = 0; i < n_; i++){
        new_targets[volumes_indices[i]] = target_volumes[targets_indices[i]];
    }
    return new_targets;
}


void** compute_voronoi_tessellation(void* container_poly_, int n_) {
    container_poly* con = (container_poly*)container_poly_;

    int found = 0;
    int i;
    double x, y, z, r;
    c_loop_all* cla = new c_loop_all(*(con));
    voronoicell_neighbor cell;
    voronoicell_neighbor* cellptr = NULL;
    
    void** vorocells = (void**)malloc(sizeof(void*) * n_);
    
    for (i = 0; i < n_; i++) vorocells[i] = NULL;
    
    if(cla->start()) do if (con->compute_cell(cell, *(cla))) {

        // Get the position and ID information for the particle
        // currently being considered by the loop.
        cla->pos(i, x, y, z, r);
        
        // Store the resulting cell instance at the appropriate index on vorocells.
        cellptr = new voronoicell_neighbor();
        *(cellptr) = cell;
        vorocells[i] = (void*)cellptr;
        found++;
        
    } while (cla->inc());
    
    delete cla;
    
    if (found != n_) {
        printf("missing cells: ");
        
        for (i = 0; i < n_; i++) {
        if (vorocells[i] != NULL) {
            delete (voronoicell_neighbor*)vorocells[i];
        } else {
            printf("%i ", i);
        }
        }
        free(vorocells);
        printf("\n");
        return NULL;
    }
    
    return vorocells;
}

pair<void**, vector<int> > compute_voronoi_tessellation_par(void* container_poly_, int n_) {
    container_poly* con = (container_poly*)container_poly_;
    
    int found = 0;
    int i;
    c_loop_all* cla = new c_loop_all(*(con)); 
    voronoicell_neighbor cell;
    voronoicell_neighbor* cellptr = NULL;
    
    void** vorocells = (void**)malloc(sizeof(void*) * n_);
    
    for (i = 0; i < n_; i++) vorocells[i] = NULL;
    
    // Maybe we can make an array/vector/list of cell pointers and iterate over that in a for loop
    
    if(cla->start()) do if (con->compute_cell(cell, *(cla))) {

        // Store the resulting cell instance at the appropriate index on vorocells.
        cellptr = new voronoicell_neighbor();
        *(cellptr) = cell;
        vorocells[cla->pid()] = (void*)cellptr;
        found++;
        
    } while (cla->inc());
    
    delete cla;
    
    vector<int> empty_indices;
    if (found != n_) {  
        for (i = 0; i < n_; i++) {
        if (vorocells[i] == NULL) {
            // kind of hacky: if a cell could not be computed populate the index with the last cell that could be computed
            // the function also returns the vector "empty_indices" containing the indices in "vorocells" with empty cells
            
            cellptr = new voronoicell_neighbor();
                *(cellptr) = cell;
                vorocells[i] = (void*)cellptr;
                empty_indices.push_back(i);
        }
        }   
    }
    
    return make_pair(vorocells, empty_indices);
}

pair<void**, vector<bool> > compute_voronoi_tessellation_bounded(void* container_poly_, int n_, vector<bool> cells_to_compute, 
    vector<bool> x_outside, vector<bool> y_outside, vector<bool> z_outside) {
    container_poly* con = (container_poly*)container_poly_;
    
    int found = 0;
    int i;
    c_loop_all* cla = new c_loop_all(*(con)); 
    voronoicell_neighbor cell;
    voronoicell_neighbor* cellptr = NULL;

    vector<bool> computed_indices(n_, false);
    void** vorocells = (void**)malloc(sizeof(void*) * n_);
    double epsilon = 0.0000000001;
    double x,y,z,r;
    bool cut_ax, cut_bx, cut_ay, cut_by, cut_az, cut_bz;
    
    double tmp;
    double c1_ax, c2_ax, c3_ax, c1_bx, c2_bx, c3_bx;
    double c1_ay, c2_ay, c3_ay, c1_by, c2_by, c3_by;
    double c1_az, c2_az, c3_az, c1_bz, c2_bz, c3_bz;
    double rsq_ax, rsq_bx, rsq_ay, rsq_by, rsq_az, rsq_bz;
    int sgn;

    for (i = 0; i < n_; i++) vorocells[i] = NULL;

    if(cla->start()) do {
        cla->pos(i, x, y, z, r);
        if (cells_to_compute[i]){
        if (con->compute_cell(cell, *(cla))) {
            
            cellptr = new voronoicell_neighbor();
            *(cellptr) = cell;
            bool computed = true;
            cut_ax = false; cut_bx = false; cut_ay = false; cut_by = false; cut_az = false; cut_bz = false;

            // intersect with x = con->ax
            tmp = -2*(x - con->ax);
            sgn = 1;
            if (tmp < 0) sgn = -1;
            if ((x - con->ax < epsilon) && (x - con->ax > -1*epsilon)){
            c1_ax = sgn; c2_ax = 0; c3_ax = 0; rsq_ax = 0;
            } else {
            c1_ax = tmp; c2_ax = 0; c3_ax = 0; rsq_ax = c1_ax*c1_ax;
            }
            if (cellptr->plane_intersects(c1_ax, c2_ax, c3_ax, rsq_ax)){
            cut_ax = true;
            }

            // intersect with y = con->ay
            sgn = 1;
            tmp = -2*(y - con->ay);
            if (tmp < 0) sgn = -1;
            if ((y - con->ay < epsilon) && (y - con->ay > -1*epsilon)){
            c1_ay = 0; c2_ay = sgn; c3_ay = 0; rsq_ay = 0;
            } else {
            c1_ay = 0; c2_ay = tmp; c3_ay = 0; rsq_ay = c2_ay*c2_ay;
            }
            if (cellptr->plane_intersects(c1_ay, c2_ay, c3_ay, rsq_ay)){
            cut_ay = true;
            }

            // intersect with z = con->az
            sgn = 1;
            tmp = -2*(z - con->az);
            if (tmp < 0) sgn = -1;
            if ((z - con->az < epsilon) && (z - con->az > -1*epsilon)){
            c1_az = 0; c2_az = 0; c3_az = sgn; rsq_az = 0;
            } else {
            c1_az = 0; c2_az = 0; c3_az = tmp; rsq_az = c3_az*c3_az;
            }
            if (cellptr->plane_intersects(c1_az, c2_az, c3_az, rsq_az)){
            cut_az = true;
            }

            // intersect with x = con->bx
            sgn = 1;
            tmp = -2*(x-con->bx);
            if (tmp < 0) sgn = -1;
            if ((x - con->bx < epsilon) && (x - con->bx > -1*epsilon)){
            c1_bx = sgn; c2_bx = 0; c3_bx = 0; rsq_bx = 0;
            } else {
            c1_bx = tmp; c2_bx = 0; c3_bx = 0; rsq_bx = c1_bx*c1_bx;
            }
            if (cellptr->plane_intersects(c1_bx, c2_bx, c3_bx, rsq_bx)){
            cut_bx = true;
            }

            // intersect with y = con->by
            sgn = 1;
            tmp = -2*(y-con->by);
            if (tmp < 0) sgn = -1;
            if ((y - con->by < epsilon) && (y - con->by > -1*epsilon)){
            c1_by = 0; c2_by = sgn; c3_by = 0; rsq_by = 0;
            } else {
            c1_by = 0; c2_by = tmp; c3_by = 0; rsq_by = c2_by*c2_by;
            }
            if (cellptr->plane_intersects(c1_by, c2_by, c3_by, rsq_by)){
            cut_by = true;
            }

            // intersect with z = con->bz
            sgn = 1;
            tmp = -2*(z-con->bz);
            if (tmp < 0) sgn = -1;
            if ((z - con->bz < epsilon) && (z - con->bz > -1*epsilon)){
            c1_bz = 0; c2_bz = 0; c3_bz = sgn; rsq_bz = 0;
            } else {
            c1_bz = 0; c2_bz = 0; c3_bz = tmp; rsq_bz = c3_bz*c3_bz;
            }
            if (cellptr->plane_intersects(c1_bz, c2_bz, c3_bz, rsq_bz)){
            cut_bz = true;
            }

            if (cut_ax){
            if (x_outside[i]){
                c1_ax = -1*c1_ax; rsq_ax = -1*rsq_ax;
            } 
            if (!(cellptr->nplane(c1_ax, c2_ax, c3_ax, rsq_ax, -1))){
                computed = false;
            } 
            }
            if (cut_ay){
            if (y_outside[i]){
                c2_ay = -1*c2_ay; rsq_ay = -1*rsq_ay;
            }
            if (!(cellptr->nplane(c1_ay, c2_ay, c3_ay, rsq_ay, -3))){
                computed = false;
            }
            }
            if (cut_az) {
            if (z_outside[i]){
                c3_az = -1*c3_az; rsq_az = -1*rsq_az;
            }
            if (!(cellptr->nplane(c1_az, c2_az, c3_az, rsq_az, -5))){
                computed = false;
            } 
            }
            if (cut_bx) {
            if (x_outside[i]){
                c1_bx = -1*c1_bx; rsq_bx = -1*rsq_bx;
            }
            if (!(cellptr->nplane(c1_bx, c2_bx, c3_bx, rsq_bx, -2))){
                computed = false;
            } 
            }
            if (cut_by) {
            if (y_outside[i]){
                c2_by = -1*c2_by; rsq_by = -1*rsq_by;
            }
            if (!(cellptr->nplane(c1_by, c2_by, c3_by, rsq_by, -4))){
                computed = false;
            } 
            }
            if (cut_bz) {
            if (z_outside[i]){
                c3_bz = -1*c3_bz; rsq_bz = -1*rsq_bz;
            }
            if (!(cellptr->nplane(c1_bz, c2_bz, c3_bz, rsq_bz, -6))){
                computed = false;
            }  
            }

            if (computed){
            computed_indices[i] = true;
            found++;
            }
            vorocells[i] = (void*)cellptr;
        }
        }
        
    } while (cla->inc());

    delete cla;
    
    if (found != n_) {  
        for (i = 0; i < n_; i++) {
        if (vorocells[i] == NULL) {
            // kind of hacky: if a cell could not be computed populate the index with the last cell that could be computed
            if (found == 0){
            voronoicell_neighbor v;
            v.init(-1,1,-1,1,-1,1);
            voronoicell_neighbor* cellptr = NULL;
            cellptr = new voronoicell_neighbor();
            *(cellptr) = v;
            vorocells[i] = (void*)cellptr;
            } else {
            cellptr = new voronoicell_neighbor();
            *(cellptr) = cell;
            vorocells[i] = (void*)cellptr;
            }
        }
        }   
    }
    
    return make_pair(vorocells, computed_indices);
}


pair<void**, vector<int> > compute_section(void* container_poly_, int n_, double a, double b, 
    double c, double x_0, double y_0, double z_0) {
    container_poly* con = (container_poly*)container_poly_;
    int i;
    c_loop_all* cla = new c_loop_all(*(con)); 
    voronoicell_neighbor cell;
    voronoicell_neighbor* cellptr = NULL;
    vector<int> section_indices;
    double x,y,z,r;
    double epsilon = 0.0000000001;

    void** vorocells = (void**)malloc(sizeof(void*) * n_);
    
    for (i = 0; i < n_; i++) vorocells[i] = NULL;
    
    double l1, l2, l3;
    l1 = con->bx - con->ax;
    l2 = con->by - con->ay;
    l3 = con->bz - con->az;

    double plane_check, factor, tmp, temp_dist, min_dist, c1, c2, c3, rsq;
    int i_per, j_per, k_per, sgn, i_, j_, k_;
    
    if(cla->start()) do if (con->compute_cell(cell, *(cla))) {

        // Store the resulting cell instance at the appropriate index on vorocells.
        cellptr = new voronoicell_neighbor();
        *(cellptr) = cell;
        cla->pos(i, x, y, z, r);
        // add code to obtain cell vertices
        
        factor = (a*(x - x_0) + b*(y - y_0) + c*(z - z_0))/(a*a + b*b + c*c);
        tmp = -2*factor;
        i_per=0;j_per=0;k_per=0;

        // Periodic case: we need the periodic distance from our cross section plane to the generator point of the cell
        if ((con->xperiodic && con->yperiodic) && con->zperiodic) {
        
        c1 = a*factor; c2 = b*factor; c3 = c*factor;
        min_dist = c1*c1 + c2*c2 + c3*c3;

        for (i_=-1; i_<2; i_++){
            for (j_=-1; j_<2; j_++){
            for (k_=-1; k_<2; k_++){
                c1 = a*factor + i_*l1; c2 = b*factor + j_*l2; c3 = c*factor + k_*l3;
                temp_dist = c1*c1 + c2*c2 + c3*c3;
    
                if (temp_dist < min_dist){
                i_per = i_; j_per = j_; k_per = k_;
                min_dist = temp_dist;
                }
            }
            }
        }

        }

        sgn = 1;
        if (tmp < 0){
        sgn = -1;
        }
        // check for boundary case, the generator point may be in the cross section plane
        plane_check = a*(x - x_0 + i_per*l1) + b*(y - y_0 + j_per*l2) + c*(z - z_0 + +k_per*l3);
        if ((plane_check < epsilon) && ((plane_check) > -1*epsilon)){
        c1 = a*sgn; c2 = b*sgn; c3 = c*sgn;
        rsq = 0;
        } else {
        c1 = -2*(a*factor+i_per*l1); c2 = -2*(b*factor+j_per*l2); c3 = -2*(c*factor+k_per*l3);
        rsq = c1*c1 + c2*c2 + c3*c3;
        }

        // verify that the plane intersects with the cell
        if (cellptr->plane_intersects(c1, c2, c3, rsq)){
        // Use n_+1 as the id of the cross section plane. This id is guaranteed to not be used yet as the amount of ids
        // that is used is at most the number of generator points n_.
        if (cellptr->nplane(c1, c2, c3, rsq, n_+1)){
            //cellptr->check_relations();
            //cellptr->check_duplicates();
            //
            section_indices.push_back(i);
        }
        }
        vorocells[i] = (void*)cellptr;
        
    } while (cla->inc());
    
    delete cla;
    
    for (i = 0; i < n_; i++) {
        if (vorocells[i] == NULL) {
        // kind of hacky: if a cell could not be computed populate the index with the last cell that could be computed
        // the function also returns the vector "empty_indices" containing the indices in "vorocells" with empty cells
        
        cellptr = new voronoicell_neighbor();
        *(cellptr) = cell;
        vorocells[i] = (void*)cellptr;
        }
    }   

    return make_pair(vorocells, section_indices);
}

pair<void**, vector<int> > compute_section_bounded(void* container_poly_, int n_, double a, double b, double c, double x_0, double y_0, double z_0, 
    vector<bool> cells_to_compute, vector<bool> x_outside, vector<bool> y_outside, vector<bool> z_outside) {
    
    container_poly* con = (container_poly*)container_poly_;
    int found = 0;
    int i;
    c_loop_all* cla = new c_loop_all(*(con)); 
    voronoicell_neighbor cell;
    voronoicell_neighbor* cellptr = NULL;
    vector<int> section_indices;
    double x,y,z,r;
    double epsilon = 0.0000000001;

    void** vorocells = (void**)malloc(sizeof(void*) * n_);
    
    for (i = 0; i < n_; i++) vorocells[i] = NULL;
    
    double l1, l2, l3;
    l1 = con->bx - con->ax;
    l2 = con->by - con->ay;
    l3 = con->bz - con->az;

    double plane_check, factor, tmp, c1, c2, c3, rsq;
    bool cut_ax, cut_bx, cut_ay, cut_by, cut_az, cut_bz;
    
    double c1_ax, c2_ax, c3_ax, c1_bx, c2_bx, c3_bx;
    double c1_ay, c2_ay, c3_ay, c1_by, c2_by, c3_by;
    double c1_az, c2_az, c3_az, c1_bz, c2_bz, c3_bz;
    double rsq_ax, rsq_bx, rsq_ay, rsq_by, rsq_az, rsq_bz;
    int sgn;

    // Check whether the cross section plane coincides with the domain boundary, if so we only cut with the section plane
    // and therefore we skip cutting with the domain boundary for the appropriate axis.
    bool skip_x, skip_y, skip_z;
    skip_x = false; skip_y = false; skip_z = false;
    if (abs(b) < epsilon && abs(c) < epsilon){
        if ((abs(x_0 - con->ax) < epsilon) || (abs(x_0 - con->bx) < epsilon)){
        skip_x = true;
        }
    }
    if (abs(a) < epsilon && abs(c) < epsilon){
        if ((abs(y_0 - con->ay) < epsilon) || (abs(y_0 - con->by) < epsilon)){
        skip_y = true;
        }
    }
    if (abs(a) < epsilon && abs(b) < epsilon){
        if ((abs(z_0 - con->az) < epsilon) || (abs(z_0 - con->bz) < epsilon)){
        skip_z = true;
        }
    }
    

    if(cla->start()) do {
        cla->pos(i, x, y, z, r);
        if (cells_to_compute[i]){
        if (con->compute_cell(cell, *(cla))) {
            
            cellptr = new voronoicell_neighbor();
            *(cellptr) = cell;
            bool computed = true;
            cut_ax = false; cut_bx = false; cut_ay = false; cut_by = false; cut_az = false; cut_bz = false;

            // intersect with x = con->ax
            tmp = -2*(x - con->ax);
            sgn = 1;
            if (tmp < 0) sgn = -1;
            if (abs(x - con->ax) < epsilon){
            c1_ax = sgn; c2_ax = 0; c3_ax = 0; rsq_ax = 0;
            } else {
            c1_ax = tmp; c2_ax = 0; c3_ax = 0; rsq_ax = c1_ax*c1_ax;
            }
            if (cellptr->plane_intersects(c1_ax, c2_ax, c3_ax, rsq_ax)){
            cut_ax = true;
            }

            // intersect with y = con->ay
            sgn = 1;
            tmp = -2*(y - con->ay);
            if (tmp < 0) sgn = -1;
            if (abs(y - con->ay) < epsilon){
            c1_ay = 0; c2_ay = sgn; c3_ay = 0; rsq_ay = 0;
            } else {
            c1_ay = 0; c2_ay = tmp; c3_ay = 0; rsq_ay = c2_ay*c2_ay;
            }
            if (cellptr->plane_intersects(c1_ay, c2_ay, c3_ay, rsq_ay)){
            cut_ay = true;
            }

            // intersect with z = con->az
            sgn = 1;
            tmp = -2*(z - con->az);
            if (tmp < 0) sgn = -1;
            if (abs(z - con->az) < epsilon){
            c1_az = 0; c2_az = 0; c3_az = sgn; rsq_az = 0;
            } else {
            c1_az = 0; c2_az = 0; c3_az = tmp; rsq_az = c3_az*c3_az;
            }
            if (cellptr->plane_intersects(c1_az, c2_az, c3_az, rsq_az)){
            cut_az = true;
            }

            // intersect with x = con->bx
            sgn = 1;
            tmp = -2*(x-con->bx);
            if (tmp < 0) sgn = -1;
            if (abs(x - con->bx) < epsilon){
            c1_bx = sgn; c2_bx = 0; c3_bx = 0; rsq_bx = 0;
            } else {
            c1_bx = tmp; c2_bx = 0; c3_bx = 0; rsq_bx = c1_bx*c1_bx;
            }
            if (cellptr->plane_intersects(c1_bx, c2_bx, c3_bx, rsq_bx)){
            cut_bx = true;
            }

            // intersect with y = con->by
            sgn = 1;
            tmp = -2*(y-con->by);
            if (tmp < 0) sgn = -1;
            if (abs(y - con->by) < epsilon){
            c1_by = 0; c2_by = sgn; c3_by = 0; rsq_by = 0;
            } else {
            c1_by = 0; c2_by = tmp; c3_by = 0; rsq_by = c2_by*c2_by;
            }
            if (cellptr->plane_intersects(c1_by, c2_by, c3_by, rsq_by)){
            cut_by = true;
            }

            // intersect with z = con->bz
            sgn = 1;
            tmp = -2*(z-con->bz);
            if (tmp < 0) sgn = -1;
            if (abs(z - con->bz) < epsilon){
            c1_bz = 0; c2_bz = 0; c3_bz = sgn; rsq_bz = 0;
            } else {
            c1_bz = 0; c2_bz = 0; c3_bz = tmp; rsq_bz = c3_bz*c3_bz;
            }
            if (cellptr->plane_intersects(c1_bz, c2_bz, c3_bz, rsq_bz)){
            cut_bz = true;
            }

            // For some reason, voro++ simply puts each point in the domain (has to do with periodicity).
            // If we know the cell is in fact outside we correct for this.

            if (x_outside[i]){
            if ((x - con->ax) < (con->bx - x)){
                x = x + l1;
            } else {
                x = x - l1;
            }
            }
            if (y_outside[i]){
            if ((y - con->ay) < (con->by - y)){
                y = y + l2;
            } else {
                y = y - l2;
            }
            }
            if (z_outside[i]){
            if ((z - con->az) < (con->bz - z)){
                z = z + l3;
            } else {
                z = z - l3;
            }
            }

            // Now perform all sections with the domain boundaries, if we were to immediately perform the sections
            // after checking that a boundary of the domain intersects, we may accidentaly skip a boundary since
            // the order of cutting with sections may matter
            if (cut_ax && (!skip_x)){
            if (x_outside[i]){
                c1_ax = -1*c1_ax; rsq_ax = -1*rsq_ax;
            } 
            if (!(cellptr->nplane(c1_ax, c2_ax, c3_ax, rsq_ax, -1))){
                computed = false;
            } 
            }
            if (cut_ay && (!skip_y)){
            if (y_outside[i]){
                c2_ay = -1*c2_ay; rsq_ay = -1*rsq_ay;
            }
            if (!(cellptr->nplane(c1_ay, c2_ay, c3_ay, rsq_ay, -3))){
                computed = false;
            }
            }
            if (cut_az && (!skip_z)) {
            if (z_outside[i]){
                c3_az = -1*c3_az; rsq_az = -1*rsq_az;
            }
            if (!(cellptr->nplane(c1_az, c2_az, c3_az, rsq_az, -5))){
                computed = false;
            } 
            }
            if (cut_bx && (!skip_x)) {
            if (x_outside[i]){
                c1_bx = -1*c1_bx; rsq_bx = -1*rsq_bx;
            }
            if (!(cellptr->nplane(c1_bx, c2_bx, c3_bx, rsq_bx, -2))){
                computed = false;
            } 
            }
            if (cut_by && (!skip_y)) {
            if (y_outside[i]){
                c2_by = -1*c2_by; rsq_by = -1*rsq_by;
            }
            if (!(cellptr->nplane(c1_by, c2_by, c3_by, rsq_by, -4))){
                computed = false;
            } 
            }
            if (cut_bz && (!skip_z)) {
            if (z_outside[i]){
                c3_bz = -1*c3_bz; rsq_bz = -1*rsq_bz;
            }
            if (!(cellptr->nplane(c1_bz, c2_bz, c3_bz, rsq_bz, -6))){
                computed = false;
            }  
            }

            if (computed){
            // If the cell has not vanished by cutting the cell with domain boundaries we check if it intersects with 
            // our cross section plane. 
            factor = -2*(a*(x - x_0) + b*(y - y_0) + c*(z - z_0))/(a*a + b*b + c*c);

            sgn = 1;
            if (factor < 0){
                sgn = -1;
            }
            // check for boundary case, the generator point may be in the cross section plane
            plane_check = a*(x - x_0) + b*(y - y_0) + c*(z - z_0);
            if ((plane_check < epsilon) && ((plane_check) > -1*epsilon)){
                c1 = a*sgn; c2 = b*sgn; c3 = c*sgn;
                rsq = 0;
            } else {
                c1 = a*factor; c2 = b*factor; c3 = c*factor;
                rsq = c1*c1 + c2*c2 + c3*c3;
            }

            // verify that the plane intersects with the cell
            if (cellptr->plane_intersects(c1, c2, c3, rsq)){
                // Use n_+1 as the id of the cross section plane. This id is guaranteed to not be used yet as the amount of ids
                // that is used is at most the number of generator points n_.

                if (cellptr->nplane(c1, c2, c3, rsq, n_+1)){
                section_indices.push_back(i);
                found++;
                }
            }
            }
            vorocells[i] = (void*)cellptr;
        }
        }
    } while (cla->inc());
    
    delete cla;

    if (found != n_) {  
        for (i = 0; i < n_; i++) {
        if (vorocells[i] == NULL) {
            // kind of hacky: if a cell could not be computed populate the index with the last cell that could be computed
            if (found == 0){
                voronoicell_neighbor v;
                v.init(-1,1,-1,1,-1,1);
                voronoicell_neighbor* cellptr = NULL;
                cellptr = new voronoicell_neighbor();
                *(cellptr) = v;
                vorocells[i] = (void*)cellptr;
            } else {
                cellptr = new voronoicell_neighbor();
                *(cellptr) = cell;
                vorocells[i] = (void*)cellptr;
            }
        }
        }   
    }   

    return make_pair(vorocells, section_indices);
}

vector< vector<bool> > compute_num_fragments(void* container_poly_, int n_) {
    container_poly* con = (container_poly*)container_poly_;
    int i;
    c_loop_all* cla = new c_loop_all(*(con)); 
    voronoicell_neighbor cell;
    voronoicell_neighbor* cellptr = new voronoicell_neighbor();
    double x,y,z,r;
    double epsilon = 0.0000000001;

    vector< vector<bool> > cell_sections(n_, vector<bool>(6, false));

    double c1, c2, c3, rsq, tmp;
    int sgn;
    
    if(cla->start()) do if (con->compute_cell(cell, *(cla))) {

        cla->pos(i, x, y, z, r);
        *(cellptr) = cell;

        // intersect with x = con->ax
        tmp = -2*(x - con->ax);
        sgn = 1;
        if (tmp < 0) sgn = -1;
        if ((x - con->ax < epsilon) && (x - con->ax > -1*epsilon)){
        c1 = sgn; c2 = 0; c3 = 0; rsq = 0;
        } else {
        c1 = tmp; c2 = 0; c3 = 0; rsq = c1*c1;
        }
        if (cellptr->plane_intersects(c1, c2, c3, rsq)){
        cell_sections[i][0] = true;
        }

        // intersect with y = con->ay
        sgn = 1;
        tmp = -2*(y - con->ay);
        if (tmp < 0) sgn = -1;
        if ((y - con->ay < epsilon) && (y - con->ay > -1*epsilon)){
        c1 = 0; c2 = sgn; c3 = 0; rsq = 0;
        } else {
        c1 = 0; c2 = tmp; c3 = 0; rsq = c2*c2;
        }
        if (cellptr->plane_intersects(c1, c2, c3, rsq)){
        cell_sections[i][2] = true;
        }

        // intersect with z = con->az
        sgn = 1;
        tmp = -2*(z - con->az);
        if (tmp < 0) sgn = -1;
        if ((z - con->az < epsilon) && (z - con->az > -1*epsilon)){
        c1 = 0; c2 = 0; c3 = sgn; rsq = 0;
        } else {
        c1 = 0; c2 = 0; c3 = tmp; rsq = c3*c3;
        }
        if (cellptr->plane_intersects(c1, c2, c3, rsq)){
        cell_sections[i][4] = true;
        }

        // intersect with x = con->bx
        sgn = 1;
        tmp = -2*(x-con->bx);
        if (tmp < 0) sgn = -1;
        if ((x - con->bx < epsilon) && (x - con->bx > -1*epsilon)){
        c1 = sgn; c2 = 0; c3 = 0; rsq = 0;
        } else {
        c1 = tmp; c2 = 0; c3 = 0; rsq = c1*c1;
        }
        if (cellptr->plane_intersects(c1, c2, c3, rsq)){
        cell_sections[i][1] = true;
        } 

        // intersect with y = con->by
        sgn = 1;
        tmp = -2*(y-con->by);
        if (tmp < 0) sgn = -1;
        if ((y - con->by < epsilon) && (y - con->by > -1*epsilon)){
        c1 = 0; c2 = sgn; c3 = 0; rsq = 0;
        } else {
        c1 = 0; c2 = tmp; c3 = 0; rsq = c2*c2;
        }
        if (cellptr->plane_intersects(c1, c2, c3, rsq)){
        cell_sections[i][3] = true;
        }  

        // intersect with z = con->bz
        sgn = 1;
        tmp = -2*(z-con->bz);
        if (tmp < 0) sgn = -1;
        if ((z - con->bz < epsilon) && (z - con->bz > -1*epsilon)){
        c1 = 0; c2 = 0; c3 = sgn; rsq = 0;
        } else {
        c1 = 0; c2 = 0; c3 = tmp; rsq = c3*c3;
        }
        if (cellptr->plane_intersects(c1, c2, c3, rsq)){
        cell_sections[i][5] = true;
        } 
        
    } while (cla->inc());

    delete cla;
    delete cellptr;
    
    return cell_sections;
}


/* access methods for retrieving voronoi cell instance data. */
double cell_get_volume(void* cell_) {
    voronoicell_neighbor* cell = (voronoicell_neighbor*)cell_;
    return cell->volume();
}

vector<double> cell_get_centroid(void* cell_) {
    voronoicell_neighbor* cell = (voronoicell_neighbor*)cell_;
    double x,y,z;
    cell->centroid(x, y, z);
    vector<double> cent{x, y, z};
    return cent;
}

vector<double> cell_get_areas(void* cell_) {
    voronoicell_neighbor* cell = (voronoicell_neighbor*)cell_;
    vector<double> areas;
    cell->face_areas(areas);
    return areas;
}

/* input: (x_, y_, z_) the position of the original input point.
 * returns:
 * vector of doubles, coord j of vertex i at ret[i*3 + j]
 */
vector<double> cell_get_vertex_positions(void* cell_, double x_, double y_, double z_) {
    voronoicell_neighbor* cell = (voronoicell_neighbor*)cell_;
    vector<double> positions;
    
    cell->vertices(x_, y_, z_, positions);
    
    return positions;
}

/* NULL-termed list (i) of vector<int>s (j) of vertices adjacent to i. */
void** cell_get_vertex_adjacency(void* cell_) {
    voronoicell_neighbor* cell = (voronoicell_neighbor*)cell_;
    int i, j, v_i_order, num_vertices = cell->p;
    
    void** adjacency = (void**)malloc(sizeof(void*) * (num_vertices + 1));
    vector<int>* vertex_adjacency;
    
    for (i = 0; i < num_vertices; i++) {
        v_i_order = cell->nu[i];
        vertex_adjacency = new vector<int>();
        for (j = 0; j < v_i_order; j++) {
        vertex_adjacency->push_back(cell->ed[i][j]);
        }
        adjacency[i] = (void*)vertex_adjacency;
    }
    adjacency[num_vertices] = NULL;
    
    return adjacency;
}

/* NULL-termed list (i) of vector<int>s of vertices on this face,
 * followed by adjacent cell id. e.g for ret[i]:
 * [2 0 5 7 3 -1 249] for loop 2,0,5,7,3 leading to cell 249.
 */
void** cell_get_faces(void* cell_) {
    voronoicell_neighbor* cell = (voronoicell_neighbor*)cell_;
    int i, j, f_i_order, num_faces = cell->number_of_faces();
    
    void** faces = (void**)malloc(sizeof(void*) * (num_faces + 1));
    vector<int> vertices;
    vector<int> neighbours;
    vector<int>* output_list = NULL;
    
    cell->neighbors(neighbours);
    cell->face_vertices(vertices);
    for (i = 0; i < num_faces; i++) {
        f_i_order = vertices[0];
        output_list = new vector<int>();
        for (j = 1; j <= f_i_order; j++) {
        output_list->push_back(vertices[j]);
        }
        output_list->push_back(neighbours[i]);
        vertices.erase(vertices.begin(),vertices.begin()+f_i_order+1);
        faces[i] = (void*)output_list;
    }
    faces[num_faces] = NULL;
    
    return faces;
}


void dispose_container(void* container_poly_){
     delete (container_poly*)container_poly_;
}


void dispose_all(void* container_poly_, void** vorocells, int n_) {
    delete (container_poly*)container_poly_;
    
    if (vorocells == NULL) return;
    
    int i;
    for (i = 0; i < n_; i++) {
        delete (voronoicell_neighbor*)vorocells[i]; 
    }
    free(vorocells);
}


void dispose_cells(void** vorocells, int n_) {
    if (vorocells == NULL) return;
    
    int i;
    for (i = 0; i < n_; i++) {
        delete (voronoicell_neighbor*)vorocells[i];
    }
    
    free(vorocells);
}

