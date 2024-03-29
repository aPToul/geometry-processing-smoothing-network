#ifndef SMOOTH_H
#define SMOOTH_H
#include <Eigen/Core>
#include <Eigen/Sparse>
// Given a mesh (`V`,`F`) and data specified per-vertex (`G`), smooth this data
// using a single implicit Laplacian smoothing step.
//
// Inputs:
//   V  #V by 3 list of mesh vertex positions
//   F  #F by 3 list of mesh triangle indices into V
//   G  #V by dim list of per-vertex data: could be scalar- (`dim =1`) or
//     vector-valued (`dim>1`).
//   lambda  >0 smoothing parameter also known as the "time step" `dt`
// Outputs: 
//   U  #V by dim list of smoothed data
//
void smooth(
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & E,
    const Eigen::MatrixXd & G,
    double lambda,
    Eigen::MatrixXd & U,
    int mode);
#endif
