#include "smooth.h"
#include <iostream>
#include <Eigen/SparseCholesky>
#include <igl/edges.h>

using namespace std;

void graph_Laplacian(
  const Eigen::MatrixXi & E,
  Eigen::SparseMatrix<double> & L)
{
  typedef Eigen::Triplet<double> T;

  std::vector<T> tripletList;
  // For each edge, two elements of L are filled in with + 1
  // We will add the diagonal elements after
  tripletList.reserve(E.rows() * 2);

  for(int edge_number = 0; edge_number < E.rows(); edge_number++)
  {
    auto start_node_index = E(edge_number, 0);
    auto end_node_index = E(edge_number, 1);
    
    tripletList.push_back(T(start_node_index, end_node_index, 1.0));
    tripletList.push_back(T(end_node_index, start_node_index, 1.0));
  }
  L.setFromTriplets(tripletList.begin(), tripletList.end());
  
  // Set up Laplacian equality: what leaves a node, enters it
  for (int diagIndex = 0; diagIndex < L.rows(); diagIndex++){
    L.coeffRef(diagIndex, diagIndex) = -1 * L.row(diagIndex).sum();
  }
}

void edge_weighted_Laplacian(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  Eigen::SparseMatrix<double> & L)
{
  // Unknown effect of added distance to all points
  double epsilon = 0.00; // 0.01 

  // Unknown effect of lower bounding the edge lengths
  double edge_threshold = 0.0000; // 0.0001

  // For debugging
  double all_edge_differences = 0;
  int edges_added = 0;
  // Slightly redundant since logic is now edge-wise
  // Will simply overwrite each half-edge double-visited
  for (int edgeIndex = 0; edgeIndex < E.rows(); edgeIndex++){
    auto edge = E.row(edgeIndex);
    int source = edge[0];
    int target = edge[1];
    double distance = (V.row(source)- V.row(target)).norm();

    if (distance < 0)
    {
      cout << "Assumption of positive side length violated!" << endl;
    }

    if (L.coeffRef(source, target) == 0)
    {
      if (distance > edge_threshold)
      {
        // Side 1 and 2
        L.coeffRef(source, target) = 1.0 / (distance + epsilon);
        L.coeffRef(target, source) = 1.0 / (distance + epsilon);
        all_edge_differences += distance;
        edges_added += 1;        
      }
    }
  }

  if (edge_threshold > 0){
    cout << "Edges added: " << edges_added << endl;    
  }

  if (epsilon > 0){
    cout << "Average edge length: " << all_edge_differences / edges_added << endl;
  }

  // Set up Laplacian equality: what leaves a node, enters it
  for (int diagIndex = 0; diagIndex < L.rows(); diagIndex++){
    // for some reason cannot actually edit the .diagonal()
    L.coeffRef(diagIndex, diagIndex) = -1 * L.row(diagIndex).sum();
  }
}

void smooth(
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & E,
    const Eigen::MatrixXd & G,
    double lambda,
    Eigen::MatrixXd & U,
    int mode)
{
  int number_of_vertices = V.rows();

  Eigen::SparseMatrix<double>Laplacian(number_of_vertices, number_of_vertices);
  Eigen::DiagonalMatrix<double,Eigen::Dynamic> Mass(number_of_vertices);

  if (mode == 0){
    graph_Laplacian(E, Laplacian);
    Mass.setIdentity(number_of_vertices);
  }
  else if (mode == 1){
    edge_weighted_Laplacian(V, E, Laplacian);
    Mass.setIdentity(number_of_vertices);
  }

  Eigen::SparseMatrix<double> healthyA;
  // Multiply by the step-size lambda
  healthyA = - lambda * Laplacian;
  // Add in mass, which preserves symmetry
  for (int diagIndex = 0; diagIndex < Mass.rows(); diagIndex++){
    healthyA.coeffRef(diagIndex, diagIndex) += Mass.diagonal()[diagIndex];
  }

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(healthyA);
  U = solver.solve(Mass * G);
}