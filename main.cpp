#include "smooth.h"
#include <igl/readDMAT.h>
#include <igl/parula.h>
#include <igl/viewer/Viewer.h>
#include <Eigen/Core>
#include <string>
#include <iostream>
#include <igl/list_to_matrix.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>

using namespace std;

int main(int argc, char *argv[])
{
  // Smoothing mode
  int mode = 1;
  // Lambda for smoothing dynamics
  double lambda = 1e-2;

  int reset_network = 1;
  int community_number = 0;
  int labels_on = 0;
  
  // Original vertices and (fixed) edges
  // Initialize to something, will be modified
  Eigen::MatrixXd OV = Eigen::MatrixXd::Zero(1, 1);
  Eigen::MatrixXi E = Eigen::MatrixXi::Zero(1, 1);

  // Original data
  Eigen::MatrixXd G;

  // Smoothed data and positions
  Eigen::MatrixXd V, U;
  // Not used in network
  Eigen::MatrixXi F;

  igl::viewer::Viewer viewer;
  std::cout<<R"(
  C,c      [DEFAULT Community 1] Increment to next Community number.
  L,l      [DEFAULT Off] Turn labels On/Off.
  D,d  smooth data
  K    10x lambda
  k    10% lambda
  M,m  smooth mesh geometry
  R,r  reset mesh geometry and data
  g    Graph Laplacian
  e    [DEFAULT] Edge-weighted Graph Laplacian
)";

  //// Loads Community
  const auto create_network = [&](int labels_on, int community_number, Eigen::MatrixXd & V_, Eigen::MatrixXi & E_, Eigen::MatrixXd & G, int reset_network)
  {
    viewer.data.clear();

    string root_folder = "../../redditCommunities/" + std::to_string(community_number + 1) + "/";

    // Load data
    if (reset_network == 1){
      cout << "Loading data" << endl;
      Eigen::MatrixXd V;
      // Load in points from file
      {
        Eigen::MatrixXd D;
        std::vector<std::vector<double> > vD;
        std::string line;
        std::fstream in;
        in.open(argc>1?argv[1]: root_folder +  "P.txt");
        while(in)
        {
          std::getline(in, line);
          std::vector<double> row;
          std::stringstream stream_line(line);
          double value;
          while(stream_line >> value) row.push_back(value);
          if(!row.empty()) vD.push_back(row);
        }
        igl::list_to_matrix(vD,D);
        assert(D.cols() == 3 && "Position file should have 3 columns");
        V = D.leftCols(3);
      }
      // cout << "Number of points: " << V.rows() << endl;

      Eigen::MatrixXi E;
      // Load in edges from file
      {
        Eigen::MatrixXi D;
        std::vector<std::vector<int> > vI;
        std::string line;
        std::fstream in;
        in.open(argc>1?argv[1]: root_folder + "E.txt");
        while(in)
        {
          std::getline(in, line);
          std::vector<int> row;
          std::stringstream stream_line(line);
          int value;
          while(stream_line >> value) row.push_back(value);
          if(!row.empty()) vI.push_back(row);
        }
        igl::list_to_matrix(vI,D);
        assert(D.cols() == 2 && "Edges file should have 2 columns");
        E = D.leftCols(2);
      }
      // cout << "Number of edges: " << E.rows() << endl;

      V_ = V;
      E_ = E;
    }

    // Add points
    Eigen::MatrixXd C;
    if (G.rows() > 0){
      igl::parula(U, G.minCoeff(), G.maxCoeff(), C);
      viewer.data.set_points(V_, C);
      // viewer.data.set_colors(C);      
    }
    else
    {
      Eigen::RowVector3d point_color = Eigen::RowVector3d(0, 0, 0);
      viewer.data.set_points(V_, point_color);
    }

    // Add edges
    Eigen::RowVector3d edge_color = Eigen::RowVector3d(0, 0, 0.5);
    viewer.data.set_edges(V_, E_, edge_color);

    if (labels_on == 1){
      // cout << "Setting labels" << endl;
      // Load in labels
      std::string line;
      std::fstream in;
      in.open(argc>1?argv[1]: root_folder + "L.txt");
      int label_count = 0;
      while(in)
      {
        std::getline(in, line);
        if (!line.empty()){
          // cout << V.row(label_count) << endl;
          // cout << line << endl;
          viewer.data.add_label(V_.row(label_count), line);
          label_count += 1;
        }
      }
      assert(V_.rows() == label_count && "Labels should be 1-1 with points");      
    }
  };

  // Updates visualization
  const auto & update = [&]()
  {
    if((V.array() != V.array()).any())
    {
      std::cout<<"Too degenerate to keep smoothing. Better reset"<<std::endl;
    }
    reset_network = 0;
    create_network(labels_on, community_number, V, E, G, reset_network);
  };
  // Resets data and points
  const auto & reset = [&]()
  {
    V = OV;
    U = G;
  };


  viewer.callback_key_pressed = 
    [&](igl::viewer::Viewer &, unsigned int key, int)
  {
    switch(key)
    {
      case 'L':
      case 'l':
        labels_on = (labels_on + 1) % 2;
        reset_network = 0;
        create_network(labels_on, community_number, V, E, G, reset_network);
        return true;
      case 'c':
      case 'C':
        community_number = (community_number + 1) % 5;
        cout << "Community " << community_number + 1 << endl;
        reset_network = 1;
        // Will load a fresh network and correctly set OV
        create_network(labels_on, community_number, OV, E, G, reset_network);
        // Just make the data signal the y-coordinate of the data
        G = OV.col(1);
        if(argc>2)
        {
          if(argv[2][0] == 'n')
          {
            // Corrupt with a bit of noise
            G += 0.1*(G.maxCoeff()-G.minCoeff())*
              Eigen::MatrixXd::Random(G.rows(),G.cols());
          }else
          {
            igl::readDMAT(argv[2],G);
          }
        }
        reset();
        update();
        reset_network = 0;
        return true;
      case 'D':
      case 'd':
        //////////////////////////////////////////////////////////////////////
        // Smooth data
        //////////////////////////////////////////////////////////////////////
        // Use copy constructor to fake in-place API (may be overly
        // conservative depending on your implementation)
        smooth(V, E ,Eigen::MatrixXd(U),lambda,U,mode);
        break;
      case 'K':
      case 'k':
        lambda = (key=='K'?10.0:0.1)*lambda;
        std::cout<<"lambda: "<<lambda<<std::endl;
        break;
      case 'M':
      case 'm':
      {
        //////////////////////////////////////////////////////////////////////
        // Smooth mesh geometry. 
        //////////////////////////////////////////////////////////////////////
        // "Linearize" simply by conducting smooth step assuming that vertex
        // data is a signal defined over current surface: copy is needed to
        // prevent memory "aliasing"
        Eigen::MatrixXd Vcpy(V);
        smooth(Vcpy, E ,Vcpy,lambda,V,mode);
        // cout << "Done smoothing" << endl;
        break;
      }
      case 'g':
      {
        mode = 0;
        break;
      }
      case 'e':
      {
        mode = 1;
        break;
      }
      case 'R':
      case 'r':
        reset();
        break;
      default:
        return false;
    }
    update();
    return true;
  };

  reset_network = 1;
  create_network(labels_on, community_number, OV, E, G, reset_network);
  // Just make the data signal the y-coordinate of the data
  G = OV.col(1);
  if(argc>2)
  {
    if(argv[2][0] == 'n')
    {
      // Corrupt with a bit of noise
      G += 0.1*(G.maxCoeff()-G.minCoeff())*
        Eigen::MatrixXd::Random(G.rows(),G.cols());
    }else
    {
      igl::readDMAT(argv[2],G);
    }
  }
  reset();
  update();

  // Set Sizes
  viewer.core.point_size = 20.0;
  viewer.core.line_width = 25.0f;
  viewer.launch();

  return EXIT_SUCCESS;
}
