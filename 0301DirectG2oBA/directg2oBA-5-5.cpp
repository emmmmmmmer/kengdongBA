//
// Created by xiang on 1/4/18.
// this program shows how to perform direct bundle adjustment
//
#include <iostream>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>



#include <Eigen/Dense>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

#include <pangolin/pangolin.h>
#include <boost/format.hpp>

using namespace std;
using namespace Eigen;

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> VecSE3;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVec3d;

// global variables
string pose_file = "/home/lu/code/0301DirectG2oBA/rgbd_dataset_freiburg1_xyz_results.txt";      //VO       切三份  滑窗 40+10
string points_file = "/home/lu/code/0301DirectG2oBA/3D_int.txt";        //改成具体的路径就可以         //3D

// intrinsics
float fx = 100.0;
float fy = 110.0;
float cx = 312.234;
float cy = 239.777;
/*double cx = 325.5;
double cy = 253.5;
double fx = 518.0;
double fy = 519.0;*/

// bilinear interpolation   双线性差值
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);    //floor() 方法将数字向下舍入为最接近的整数,并返回结果。
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
            );
}

// g2o vertex that use sophus::SE3d as pose
class VertexSophus : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexSophus() {}

    ~VertexSophus() {}

    bool read(std::istream &is) {}

    bool write(std::ostream &os) const {}

    virtual void setToOriginImpl() {
        _estimate = Sophus::SE3d();
        //std::cout<<&_estimate<<endl;
    }

    virtual void oplusImpl(const double *update_) {         //oplusImpl(计算下一次的估计值，相当于一次setEstimate)：函数处理的是 xk+1 = xk + ∆x 的过程；
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> update(update_);
        setEstimate(Sophus::SE3d::exp(update) * estimate());
    }
};

// TODO edge of projection error, implement it
// 16x1 error, which is the errors in patch
typedef Eigen::Matrix<double,16,1> Vector16d;
class EdgeDirectProjection : public g2o::BaseBinaryEdge<16, Vector16d, g2o::VertexPointXYZ, VertexSophus> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeDirectProjection(float *color, cv::Mat &target) {
        this->origColor = color;
        this->targetImg = target;
    }

    ~EdgeDirectProjection() {}
    //computeError(返回：_error)：边的误差项，观测值与估计值的差距；(曲线拟合中y的测量值与估计值的差)。
    virtual void computeError() override {
        // TODO START YOUR CODE HERE
        // compute projection error ...
        const g2o::VertexPointXYZ* vertex_point = static_cast<const g2o::VertexPointXYZ* >(vertex(0));
        const VertexSophus* vertex_pose = static_cast<const VertexSophus* >(vertex(1));
        Eigen::Vector3d XYZ = vertex_pose->estimate() * vertex_point->estimate();
        float u = XYZ[0]*fx/XYZ[2] + cx;
        float v = XYZ[1]*fy/XYZ[2] + cy;
        for (int i = 0; i < 16; i++) {
            int m = i/4;
            int n = i%4;
            if (u - 2 < 0 || v - 2 < 0 || (u + 1) > targetImg.cols || (v + 1) > targetImg.rows) {
                _error[i] = 0;
                this->setLevel(1);
            }else{
                _error[i] = origColor[i] - GetPixelValue(targetImg, u - 2 + m, v - 2 + n);
            }
        }
        // END YOUR CODE HERE
    }

    // Let g2o compute jacobian for you

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}

private:
    cv::Mat targetImg;  // the target image
    float *origColor = nullptr;   // 16 floats, the color of this point
};

// plot the poses and points for you, need pangolin


int main(int argc, char **argv) {

    // read poses and points
    VecSE3 poses,poses0;
    VecVec3d points,points0;
    ifstream fin(pose_file);
    //double timestamp = 0;
    vector<double> time(0);
    while (!fin.eof()) {
        double timestamp = 0;
        fin >> timestamp;
        if (timestamp == 0) break;
        double data[7];
        for (auto &d: data) fin >> d;
        poses.push_back(Sophus::SE3d(
                Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                Eigen::Vector3d(data[0], data[1], data[2])
                ));
        if (!fin.good()) break;
        //std::cout<< std::fixed << std::showpoint << timestamp <<"time:"<<endl;      //将这里的时间timestamp提取出来放到优化结果前面
        //time存放了时间戳   fixed 和 showpoint 让输出的数值不要科学计数法

        //std::cout<< std::fixed << std::showpoint << time <<"time11:"<<endl;
    }
    for (int q=0;q< time.size();q++){
        cout<<std::fixed << std::showpoint<<time[q]<<"    timestamp"<<q<<endl;
    }
    fin.close();
    //std::cout<<"1"<<endl;


    vector<float *> color;
    fin.open(points_file);
    //std::cout<<"1"<<endl;
    while (!fin.eof()) {
        double xyz[3] = {0};
        for (int i = 0; i < 3; i++) fin >> xyz[i];
        if (xyz[0] == 0) break;
        points.push_back(Eigen::Vector3d(xyz[0], xyz[1], xyz[2]));
        float *c = new float[16];
        for (int i = 0; i < 16; i++) fin >> c[i];
        color.push_back(c);
        //std::cout<<"1"<<endl;
        if (fin.good() == false) break;     //这里有问题         像素值没有小数点，这里李代数就会报错
        //std::cout<<"1"<<endl;
    }
    ////////////points文件行数太多就会终端程序
    fin.close();

    cout << "poses: " << poses.size()  << ", points: " << points.size() << endl;
    
    poses0.assign(poses.begin(),poses.end());
    points0.assign(points.begin(),points.end());
    
    // read images
    vector<cv::Mat> images;
    boost::format fmt("/home/lu/code/0301DirectG2oBA/color/%d.png");
    for (int i = 0; i < 798; i++) {
        images.push_back(cv::imread((fmt % i).str(), 0));
        /*cv::Mat image;
        image = images[i];
        cv::imshow("11",image);*/
    }

    // pose dimension 6, landmark is 3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    // use LM
    auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // TODO add vertices, edges into the graph optimizer
    // START YOUR CODE HERE
    //添加3D点节点,设为0,并从3D点开始计数
    for (size_t i = 0; i < points.size(); i++) {
        g2o::VertexPointXYZ* vertex_point = new g2o::VertexPointXYZ();
        vertex_point->setId(i);
        vertex_point->setEstimate(points[i]);
        vertex_point->setMarginalized(true);
        optimizer.addVertex(vertex_point);
    }
    //添加位姿节点,设为1
    for (size_t i = 0; i < poses.size()  ; i++) {
        VertexSophus* vertex_pose = new VertexSophus();
        vertex_pose->setId(i + points.size());
        if (i == 0 ) vertex_pose->setFixed(true);
        vertex_pose->setEstimate(poses[i]);
        optimizer.addVertex(vertex_pose);
    }
    //添加边
    int index = 1;
    vector<EdgeDirectProjection*> edges;
    for (size_t i = 0; i < poses.size() ; i++) {
        for (size_t j = 0; j < points.size(); j++) {
            EdgeDirectProjection* edge = new EdgeDirectProjection( color[j], images[i] );
            edge->setId(index);
            edge->setVertex(0, dynamic_cast<g2o::VertexPointXYZ*>(optimizer.vertex(j)));
            edge->setVertex(1, dynamic_cast<VertexSophus*>
            (optimizer.vertex(i+points.size())));
            edge->setInformation(Eigen::Matrix<double,16,16>::Identity());
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            rk->setDelta(0.8);
            edge->setRobustKernel(rk);
            optimizer.addEdge(edge);
            index++;
            edges.push_back(edge);
        }
    }
    // END YOUR CODE HERE

    // perform optimization
    optimizer.initializeOptimization(0);
    optimizer.optimize(200);        //迭代次数       int 的lambda值不好缩小

    // TODO fetch data from the optimizer
    // START YOUR CODE HERE
    int poses_number = poses.size() ;            //pose太多也会报错
    double points_number = points.size();        //int型在2W行的时候就报错，改成float   double 7W行
    poses.clear();
    points.clear();
    for (size_t i = 0; i < points_number; i++) {
        g2o::VertexPointXYZ* point_XYZ_new =
                dynamic_cast<g2o::VertexPointXYZ* >(optimizer.vertex(i));
        Eigen::Vector3d point = point_XYZ_new->estimate();
        points.push_back(point);
    }
    for (size_t i = 0; i < poses_number; i++) {
        VertexSophus *pose_vertex_new = dynamic_cast<VertexSophus* >(optimizer.vertex(i
                + points_number));
        Sophus::SE3d pose = pose_vertex_new->estimate();
        //cout<<"T1="<<endl<<pose_vertex_new->estimate().matrix()<<endl;
        Eigen::Matrix<double,3,3,Eigen::RowMajor> R = pose.matrix().block<3,3>(0,0);
        Eigen::Matrix<double,3,Eigen::RowMajor> t = pose.matrix().block<3,1>(0,3);


        Eigen::Quaternion<double> quat(R);
        //Eigen::Quaternion <double>R={quat.w(),quat.x(),quat.y(),quat.z()};
        std::cout<< std::fixed << std::showpoint <<t[0]<<" "<<t[1]<<" "<<t[2]<<" "<<quat.x()<<" "<<quat.y()<<" "<<quat.z()<<" "<<quat.w()<<endl;
        //cout<<"T2="<<endl<<pose.matrix()<<endl;          //输出BA后的位子
        poses.push_back(pose);
    }

    // END YOUR CODE HERE

    // plot the optimized points and poses

    // delete color data
    for (auto &c: color) delete[] c;

    return 0;
}
//*


// */
