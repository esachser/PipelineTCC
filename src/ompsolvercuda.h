#include <eigen3/Eigen/Eigen>
// #include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <mkl.h>
#include <cublas_v2.h>
#include <iostream>

// #define ENABLEPRINT

#ifdef ENABLEPRINT
#define printval(header, divisor, value, unidade) {std::cout << header << divisor << value << unidade << std::endl;}
#else
#define printval(header, divisor, value, unidade)
#endif

extern "C" void matching_pursuit_init(int m_D, int n_D, float * h_D,
                                 int n_X,
                                 float epsilon,
                                 int _L,
                                 cudaError_t * error);

extern "C" void matching_pursuit_set_vectors(int m_D, int n_D,
                                 int n_X, const float * h_X, const int * h_calc);

extern "C" void matching_pursuit_solve(int m_D, int n_D,int n_X, cudaError_t * error);

extern "C" void matching_pursuit_get_results(const int * rms, const int * idms, const float *vals);

extern "C" void matching_pursuit_destroy(cudaError_t * error);

extern "C" void sendimage(unsigned char* img, int rows, int cols, int rp, int cp);

/// ------------------------------------------------------------
/// Class implementing OMP with allocation at start
class OMPSolverCUDAEigen {
    public:
        OMPSolverCUDAEigen(int num_patches, Eigen::MatrixXf & D, int L, float eps, float lambda, int num_threads=-1);
        void solve(const Eigen::MatrixXf& X, const Eigen::VectorXi& calc);
        void solve(unsigned char *img, int rows, int cols, int rp, int cp);
        void getResults(Eigen::MatrixXf& spalpha);
        void getResults(Eigen::MatrixXf& spalpha, int maxquality);
        void decode(Eigen::MatrixXf& res, int maxquality);
        
        ~OMPSolverCUDAEigen();
    private:
        int _npatches;
        Eigen::MatrixXf& _D;
        int _L;
        float _eps;
        float _lambda;
        int _NUM_THREADS;
        bool _transformed;
        float _minval;
        float _ptp;

        // Resultados
        Eigen::MatrixXi _rM;

        Eigen::MatrixXf _mresults;
        Eigen::MatrixXi _mresultsint;
        Eigen::VectorXi _idxm;

        void mksolve();
};


OMPSolverCUDAEigen::OMPSolverCUDAEigen(int num_patches, Eigen::MatrixXf & D, int L, float eps, float lambda, int num_threads) :
    _npatches(num_patches), _D(D), _L(L), _eps(eps), _lambda(lambda)
{
    int K = _D.cols();
    _L = std::min(_L, K);

    _rM.resize(_L, _npatches);

    _mresults.resize(_L*_L, _npatches);
    _mresultsint.resize(_L*_L, _npatches);
    _idxm.resize(_npatches);


    cudaError_t error;
    matching_pursuit_init(_D.rows(), K, _D.data(), _npatches, _eps, _L, &error);
    if (error != cudaSuccess){
        std::cerr << "Erro na inicialização!" << std::endl;
        exit(-1);
    }
}

void OMPSolverCUDAEigen::mksolve(){
    cudaError_t error;
    matching_pursuit_solve(_D.rows(), _D.cols(), _npatches, &error);
    if (error != cudaSuccess){
        std::cerr << "Erro no processamento!" << std::endl;
        exit(-1);
    }

    matching_pursuit_get_results(_rM.data(), _idxm.data(), _mresults.data());

    // Para testes, faz quantização.
    auto max = _mresults.maxCoeff();
    auto min = _mresults.minCoeff();
    auto ptp = max - min;
    _mresultsint = ((_mresults.array() - min) * 255 / ptp + 0.5).cast <int> ();

    _minval = min;
    _ptp = ptp;
}


void OMPSolverCUDAEigen::solve(const Eigen::MatrixXf & X, const Eigen::VectorXi& calc) {
    matching_pursuit_set_vectors(_D.rows(), _D.cols(), _npatches, X.data(), calc.data());
    mksolve();
}

void OMPSolverCUDAEigen::solve(unsigned char *img, int rows, int cols, int rp, int cp){
    sendimage(img, rows, cols, rp, cp);    
    mksolve();
}


void OMPSolverCUDAEigen::getResults(Eigen::MatrixXf& spalpha, int maxquality){

    // maxquality = std::max(1, maxquality);
    // maxquality = std::min(maxquality, _L);
    // _transformed = false;
    // spalpha.setZero();
    // for (int j=0; j<_npatches; j++){
    //     int idmax = std::min(maxquality-1, _idxm[j]);
    //     for (int i=0; i<_L; i++){
    //         auto idx = _rM(i,j);
    //         if (idx>=0) spalpha(idx,j) = _mresults[j](i,idmax);
    //         else break;
    //     }
    // }
}

void OMPSolverCUDAEigen::decode(Eigen::MatrixXf& res, int maxquality){
    maxquality = std::max(1, maxquality);
    maxquality = std::min(maxquality, _L);
    _transformed = false;

    res.setZero();

    
    // _mresults = (_mresults * ptp / 255).array() + min;

    for (int j=0; j<_npatches; j++){
        int idmax = std::min(maxquality-1, _idxm[j]);
        auto resc = res.col(j);
        auto mresult = _mresultsint.col(j).segment(_L*idmax, _L);
        for (int i=0; i<_L; i++){
            auto idx = _rM(i,j);
            // if (idx>=0) resc += _mresults[j](i,idmax) * _D.col(idx);
            if (idx>=0) resc += (mresult[i]*_ptp / 255 + _minval) * _D.col(idx);
            else break;
        }
        // if(idmax==maxquality-1){
        // mresult.resize(_L, _L);
        // std::cout << idmax << std::endl;
        // std::cout << mresult << std::endl;
        // std::cout << _mresults.col(j) << std::endl;
        // exit(0);}
    }
}

void OMPSolverCUDAEigen::getResults(Eigen::MatrixXf& spalpha){
    this->getResults(spalpha, _L);
}


OMPSolverCUDAEigen::~OMPSolverCUDAEigen(){

}