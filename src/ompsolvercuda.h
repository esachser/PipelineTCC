#include <eigen3/Eigen/Eigen>
// #include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <mkl.h>
#include <cublas_v2.h>
#include <iostream>
#include <sstream>

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

extern "C" void matching_pursuit_get_results(const int * rms, const int * idms, const float *vals, const int *calc);

extern "C" void matching_pursuit_destroy(cudaError_t * error);

extern "C" void sendimage(unsigned char* img, int rows, int cols, int rp, int cp);

/// ------------------------------------------------------------
/// Class implementing OMP with allocation at start
class OMPSolverCUDAEigen {
    public:
        OMPSolverCUDAEigen(int num_patches, Eigen::MatrixXf & D, int L, float eps, float lambda, int num_threads=-1);
        void solve(const Eigen::MatrixXf& X, const Eigen::VectorXi& calc);
        void solve(unsigned char *img, int rows, int cols, int rp, int cp);
        void solve(unsigned char *img, int rows, int cols, int rp, int cp, std::stringstream& strfile);
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
        Eigen::VectorXi _calc;

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
    _calc.resize(_npatches);


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

    // auto tic = std::chrono::system_clock::now();
    matching_pursuit_get_results(_rM.data(), _idxm.data(), _mresults.data(), _calc.data());
    // auto tac = std::chrono::system_clock::now();
    // auto tictac = std::chrono::duration_cast<std::chrono::milliseconds>(tac-tic).count();
    // printval("Getres time", ": ", tictac, "ms");

    // Para testes, faz quantização.
    // tic = std::chrono::system_clock::now();
    auto max = _mresults.maxCoeff();
    auto min = _mresults.minCoeff();
    auto ptp = max - min;
    _mresultsint = ((_mresults.array() - min) * 65535 / ptp + 0.5).cast <int> ();

    _minval = min;
    _ptp = ptp;
    // tac = std::chrono::system_clock::now();
    // tictac = std::chrono::duration_cast<std::chrono::milliseconds>(tac-tic).count();
    // printval("Quantization time", ": ", tictac, "ms");
}


void OMPSolverCUDAEigen::solve(const Eigen::MatrixXf & X, const Eigen::VectorXi& calc) {
    matching_pursuit_set_vectors(_D.rows(), _D.cols(), _npatches, X.data(), calc.data());
    mksolve();
}

void OMPSolverCUDAEigen::solve(unsigned char *img, int rows, int cols, int rp, int cp){
    auto tic = std::chrono::system_clock::now();
    sendimage(img, rows, cols, rp, cp);    
    auto tac = std::chrono::system_clock::now();
    auto tictac = std::chrono::duration_cast<std::chrono::milliseconds>(tac-tic).count();
    printval("SendImage time", ": ", tictac, "ms");
    mksolve();
}
void OMPSolverCUDAEigen::solve(unsigned char *img, int rows, int cols, int rp, int cp, std::stringstream& strfile){
    auto tic = std::chrono::system_clock::now();
    sendimage(img, rows, cols, rp, cp);    
    mksolve();
    auto tac = std::chrono::system_clock::now();
    auto tictac = std::chrono::duration_cast<std::chrono::milliseconds>(tac-tic).count();
    printval("Calculation time", ": ", tictac, "ms");
    std::cout << tictac << " ";

    tic = std::chrono::system_clock::now();
    // // Grava cfe TCC escrito
    // uint16_t cnt = 0;
    // int8_t mone = -1;
    // int8_t mtwo = -2;
    // strfile.write((const char *)(&_minval), sizeof(_minval));
    // strfile.write((const char *)(&_ptp), sizeof(_ptp));

    // for (int j=0; j<_npatches; j++){
    //     if (_calc[j] && cnt==0){
    //         // Incrementa contagem
    //         cnt+=1;
    //         continue;
    //     } else if(cnt>0){
    //         // se cnt==1, escreve -1, senão, -2 e cnt, sempre nos índices
    //         // zera o contador
    //         if (cnt==1) strfile.write((const char *)(&mone), sizeof(char));
    //         else{
    //             strfile.write((const char *)(&mtwo), sizeof(char));
    //             strfile.write((const char *)(&cnt), sizeof(cnt));
    //         }
    //     }
    //     cnt = 0;
    //     int idmax = std::min(_L-1, _idxm[j]);
    //     auto mresult = _mresultsint.col(j).segment(_L*idmax, _L);
    //     for (int i=0; i<_L; i++){
    //         int8_t idx = _rM(i,j);
    //         if (idx<0){
    //             strfile.write((const char *)(&mone), sizeof(char));
    //             break;
    //         }
    //         // uint8_t val = mresult(i);
    //         // strfile.write((const char *)(&val), sizeof(val));
    //     }
    // }
    // if(cnt>0){
    //     // se cnt==1, escreve -1, senão, -2 e cnt, sempre nos índices
    //     // zera o contador
    //     if (cnt==1) strfile.write((const char *)(&mone), sizeof(char));
    //     else{
    //         strfile.write((const char *)(&mtwo), sizeof(char));
    //         strfile.write((const char *)(&cnt), sizeof(cnt));
    //     }
    // }

    // for (int j=0; j<_npatches; j++){
    //     int idmax = std::min(_L-1, _idxm[j]);
    //     auto mresult = _mresultsint.col(j).segment(_L*idmax, _L);
    //     for (int i=0; i<_L; i++){
    //         int8_t idx = _rM(i,j);
    //         if (idx<0){
    //             break;
    //         }
    //         uint8_t val = mresult(i);
    //         strfile.write((const char *)(&val), sizeof(val));
    //     }
    // }
    tac = std::chrono::system_clock::now();
    tictac = std::chrono::duration_cast<std::chrono::milliseconds>(tac-tic).count();
    printval("Save time", ": ", tictac, "ms");
    std::cout << tictac << " ";
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


    for (int j=0; j<_npatches; j++){
        int idmax = std::min(maxquality-1, _idxm[j]);
        auto resc = res.col(j);
        auto mresult = _mresultsint.col(j).segment(_L*idmax, _L);
        auto mresultf = _mresults.col(j).segment(_L*idmax, _L);
        for (int i=0; i<_L; i++){
            auto idx = _rM(i,j);
            // if (idx>=0) resc += mresultf[i] * _D.col(idx);
            if (idx>=0) resc += (mresult[i]*_ptp / 65535 + _minval) * _D.col(idx);
            else break;
        }
    }
}

void OMPSolverCUDAEigen::getResults(Eigen::MatrixXf& spalpha){
    this->getResults(spalpha, _L);
}


OMPSolverCUDAEigen::~OMPSolverCUDAEigen(){

}