
#include <eigen3/Eigen/Eigen>
// #include <eigen3/Eigen/Dense>
#include <chrono>

#define ENABLEPRINT

#ifdef ENABLEPRINT
#define printval(header, divisor, value, unidade) {std::cout << header << divisor << value << unidade << std::endl;}
#else
#define printval(header, divisor, value, unidade)
#endif

#define MAX_THREADS 64

static inline int init_omp(const int numThreads) {
   int NUM_THREADS;
#ifdef _OPENMP
   NUM_THREADS = (numThreads == -1) ? MIN(MAX_THREADS,omp_get_num_procs()) : numThreads;
   omp_set_nested(0);
   omp_set_dynamic(0);
   omp_set_num_threads(NUM_THREADS);
#else
   NUM_THREADS = 1;
#endif
   return NUM_THREADS;
}

/// ------------------------------------------------------------
/// Class implementing OMP with allocation at start
class OMPSolverEigen {
    public:
        OMPSolverEigen(int num_patches, Eigen::MatrixXf & D, int L, float eps, float lambda, int num_threads=-1);
        void solve(const Eigen::MatrixXf& X);
        void getResults(Eigen::MatrixXf& spalpha);
        void transform(float lowval, float higval);
        void transform0(float ptp);
        void roundValues();
        
        ~OMPSolverEigen();
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

        // Auxiliares
        Eigen::MatrixXf _G;

        Eigen::VectorXf* _scoresT;
        Eigen::VectorXf* _normT;
        Eigen::VectorXf* _tmpT;
        Eigen::VectorXf* _RdnT;
        Eigen::MatrixXf* _UnT;
        Eigen::MatrixXf* _UndnT;
        Eigen::MatrixXf* _GsT;

        // Resultados
        Eigen::MatrixXi _rM;
        Eigen::MatrixXf _vM;
};


OMPSolverEigen::OMPSolverEigen(int num_patches, Eigen::MatrixXf & D, int L, float eps, float lambda, int num_threads) :
    _npatches(num_patches), _D(D), _L(L), _eps(eps), _lambda(lambda)
{
    int K = _D.cols();
    _L = std::min(_L, K);
    _G = Eigen::MatrixXf(K, K);
    _G << _D.transpose() * _D;


    _NUM_THREADS = init_omp(num_threads);
    _transformed = false;
    _scoresT=new Eigen::VectorXf[_NUM_THREADS];
    _normT=new Eigen::VectorXf[_NUM_THREADS];
    _tmpT=new Eigen::VectorXf[_NUM_THREADS];
    _RdnT=new Eigen::VectorXf[_NUM_THREADS];
    _UnT=new Eigen::MatrixXf[_NUM_THREADS];
    _UndnT=new Eigen::MatrixXf[_NUM_THREADS];
    _GsT=new Eigen::MatrixXf[_NUM_THREADS];
    for (int i = 0; i<_NUM_THREADS; ++i) {
        _scoresT[i].resize(K);
        _normT[i].resize(K);
        _tmpT[i].resize(K);
        _RdnT[i].resize(K);
        _UnT[i].resize(_L,_L);
        _UnT[i].setZero();
        _UndnT[i].resize(K,_L);
        _GsT[i].resize(K,_L);
    }

    _rM.resize(_L, _npatches);
    _vM.resize(_L, _npatches);
}


void OMPSolverEigen::solve(const Eigen::MatrixXf & X) {
    auto start = std::chrono::system_clock::now();

    if (X.cols() != _npatches){
        fprintf(stderr, "Error! Wrong number of patches. %ld != %d", X.cols(), _npatches);
    }
    int M = _npatches;

    int i;
    const int K = _D.cols();
    const float eps = _eps;
    const int L = _L;
    const float lambda=_lambda;
    #define EIGEN_RUNTIME_NO_MALLOC
#pragma omp parallel for private(i)
   for (i = 0; i< M; ++i) {
#ifdef _OPENMP
        int numT=omp_get_thread_num();
#else
        int numT=0;
#endif
        auto Xi = X.col(i);
        float normX = Xi.squaredNorm();

        // Eigen::MatrixXi ind;
        // _rM.refCol(i,ind);
        // ind.set(-1);
        auto ind = _rM.col(i);
        ind.fill(-1);

        auto RUn = _vM.col(i);
        RUn.setZero();

        Eigen::VectorXf& Rdn=_RdnT[numT];
        Rdn << _D.transpose() * Xi;

        Eigen::VectorXf& scores(_scoresT[numT]);
        Eigen::VectorXf& norm(_normT[numT]);
        Eigen::MatrixXf& Un(_UnT[numT]);
        Eigen::MatrixXf& Undn(_UndnT[numT]);
        Eigen::MatrixXf& Gs(_GsT[numT]);
        
        
        if (!((normX <= eps) || L == 0)) {
            scores = Rdn;
            norm.fill(1.0);
            Un.setZero();

            int j;
            for (j = 0; j<L; ++j) {
                int currentInd;
                scores.cwiseAbs().maxCoeff(&currentInd);
                // currentInd = cblas_isamax(K, scores.data(), 1);
                if (norm[currentInd] < 1e-8) {
                    ind[j]=-1;
                    break;
                }
                const float invNorm=(1.0f)/sqrt(norm[currentInd]);
                const float RU=Rdn[currentInd]*invNorm;
                const float delta = RU*RU;
                if (delta < 2*lambda) {
                    break;
                }

                RUn[j]=RU;
                normX -= delta;
                ind[j]=currentInd;

                
                Un(j,j) = -1.0f;
                // cblas_copy<T>(j,prUndn+currentInd,K,prUn+j*L,1);
                Un.col(j) << Undn.row(currentInd).leftCols(j).transpose();
                cblas_strmv(CblasColMajor,CblasUpper,CblasNoTrans,CblasNonUnit,j,Un.data(),L,Un.data()+j*L,1);
                // auto Unj = Un.topRows(j).col(j);
                // Un.topRows(j).col(j) << Un.topLeftCorner(j,j).triangularView<Eigen::Upper>() * Un.topRows(j).col(j);
                // Un.block(0,j,j,1) << (Un.block(0,0,j,j).triangularView<Eigen::Upper>() * Un.col(j));
                // cblas_scal<T>(j+1,-invNorm,prUn+j*L,1);
                Un.topRows(j+1).col(j) *= (-invNorm);

                if (j == L-1 || (normX <= eps)) {
                    ++j;
                    break;
                }
                Gs.col(j) << _G.col(currentInd);
                // Undn.col(j) << Gs.leftCols(j+1) * Un.topRows(j+1).col(j);
                cblas_sgemv(CblasColMajor,CblasNoTrans,K,j+1,1.0f,Gs.data(),K,Un.data()+j*L,1,0.0f,Undn.data()+j*K,1);
                auto Undnj = Undn.col(j);
                // Undn.refCol(j,Undnj);
                Rdn -= (Undnj * RUn[j]);
                norm -= Undnj.cwiseProduct(Undnj);
                // scores.sqr(Rdn);
                scores << Rdn.cwiseProduct(Rdn).cwiseQuotient(norm);
                // scores.div(norm);
                for (int k = 0; k<=j; ++k) scores[ind[k]]=0.0;
            }
            // compute the final coefficients
            // cblas_trmv<T>(CblasColMajor,CblasUpper,CblasNoTrans,CblasNonUnit,
            //         j,prUn,L,prRUn,1);
            cblas_strmv(CblasColMajor,CblasUpper,CblasNoTrans,CblasNonUnit,j,Un.data(),L,RUn.data(),1);
            // RUn.block(0,0,j,1) = Un.block(0,0,j,j).triangularView<Eigen::Upper>() * RUn;
        }
   }

   auto stop = std::chrono::system_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
   printval("Elapsed time", ": ", duration, "ms");
}


void OMPSolverEigen::roundValues(){
    // Eigen::MatrixXf vals;
    // _vM.toVect(vals);
    // for (auto i=0; i<vals.n(); i++) vals[i] = round(vals[i]);
}


void OMPSolverEigen::transform(float lowval, float highval){
    // Usa valores máximos e mínimos para converter em 16bit
    // if (highval < lowval) std::swap(lowval, highval);
    // Eigen::MatrixXf vals;
    // _vM.toVect(vals);
    // float maxval = vals.maxval();
    // _minval = vals.minval();
    // _ptp = maxval - _minval;
    // float diff = highval - lowval;
    // _vM.add(-_minval);
    // _vM.scal(diff/_ptp);
    // _vM.add(lowval);
    _transformed = true;
}


void OMPSolverEigen::transform0(float ptp){
    // Usa valores máximos e mínimos para converter em 16bit
    // Eigen::MatrixXf vals;
    // _vM.toVect(vals);
    // float maxval = vals.maxval();
    // _minval = vals.minval();
    // _ptp = maxval - _minval;
    // _vM.add(-_minval);
    // _vM.scal(ptp/_ptp);
    _transformed = true;
}


void OMPSolverEigen::getResults(Eigen::MatrixXf& spalpha){
    // if (_transformed)
    //     transform(_minval, _minval+_ptp);
    // transform(_minval, _minval+_ptp);
    _transformed = false;
    // spalpha.convert(_vM, _rM, _D.n());
    spalpha.setZero();
    for (int j=0; j<_npatches; j++){
        for (int i=0; i<_L; i++){
            auto idx = _rM(i,j);
            if (idx>=0) spalpha(idx,j) = _vM(i,j);
            else break;
            // printval(idx, " --> ", _vM(i,j), "");
        }
        // printval("","","","");
    }
    // exit(0);
}


OMPSolverEigen::~OMPSolverEigen(){
    delete[](_scoresT);
    delete[](_normT);
    delete[](_tmpT);
    delete[](_RdnT);
    delete[](_UnT);
    delete[](_UndnT);
    delete[](_GsT);
}



class BOMPSolverEigen {
    public:
        BOMPSolverEigen(int num_patches, Eigen::MatrixXf & D, int L, float eps, float lambda, int num_threads=-1);
        void solve(const Eigen::MatrixXf& X);
        void getResults(Eigen::MatrixXf& spalpha);
        
        ~BOMPSolverEigen();
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

        // Auxiliares
        Eigen::MatrixXf _G;

        Eigen::VectorXf* _gilkT;
        Eigen::VectorXf* _wT;
        Eigen::VectorXf* _alphaT;
        Eigen::VectorXf* _alphanT;
        Eigen::VectorXf* _betaT;
        Eigen::VectorXf* _betailT;
        Eigen::MatrixXf* _LowT;
        Eigen::MatrixXf* _GilT;

        // Resultados
        Eigen::MatrixXi _rM;
        Eigen::MatrixXf _vM;
};


BOMPSolverEigen::BOMPSolverEigen(int num_patches, Eigen::MatrixXf & D, int L, float eps, float lambda, int num_threads) :
    _npatches(num_patches), _D(D), _L(L), _eps(eps), _lambda(lambda)
{
    int K = _D.cols();
    _L = std::min(_L, K);
    _G = Eigen::MatrixXf(K, K);
    _G << _D.transpose() * _D;


    _NUM_THREADS = init_omp(num_threads);
    _transformed = false;
    _gilkT = new Eigen::VectorXf[_NUM_THREADS];
    _wT = new Eigen::VectorXf[_NUM_THREADS];
    _alphaT = new Eigen::VectorXf[_NUM_THREADS];
    _alphanT = new Eigen::VectorXf[_NUM_THREADS];
    _betaT = new Eigen::VectorXf[_NUM_THREADS];
    _betailT = new Eigen::VectorXf[_NUM_THREADS];
    _LowT = new Eigen::MatrixXf[_NUM_THREADS];
    _GilT = new Eigen::MatrixXf[_NUM_THREADS];
    for (int i = 0; i<_NUM_THREADS; ++i) {
        _gilkT[i].resize(_L);
        _wT[i].resize(_L);
        _alphaT[i].resize(_L);
        _alphanT[i].resize(K);
        _betaT[i].resize(K);
        _betailT[i].resize(_L);
        _LowT[i].resize(_L,_L);
        _GilT[i].resize(_L,K);
    }

    _rM.resize(_L, _npatches);
    _vM.resize(_L, _npatches);
}


void BOMPSolverEigen::solve(const Eigen::MatrixXf & X) {
    auto start = std::chrono::system_clock::now();

    if (X.cols() != _npatches){
        fprintf(stderr, "Error! Wrong number of patches. %ld != %d", X.cols(), _npatches);
    }
    int M = _npatches;

    int i;
    const int K = _D.cols();
    const float eps = _eps;
    const int L = _L;
    const float lambda=_lambda;
    #define EIGEN_RUNTIME_NO_MALLOC
#pragma omp parallel for private(i)
   for (i = 0; i< M; ++i) {
#ifdef _OPENMP
        int numT=omp_get_thread_num();
#else
        int numT=0;
#endif
        auto Xi = X.col(i);
        float normX = Xi.squaredNorm();

        auto ind = _rM.col(i);
        ind.fill(-1);

        auto c = _vM.col(i);
        c.setZero();

        Eigen::VectorXf& alphan=_alphanT[numT];
        alphan << _D.transpose() * Xi;
        Eigen::VectorXf alpha0(K);
        alpha0 << alphan;
        
        if (!((normX <= eps) || L == 0)) {
            // Eigen::VectorXf& gilk(_gilkT[numT]);
            Eigen::VectorXf& w(_wT[numT]);
            Eigen::VectorXf& alpha(_alphaT[numT]);
            Eigen::VectorXf& beta(_betaT[numT]);
            Eigen::VectorXf& betail(_betailT[numT]);
            Eigen::MatrixXf& Low(_LowT[numT]);
            Eigen::MatrixXf& Gil(_GilT[numT]);
            Low(0,0) = 1.0f;
            float delta=.0f;
            w.setZero();

            int j;
            for (j = 0; j<L && normX>eps; ++j) {
                int k;
                // alphan.cwiseAbs().maxCoeff(&k);
                k = cblas_isamax(K, alphan.data(), 1);
                
                // Se j maior que 0
                w << Gil.topRows(j).col(k);
                // cblas_scopy(j, Gil.data()+k*L, 1, w.data(), 1);
                // cblas_strsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, j, Low.data(), L, w.data(), 1);
                Low.topRows(j).triangularView<Eigen::Lower>().solveInPlace(w);
                Low.row(j) << w.transpose();
                Low(j,j) = sqrtf(1 - (w.topRows(j).squaredNorm()));

                // Atualizar os novos ks para tudo o que for necessário
                ind[j] = k;
                alpha[j] = alpha0[k];
                Gil.row(j) << _G.row(k);

                // Atualizar solução
                c << alpha;
                LAPACKE_spotrs(CblasColMajor, 'L', j+1, 1, Low.data(), L, c.data(), L);

                // printval(c.topRows(j+1),"\n","","");
                // std::cout << c.topRows(j+1) << std::endl << std::endl;

                // Atualizar valores de erros
                // beta << Gil.topRows(j+1).transpose() * c.topRows(j+1);
                cblas_sgemv(CblasColMajor, CblasTrans, j+1, K, 1.0f, Gil.data(), L, c.data(), 1, 0.0f, beta.data(), 1);
                alphan << alpha0 - beta;
                for(int m=0; m<=j;m++) {
                    betail[m]=beta[ind[m]];
                }

                float ndelta = c.topRows(j+1).dot(betail.topRows(j+1));
                normX = normX - ndelta + delta;
                delta = ndelta;
            }
        }
        // exit(0);
   }

   auto stop = std::chrono::system_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
   printval("Elapsed time", ": ", duration, "ms");
}





void BOMPSolverEigen::getResults(Eigen::MatrixXf& spalpha){
    // if (_transformed)
    //     transform(_minval, _minval+_ptp);
    // transform(_minval, _minval+_ptp);
    _transformed = false;
    // spalpha.convert(_vM, _rM, _D.n());
    spalpha.setZero();
    for (int j=0; j<_npatches; j++){
        for (int i=0; i<_L; i++){
            auto idx = _rM(i,j);
            if (idx>=0) spalpha(idx,j) = _vM(i,j);
            else break;
            // printval(idx, " --> ", _vM(i,j), "");
        }
        // printval("","","","");
    }
    // exit(0);
}


BOMPSolverEigen::~BOMPSolverEigen(){
    delete[](_gilkT);
    delete[](_wT);
    delete[](_alphaT);
    delete[](_alphanT);
    delete[](_betaT);
    delete[](_betailT);
    delete[](_LowT);
    delete[](_GilT);
}