/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * Routines for testing the device API of CUBLAS.
 */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

/* Includes, cuda */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* Includes, cuda helper functions */
// #include <helper_cuda.h>

#define id(m, n, ld) (((n) * (ld) + (m)))

__device__ cublasHandle_t *handle;
__global__ void
fill_value(float * vec, int n, float value){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < n){
        vec[i] = value;
        // printf("vec[%d] = %f\n", i, vec[i]);
        i += gridDim.x * blockDim.x;
    }
}
__global__ void
square_vector(float * vec_in, float * vec_out, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < n){
        vec_out[i] = vec_in[i]*vec_in[i];
        i += gridDim.x * blockDim.x;
    }
}
__global__ void
divide_vectors(float * vecn, float * vecd, float * vec_out, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < n){
        vec_out[i] = vecn[i]/vecd[i];
        i += gridDim.x * blockDim.x;
    }
}
__global__ void
fill_indexed(float * vec, int * vecidx, float value, int nidx){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < nidx){
        vec[vecidx[i]] = value;
        i += gridDim.x * blockDim.x;
    }
}
__global__ void
print_vec(float * vec, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < n){
        printf("vec[%d] = %f\n", i, vec[i]);
        i += gridDim.x * blockDim.x;
    }
}
__global__ void
print_vec_int(int * vec, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < n){
        printf("vec[%d] = %d\n", i, vec[i]);
        i += gridDim.x * blockDim.x;
    }
}

__device__ int
isamax(int n, float* v){
    int idx=0;
    for(int i=1; i<n; i++){
        if (fabs(v[i])>fabs(v[idx])){
            idx=i;
        }
    }
    return idx;
}
__device__ float
snrm2sqr(int n, float* v){
    float sum=0.0f;
    for (int i=0;i<n;i++) sum += (v[i]*v[i]);
    return sum;
}
__device__ void
sgemvT(int m, int n, float * mat, int lda, float * vec, float * out_vec){
    float * maux = mat;
    // int i = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i=0; i<n; i++){
        out_vec[i] = 0.0f;
        for(int j=0; j<m; j++, maux++) out_vec[i] += (maux[0]*vec[j]);
    }
}

__device__ void
sgemv(int m, int n, float * mat, int lda, float * vec, float * out_vec){
    float * maux;
    // int i = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i=0; i<m; i++){
    	maux = mat + i;
        out_vec[i] = 0.0f;
        for(int j=0; j<n; j++, maux+=lda) out_vec[i] += (maux[0]*vec[j]);
    }
}
__device__ void
struppermv(int m, int n, float * mat, int lda, float * vec, float * out_vec){
    float * maux;
    // int i = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i=0; i<m; i++){
    	maux = mat + i + i*lda;
        out_vec[i] = 0.0f;
        for(int j=i; j<n; j++, maux+=lda) out_vec[i] += (maux[0]*vec[j]);
    }
}
__global__ void
match_pursuit(int * m_D, int * n_D, 
              float *d_D, float *d_X, float *G,
              float *scoresT, float *normT, float *tmpT,
              float *RdnT, float *UnT, float *UndnT, float *GsT,
              float *vM, int *rM, float * mresults, int *idxm, int *d_calc,
              float *epsilon,
              int * L,
              int * numSamples)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int K = *n_D;
    float *scores = &scoresT[i*(K)];
    float *norm = &normT[i*(K)];
    float *tmp = &tmpT[i*(K)];
    float *Un = &UnT[i*(*L)*(*L)];
    float *Undn = &UndnT[i*(K)*(*L)];
    float *Gs = &GsT[i*(K)*(*L)];
    float eps = *epsilon;

    for(;i < (*numSamples); i+=gridDim.x * blockDim.x)
    {
        if(d_calc[i]) continue;
        float *Xi;
        Xi = &d_X[i*(*m_D)];

        float normX;
		normX = snrm2sqr(*m_D, Xi);

		int * ind = &rM[i*(*L)];
		float * RUn = &vM[i*(*L)];

		float *Rdn = &RdnT[i*(K)];
//		sgemvT(*m_D, *n_D, d_D, *m_D, Xi, Rdn);
		float *mresult = &mresults[i*(*L)*(*L)];

		// COREORMP
		// Copia Rdn ---> scores
		for (int k=0;k<K;k++) scores[k]=Rdn[k];


		// Seta 1.0 em todas as entradas de norm
		for (int k=0;k<K;k++) norm[k]=1.0f;

		int j;
		for (j=0; j<*L && normX > eps; j++){
			int currentInd=0;
			// Índice do maior valor 1,...,K
			currentInd = isamax(K, scores);
			if (norm[currentInd] < 1e-8f){
				ind[j] = -1;
				break;
			}


			float invNorm = -1.0/sqrt(norm[currentInd]);
			float RU = Rdn[currentInd]*invNorm;
			float delta = RU*RU;
			if (delta < eps*eps){
				break;
			}

			RUn[j] = -RU;
			normX -= (RUn[j]*RUn[j]);
			ind[j] = currentInd;

			Un[j*(*L)+j] = -1.0;

			for (int k=0;k<j;k++) tmp[k]=Undn[currentInd+k*K];
			struppermv(j, j, Un, *L, tmp, &Un[(j*(*L))]);

			for (int k=0;k<j+1;k++) Un[j*(*L)+k] *= invNorm;

			if (j == (*L)-1 || (normX <= eps)){
				++j;
				break;
			}

			//for (int k=0;k<K;k++) Gs[j*(K)+k]=G[currentInd*K+k];
			float *d=&Gs[j*K], *s=&G[currentInd*K];
			for (int k=0; k<K; k++, d++, s++) *d = *s;
			// memcpy((void*)&Gs[j*K], (void*)&G[currentInd*K], K*sizeof(Gs[0]));

			sgemv(K, j+1, Gs, K, &Un[(j*(*L))], &Undn[j*K]);

			float *Undnj = &Undn[j*K];
			for (int k=0;k<K;k++) Rdn[k] -= (RUn[j]*Undnj[k]);
			for (int k=0;k<K;k++) tmp[k] = Undnj[k]*Undnj[k];
			for (int k=0;k<K;k++) norm[k] -= tmp[k];
			for (int k=0;k<K;k++) scores[k] = (Rdn[k]*Rdn[k])/norm[k];
			for (int k=0; k<=j;++k) scores[ind[k]] = 0.0;
			struppermv(j+1, j+1, Un, *L, RUn, mresult);
            mresult += (*L);
		}
		idxm[i] = j-1;
		struppermv(j, j, Un, *L, RUn, mresult);

        if(j<(*L)-1) ind[j] = -1;
    }
}

__global__ void
match_pursuit_streamed(int n,int * m_D, int * n_D,
              float *d_D, float *Xi, float *G,
              float *scores, float *norm, float *tmp,
              float *Rdn, float *Un, float *Undn, float *Gs,
              float *RUn, int *ind,
              float *epsilon,
              int * L)
{
    int K = *n_D;
    float eps = *epsilon;

	float normX = snrm2sqr(*m_D, Xi);

	// COREORMP
	// Copia Rdn ---> scores
	for (int k=0;k<K;k++) scores[k]=Rdn[k];
//	cublasScopy(hand, K, Rdn, 1, scores, 1);


	// Seta 1.0 em todas as entradas de norm
	for (int k=0;k<K;k++) norm[k]=1.0f;

	int j;
	for (j=0; j<*L && normX > eps; j++){
		int currentInd=0;
		// Índice do maior valor 1,...,K
		currentInd = isamax(K, scores);


		if (norm[currentInd] < 1e-8){
			ind[j] = -1;
			break;
		}


		float invNorm = -1.0/sqrt(norm[currentInd]);
		float RU = Rdn[currentInd]*invNorm;
		float delta = RU*RU;
		if (delta < eps*eps){
			break;
		}

		RUn[j] = -RU;
		// normX -= delta;
		normX -= (RUn[j]*RUn[j]);
		ind[j] = currentInd;

		Un[j*(*L)+j] = -1.0;
		for (int k=0;k<j;k++) tmp[k]=Undn[currentInd+k*K];
		struppermv(j, j, Un, *L, tmp, &Un[(j*(*L))]);
		// cublasSgemv(hand, CUBLAS_OP_N, j, j, )


		for (int k=0;k<j+1;k++) Un[j*(*L)+k] *= invNorm;

		if (j == (*L)-1 || (normX <= eps)){
			++j;
			break;
		}

		for (int k=0;k<K;k++) Gs[j*(K)+k]=G[currentInd*K+k];

		sgemv(K, j+1, Gs, K, &Un[(j*(*L))], &Undn[j*K]);
		float *Undnj = &Undn[j*K];
		for (int k=0;k<K;k++) Rdn[k] -= (RUn[j]*Undnj[k]);
		for (int k=0;k<K;k++) tmp[k] = Undnj[k]*Undnj[k];
		for (int k=0;k<K;k++) norm[k] -= tmp[k];
		for (int k=0;k<K;k++) scores[k] = (Rdn[k]*Rdn[k])/norm[k];
		for (int k=0; k<=j;++k) scores[ind[k]] = 0.0;
	}

	for (int k=0;k<j;k++) tmp[k]=RUn[k];
	struppermv(j, j, Un, *L, tmp, RUn);
}

struct MPParams
{
    int m, n;
    float epsilon;
    int L, numSamples;
};


#define threadsPerBlock (128)
#define blocksPerGrid (8)
#define parops (threadsPerBlock * blocksPerGrid)
clock_t start;
cublasStatus_t status;
cublasHandle_t hand;
float dtime;
float *d_D=0, *d_DD=0, *d_X=0;
int *d_calc=0;
int MN;
float alpha=1.0f, beta=0.0f;
int M;
int K;
int L;
float *scoresT=0, *normT=0, *tmpT=0, *RdnT=0, *UnT=0, *UndnT=0, *GsT=0;
float *G=0;
MPParams * d_params=0;
float *vM=0;
int *rM=0;
float *mresults=0;
int *idxm=0;


extern "C" void matching_pursuit_init(int m_D, int n_D, float * h_D,
                                 int n_X, 
                                 float epsilon,
                                 int _L,
                                 cudaError_t * error){

	MN = m_D*n_D;


	// --- Iniciando ----
	M = n_X;
	K = n_D;
	L = min(_L, K);

	status = cublasCreate(&hand);

	// ----------------------------------------------------------------------------------
	// Cria variáveis auxiliares
	*error = cudaMalloc((void **)&scoresT, parops*K*sizeof(float));
	if (*error != cudaSuccess)
	{
		fprintf(stderr, "! device memory allocation *error %s\n", cudaGetErrorString(*error));
		return;
	}
	*error = cudaMalloc((void **)&normT, parops*K*sizeof(float));
	if (*error != cudaSuccess)
	{
		fprintf(stderr, "! device memory allocation *error %s\n", cudaGetErrorString(*error));
		return;
	}
	*error = cudaMalloc((void **)&tmpT, parops*K*sizeof(float));
	if (*error != cudaSuccess)
	{
		fprintf(stderr, "! device memory allocation *error %s\n", cudaGetErrorString(*error));
		return;
	}
	*error = cudaMalloc((void **)&RdnT, M*K*sizeof(float));
	if (*error != cudaSuccess)
	{
		fprintf(stderr, "! device memory allocation *error %s\n", cudaGetErrorString(*error));
		return;
	}
	*error = cudaMalloc((void **)&UnT, parops*L*L*sizeof(float));
	if (*error != cudaSuccess)
	{
		fprintf(stderr, "! device memory allocation *error %s\n", cudaGetErrorString(*error));
		return;
	}
	*error = cudaMalloc((void **)&UndnT, parops*K*L*sizeof(float));
	if (*error != cudaSuccess)
	{
		fprintf(stderr, "! device memory allocation *error %s\n", cudaGetErrorString(*error));
		return;
	}
	*error = cudaMalloc((void **)&GsT, parops*K*L*sizeof(float));
	if (*error != cudaSuccess)
	{
		fprintf(stderr, "! device memory allocation *error %s\n", cudaGetErrorString(*error));
		return;
	}

	// ----------------------------------------------------------------------------------
	/* Host to device data transfer: dictionary */
	*error = cudaMalloc((void **)&d_D, MN*sizeof(d_D[0]));
	if (*error != cudaSuccess)
	{
		fprintf(stderr, "! device memory allocation *error %s\n", cudaGetErrorString(*error));
		return;
	}
	status = cublasSetVector(MN, sizeof(h_D[0]), h_D, 1, d_D, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "! device access *error (write dictionary)\n");
		return;
	}
	/* Host to device data transfer: signal */
	*error = cudaMalloc((void **)&d_X, m_D * n_X * sizeof(d_X[0]));
	if (*error != cudaSuccess)
	{
		fprintf(stderr, "! device memory allocation *error (signal)\n");
		return;
	}
    *error = cudaMalloc((void **)&d_calc, n_X * sizeof(d_calc[0]));
	if (*error != cudaSuccess)
	{
		fprintf(stderr, "! device memory allocation *error (calculate)\n");
		return;
	}


	// ----------------------------------------------------------------------------------
	// --- Dt*D ----
	*error = cudaMalloc((void **)&G, n_D * n_D * sizeof(G[0]));
	if (*error != cudaSuccess)
	{
		fprintf(stderr, "! device memory allocation *error (projected vector)\n");
		return;
	}
	cublasSgemm(hand, CUBLAS_OP_T, CUBLAS_OP_N, n_D, n_D, m_D, &alpha, d_D, m_D, d_D, m_D, &beta, G, n_D);

	// ----------------------------------------------------------------------------------
	// --- Resultados ---
	*error = cudaMalloc((void **)&vM, L * M * sizeof(vM[0]));
	if (*error != cudaSuccess)
	{
		fprintf(stderr, "! device memory allocation *error (result values)\n");
		return;
	}

	*error = cudaMalloc((void **)&rM, L * M * sizeof(rM[0]));
	if (*error != cudaSuccess)
	{
		fprintf(stderr, "! device memory allocation *error (result idxs)\n");
		return;
	}

	*error = cudaMalloc((void **)&mresults, L * L * M * sizeof(mresults[0]));
	if (*error != cudaSuccess)
	{
		fprintf(stderr, "! device memory allocation *error (result values)\n");
		return;
	}

	*error = cudaMalloc((void **)&idxm, M * sizeof(idxm[0]));
	if (*error != cudaSuccess)
	{
		fprintf(stderr, "! device memory allocation *error (result idxm)\n");
		return;
	}

	// ----------------------------------------------------------------------------------
	// --- Parâmetros ----
	MPParams h_params = {m_D,n_D,epsilon, L, n_X};
	*error = cudaMalloc((void **)&d_params, sizeof(MPParams));
	if (*error != cudaSuccess)
	{
		fprintf(stderr, "! device memory allocation *error (params)\n");
		return;
	}
	*error = cudaMemcpy((void*)d_params, (void*)&h_params, sizeof(MPParams), cudaMemcpyHostToDevice);
	if (*error != cudaSuccess)
	{
		fprintf(stderr, "! device memory allocation *error (params)\n");
		return;
	}
	// ----------------------------------------------------------------------------------
	cudaDeviceSynchronize();

	dtime = ((float)clock() - start) / CLOCKS_PER_SEC;
	printf("\nTime for Host to Device data transfer: %f (s)\n", dtime);
}


extern "C" void matching_pursuit(int m_D, int n_D,
                                 int n_X, const float * h_X, const int * h_calc,
                                 cudaError_t * error){

    start = clock();

    status = cublasSetVector(m_D * n_X, sizeof(h_X[0]), h_X, 1, d_X, 1);
    status = cublasSetVector(n_X, sizeof(h_calc[0]), h_calc, 1, d_calc, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "! device access *error (write signal)\n");
		return;
	}
    cublasSgemm(hand, CUBLAS_OP_T, CUBLAS_OP_N, n_D, M, m_D, &alpha, d_D, m_D, d_X, m_D, &beta, RdnT, n_D);
    cudaMemset((void*)vM, 0, L * M * sizeof(vM[0]));
    // cudaMemset((void*)rM, -1, L * M * sizeof(rM[0]));

    match_pursuit<<<blocksPerGrid, threadsPerBlock>>>(&d_params->m, &d_params->n,
                                                      d_D, d_X, G,
                                                      scoresT, normT, tmpT, RdnT, UnT, UndnT, GsT,
                                                      vM, rM, mresults, idxm, d_calc,
                                                      &d_params->epsilon,
                                                      &d_params->L, &d_params->numSamples);

    cudaDeviceSynchronize();
    *error = cudaGetLastError();
	if (*error != cudaSuccess){
		fprintf(stderr, "Failed to launch match_pursuit kernel (error code %s)!\n", cudaGetErrorString(*error));
		return;
	}
    dtime = ((float)clock() - start) / CLOCKS_PER_SEC;
    printf("Time for decoding: %f (s)\n", dtime);
}

extern "C" void matching_pursuit_get_results(const int * rms, const int * idms, const float *vals){
                                    
	start = clock();

	status = cublasGetVector(L*M, sizeof(rms[0]), (const void *)rM, 1, (void*)rms, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "! Get Vector Error, code: %d !\n", status);
		return;
	}
    status = cublasGetVector(M, sizeof(idms[0]), (const void *)idxm, 1, (void*)idms, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "! Get Vector Error, code: %d !\n", status);
		return;
	}
    status = cublasGetVector(M*L*L, sizeof(vals[0]), (const void *)mresults, 1, (void*)vals, 1);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "! Get Vector Error, code: %d !\n", status);
		return;
	}

	dtime = ((float)clock() - start) / CLOCKS_PER_SEC;
	printf("Time for getting results: %f (s)\n", dtime);
}


extern "C" void matching_pursuit_destroy(cudaError_t * error){
	cublasDestroy(hand);
	*error = cudaFree(scoresT);
	*error = cudaFree(normT);
	*error = cudaFree(tmpT);
	*error = cudaFree(RdnT);
	*error = cudaFree(UnT);
	*error = cudaFree(UndnT);
	*error = cudaFree(GsT);
	*error = cudaFree(d_D);
	*error = cudaFree(G);
	*error = cudaFree(d_X);
	*error = cudaFree(vM);
	*error = cudaFree(rM);
	*error = cudaFree(d_params);
}




// -------------------------------------------------------------------------------
// Implementação usando CUBLAS no host

// extern "C" void matching_pursuit2(int m_D, int n_D, float *h_D,
//                                   int n_X, float *h_X,
//                                   float *h_CC,
//                                   float epsilon,
//                                   int T,
//                                   int L,
//                                   cudaError_t *error)
// {
//     clock_t start;
//     cublasStatus_t status;
//     float dtime;
//     float *d_D = 0, *d_X = 0;
//     int MN = m_D * n_D;

//     // int threadsPerBlock = 32;
//     // int blocksPerGrid = 8;
//     // int parops = threadsPerBlock * blocksPerGrid;
//     cublasHandle_t hand;
//     status = cublasCreate(&hand);
//     // cudaDeviceSynchronize();

//     // --- Iniciando ----
//     int M = n_X;
//     int K = n_D;
//     L = min(L, K);
//     start = clock();

//     // ----------------------------------------------------------------------------------
//     // Cria variáveis auxiliares
//     // printf("OK!\n");
//     float *scoresT = 0, *DtX=0;
//     *error = cudaMalloc((void **)&scoresT, M * K * sizeof(float));
//     if (*error != cudaSuccess)
//     {
//         fprintf(stderr, "! device memory allocation *error %s\n", cudaGetErrorString(*error));
//         return;
//     }

//     *error = cudaMalloc((void **)&DtX, M * K * sizeof(float));
// 	if (*error != cudaSuccess)
// 	{
// 		fprintf(stderr, "! device memory allocation *error %s\n", cudaGetErrorString(*error));
// 		return;
// 	}


//     // ----------------------------------------------------------------------------------
//     /* Host to device data transfer: dictionary */
//     *error = cudaMalloc((void **)&d_D, MN * sizeof(d_D[0]));
//     if (*error != cudaSuccess)
//     {
//         fprintf(stderr, "! device memory allocation *error %s\n", cudaGetErrorString(*error));
//         return;
//     }
//     status = cublasSetVector(MN, sizeof(h_D[0]), h_D, 1, d_D, 1);
//     if (status != CUBLAS_STATUS_SUCCESS)
//     {
//         fprintf(stderr, "! device access *error (write dictionary)\n");
//         return;
//     }

//     /* Host to device data transfer: signal */
//     *error = cudaMalloc((void **)&d_X, m_D * n_X * sizeof(d_X[0]));
//     if (*error != cudaSuccess)
//     {
//         fprintf(stderr, "! device memory allocation *error (signal)\n");
//         return;
//     }
//     status = cublasSetVector(m_D * n_X, sizeof(h_X[0]), h_X, 1, d_X, 1);
//     if (status != CUBLAS_STATUS_SUCCESS)
//     {
//         fprintf(stderr, "! device access *error (write signal)\n");
//         return;
//     }
//     // ----------------------------------------------------------------------------------
//     // --- Dt*D ----
//     float *G = 0;
//     *error = cudaMalloc((void **)&G, n_D * n_D * sizeof(G[0]));
//     if (*error != cudaSuccess)
//     {
//         fprintf(stderr, "! device memory allocation *error (projected vector)\n");
//         return;
//     }
//     float alpha=1.0, beta=.0;
//     cublasSgemm(hand, CUBLAS_OP_T, CUBLAS_OP_N, K, K, m_D, &alpha, d_D, m_D, d_D, m_D, &beta, G, K);

//     // ----------------------------------------------------------------------------------
//     // --- Resultados ---
//     float *vM = 0;
//     *error = cudaMalloc((void **)&vM, K * M * sizeof(vM[0]));
//     if (*error != cudaSuccess)
//     {
//         fprintf(stderr, "! device memory allocation *error (projected vector)\n");
//         return;
//     }
// //    cudaMemset((void *)vM, 0, J * M * sizeof(vM[0]));

//     // ----------------------------------------------------------------------------------

//     dtime = ((float)clock() - start) / CLOCKS_PER_SEC;
//     printf("\nTime for Host to Device data transfer: %f (s)\n", dtime);
//     start = clock();

//     // Iterative Hard T...

//     const int niters = 100;
//     const float step = 0.4f;
//     const float mstep = -step;

//     cublasSgemm(hand, CUBLAS_OP_T, CUBLAS_OP_N, K, M, m_D, &step, d_D, m_D, d_X, m_D, &beta, DtX, K);

//     for(int j=0; j<niters; j++){
//     	cudaMemset((void *)vM, 0, K * M * sizeof(vM[0]));

//     	// Zera valores

//     	if(j == niters-1) break;

//     	cublasSgemm(hand, CUBLAS_OP_N, CUBLAS_OP_N, K, M, K, &mstep, G, K, vM, K, &beta, scoresT, K);
//     	cublasSaxpy(hand, K*M, &alpha, DtX, 1, vM, 1);
//     	cublasSaxpy(hand, K*M, &alpha, vM, 1, scoresT, 1);
//     }



//     *error = cudaGetLastError();
//     if (*error != cudaSuccess)
//     {
//         fprintf(stderr, "Failed to launch match_pursuit kernel (error code %s)!\n", cudaGetErrorString(*error));
//         return;
//     }

//     cudaDeviceSynchronize();
//     dtime = ((float)clock() - start) / CLOCKS_PER_SEC;
//     printf("Time for decoding: %f (s)\n", dtime);

//     start = clock();

//     status = cublasGetVector(K * M, sizeof(h_CC[0]), (const void *)vM, 1, h_CC, 1);

//     dtime = ((float)clock() - start) / CLOCKS_PER_SEC;
//     printf("Time for getting results: %f (s)\n", dtime);

//     cublasDestroy(hand);
//     *error = cudaFree(scoresT);
//     *error = cudaFree(d_D);
//     *error = cudaFree(G);
//     *error = cudaFree(d_X);
//     *error = cudaFree(vM);
// }
