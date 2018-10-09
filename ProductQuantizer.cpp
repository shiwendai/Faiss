/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "ProductQuantizer.h"


#include <cstddef>
#include <cstring>
#include <cstdio>

#include <algorithm>

#include "FaissAssert.h"
#include "VectorTransform.h"
#include "IndexFlat.h"
#include "utils.h"


extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_ (const char *transa, const char *transb, FINTEGER *m, FINTEGER *
            n, FINTEGER *k, const float *alpha, const float *a,
            FINTEGER *lda, const float *b, FINTEGER *
            ldb, float *beta, float *c, FINTEGER *ldc);

}


namespace faiss {



template <typename CT, class C>
void pq_estimators_from_tables_Mmul4 (int M, const CT * codes,
                                      size_t ncodes,
                                      const float * __restrict dis_table,
                                      size_t ksub,
                                      size_t k,
                                      float * heap_dis,
                                      long * heap_ids)
{

    for (size_t j = 0; j < ncodes; j++) {
        float dis = 0;
        const float *dt = dis_table;

        for (size_t m = 0; m < M; m+=4) {
            float dism = 0;
            dism  = dt[*codes++]; dt += ksub;
            dism += dt[*codes++]; dt += ksub;
            dism += dt[*codes++]; dt += ksub;
            dism += dt[*codes++]; dt += ksub;
            dis += dism;
        }

        if (C::cmp (heap_dis[0], dis)) {
            heap_pop<C> (k, heap_dis, heap_ids);
            heap_push<C> (k, heap_dis, heap_ids, dis, j);
        }
    }
}


/* compute an estimator using look-up tables for typical values of M */
/* 
  codes  : 数据库码本
  ncodes : 数据库大小
  dis_table : 子向量到每个质心的距离
  ksub ：子空间质心个数
  k ：堆保存前topk个查询结果
*/
template <typename CT, class C>
void pq_estimators_from_tables_M4 (const CT * codes,
                                   size_t ncodes,
                                   const float * __restrict dis_table,
                                   size_t ksub,
                                   size_t k,
                                   float * heap_dis,
                                   long * heap_ids)
{
    /// 计算查询向量到数据库向量的距离
    for (size_t j = 0; j < ncodes; j++) {
        float dis = 0;
        const float *dt = dis_table;
        /// 计算查询向量到每个向量的距离
        dis  = dt[*codes++]; dt += ksub;
        dis += dt[*codes++]; dt += ksub;
        dis += dt[*codes++]; dt += ksub;
        dis += dt[*codes++];
        ///并把满足条件的值存入堆中
        if (C::cmp (heap_dis[0], dis)) {
            heap_pop<C> (k, heap_dis, heap_ids);
            heap_push<C> (k, heap_dis, heap_ids, dis, j);
        }
    }
}


template <typename CT, class C>
static inline void pq_estimators_from_tables (const ProductQuantizer * pq,
                                              const CT * codes,
                                              size_t ncodes,
                                              const float * dis_table,
                                              size_t k,
                                              float * heap_dis,
                                              long * heap_ids)
{
    /// 计算查询向量到数据库向量的距离
    if (pq->M == 4)  {

        pq_estimators_from_tables_M4<CT, C> (codes, ncodes,
                                         dis_table, pq->ksub, k,
                                         heap_dis, heap_ids);
        return;
    }

    /// 计算查询向量到数据库向量的距离
    if (pq->M % 4 == 0) {
        pq_estimators_from_tables_Mmul4<CT, C> (pq->M, codes, ncodes,
                                            dis_table, pq->ksub, k,
                                            heap_dis, heap_ids);
        return;
    }

    /// 计算查询向量到数据库向量的距离
    /* Default is relatively slow */
    const size_t M = pq->M;
    const size_t ksub = pq->ksub;
    for (size_t j = 0; j < ncodes; j++) {
        float dis = 0;
        const float * __restrict dt = dis_table;
        for (int m = 0; m < M; m++) {
            dis += dt[*codes++];
            dt += ksub;
        }
        if (C::cmp (heap_dis[0], dis)) {
            heap_pop<C> (k, heap_dis, heap_ids);
            heap_push<C> (k, heap_dis, heap_ids, dis, j);
        }
    }
}


/*********************************************
 * PQ implementation
 *********************************************/



ProductQuantizer::ProductQuantizer (size_t d, size_t M, size_t nbits):
    d(d), M(M), nbits(nbits), assign_index(nullptr)
{
    set_derived_values ();
}

ProductQuantizer::ProductQuantizer ():
    d(0), M(1), nbits(0), assign_index(nullptr)
{ 
    set_derived_values ();  
}


//根据d, M and nbits计算dsub byte_per_idx code_size ksub的值
void ProductQuantizer::set_derived_values () {
    // quite a few derived values
    FAISS_THROW_IF_NOT (d % M == 0);
    dsub = d / M;
    byte_per_idx = (nbits + 7) / 8;
    code_size = byte_per_idx * M;
    ksub = 1 << nbits;
    centroids.resize (d * ksub);
    verbose = false;
    train_type = Train_default;
}

/// 质心拷贝
void ProductQuantizer::set_params (const float * centroids_, int m)
{
  memcpy (get_centroids(m, 0), centroids_,
            ksub * dsub * sizeof (centroids_[0]));
}


/// 初始化质心
static void init_hypercube (int d, int nbits,
                            int n, const float * x,
                            float *centroids)
{
    /// 把数据集中每个维度的值求和存入mean[j]
    std::vector<float> mean (d);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            mean [j] += x[i * d + j];

    /// 对数据集中每个维度数据求平均值，并把各维度平均值的最大值存入maxm中
    float maxm = 0;
    for (int j = 0; j < d; j++) {
        mean [j] /= n;
        if (fabs(mean[j]) > maxm) maxm = fabs(mean[j]);
    }

    /// 对每个质心的每个维度进行初始化
    for (int i = 0; i < (1 << nbits); i++) {
        float * cent = centroids + i * d;
        for (int j = 0; j < nbits; j++)
            cent[j] = mean [j] + (((i >> j) & 1) ? 1 : -1) * maxm;
        for (int j = nbits; j < d; j++)
            cent[j] = mean [j];
    }


}

/// 用hypercube_pca方式初始化质心
static void init_hypercube_pca (int d, int nbits,
                                int n, const float * x,
                                float *centroids)
{
    /// 对数据集进行pca降维
    PCAMatrix pca (d, nbits);
    pca.train (n, x);

    /// 初始化质心
    for (int i = 0; i < (1 << nbits); i++) {
        float * cent = centroids + i * d;
        for (int j = 0; j < d; j++) {
            cent[j] = pca.mean[j];
            float f = 1.0;
            for (int k = 0; k < nbits; k++)
                cent[j] += f *
                    sqrt (pca.eigenvalues [k]) *
                    (((i >> k) & 1) ? 1 : -1) *
                    pca.PCAMat [j + k * d];
        }
    }

}

/// PQ算法训练,其中 x 为训练集数据，n 为训练集大小
void ProductQuantizer::train (int n, const float * x)
{
    /// 各个子空间不共享质心(即：需要对每个子空间进行训练)
    if (train_type != Train_shared) {
        train_type_t final_train_type;
        final_train_type = train_type;
        if (train_type == Train_hypercube ||
            train_type == Train_hypercube_pca) {
            if (dsub < nbits) {
                final_train_type = Train_default;
                printf ("cannot train hypercube: nbits=%ld > log2(d=%ld)\n",
                        nbits, dsub);
            }
        }

        /// xslice用来存储训练集的第 m 段的数据集
        float * xslice = new float[n * dsub];
        ScopeDeleter<float> del (xslice);
        /// 用训练集训练 M 个子空间（即：获取每个子空间的质心）
        for (int m = 0; m < M; m++) {
            /// 把训练集的第 m 个段存入xslice
            for (int j = 0; j < n; j++)
                memcpy (xslice + j * dsub,
                        x + j * d + m * dsub,
                        dsub * sizeof(float));

            /// 创建一个聚类对象
            Clustering clus (dsub, ksub, cp);

            // we have some initialization for the centroids
            // 初始化聚类质心
            if (final_train_type != Train_default) {
                clus.centroids.resize (dsub * ksub);
            }

            switch (final_train_type) {
            /// 使用hypercube方式初始化质心
            case Train_hypercube:
                init_hypercube (dsub, nbits, n, xslice,
                                clus.centroids.data ());
                break;
            /// 使用Train_hypercube_pca方式初始化质心
            case  Train_hypercube_pca:
                init_hypercube_pca (dsub, nbits, n, xslice,
                                    clus.centroids.data ());
                break;
            /// 使用已训练的好的质心初始化质心
            case  Train_hot_start:
                memcpy (clus.centroids.data(),
                        get_centroids (m, 0),
                        dsub * ksub * sizeof (float));
                break;
            default: ;
            }

            if(verbose) {
                clus.verbose = true;
                printf ("Training PQ slice %d/%zd\n", m, M);
            }
            IndexFlatL2 index (dsub);
            /// 训练质心
            clus.train (n, xslice, assign_index ? *assign_index : index);
            /// 把训练好的质心拷贝到ProductQuantizer的centroids中，然后训练完成
            set_params (clus.centroids.data(), m);
        }

    /// 如果train_type == Train_shared，表示用数据集中所有子向量训练 k 个公共的质心
    } else {

        Clustering clus (dsub, ksub, cp);

        if(verbose) {
            clus.verbose = true;
            printf ("Training all PQ slices at once\n");
        }

        IndexFlatL2 index (dsub);

        // 用数据集中所有子向量训练 k 个公共的质心
        clus.train (n * M, x, assign_index ? *assign_index : index);
        // 所有每个子空间质心是一样的
        for (int m = 0; m < M; m++) {
            set_params (clus.centroids.data(), m);
        }

    }
}


/// 对一个向量进行product quantizer 编码 即：编码过程 并把编码结果存入code中
void ProductQuantizer::compute_code (const float * x, uint8_t * code)  const
{
    /// 创建存放一个向量 x 到 ksub 个质心中的距离
    float distances [ksub];
    for (size_t m = 0; m < M; m++) {
        /// 用来保存向量 x 到 ksub 个质心中的距离的最短距离
        float mindis = 1e20;
        /// 用来保存向量 x 最近质心的索引
        int idxm = -1;

        /// 获取子向量 xsub
        const float * xsub = x + m * dsub;

        /// 计算子向量到 xsub 到对应子空间质心的距离，并把各个距离保存到distance中
        fvec_L2sqr_ny (distances, xsub, get_centroids(m, 0), dsub, ksub);

        /* Find best centroid */
        /// 从distance中获取最短距离和质心编号
        size_t i;
        for (i = 0; i < ksub; i++) {
            float dis = distances [i];
            if (dis < mindis) {
                mindis = dis;
                idxm = i;
            }
        }

        /// 保存向量编码结果到code中
        switch (byte_per_idx) {
          case 1:  code[m] = (uint8_t) idxm;  break;
          case 2:  ((uint16_t *) code)[m] = (uint16_t) idxm;  break;
        }
    }

}

/// 对单个pq码进行解码 即：根据pq码获取 向量
void ProductQuantizer::decode (const uint8_t *code, float *x) const
{
    if (byte_per_idx == 1) {
        for (size_t m = 0; m < M; m++) {
            memcpy (x + m * dsub, get_centroids(m, code[m]),
                    sizeof(float) * dsub);
        }
    } else {
        const uint16_t *c = (const uint16_t*) code;
        for (size_t m = 0; m < M; m++) {
            memcpy (x + m * dsub, get_centroids(m, c[m]),
                    sizeof(float) * dsub);
        }
    }
}

/// 对多个pq码进行解码
void ProductQuantizer::decode (const uint8_t *code, float *x, size_t n) const
{
    for (size_t i = 0; i < n; i++) {
        this->decode (code + code_size * i, x + d * i);
    }
}

// 根据距离池得到pq码
void ProductQuantizer::compute_code_from_distance_table (const float *tab,
                                                         uint8_t *code) const
{
    for (size_t m = 0; m < M; m++) {
        float mindis = 1e20;
        int idxm = -1;

        /* Find best centroid */
        for (size_t j = 0; j < ksub; j++) {
            float dis = *tab++;
            if (dis < mindis) {
                mindis = dis;
                idxm = j;
            }
        }
        switch (byte_per_idx) {
        case 1:  code[m] = (uint8_t) idxm;  break;
        case 2:  ((uint16_t *) code)[m] = (uint16_t) idxm;  break;
        }
    }
}

/// 对多个向量进行product quantizer 编码 
void ProductQuantizer::compute_codes (const float * x,
                                      uint8_t * codes,
                                      size_t n)  const
{
    /// 如果向量小于16维则直接计算pq码
    if (dsub < 16) { // simple direct computation

#pragma omp parallel for
        for (size_t i = 0; i < n; i++)
            compute_code (x + i * d, codes + i * code_size);

    /// 否则间接计算pq码,先计算距离池，然后再根据距离池计算pq码
    } else { // worthwile to use BLAS
        /// 创建一个大小为 n * ksub * M 的距离池
        float *dis_tables = new float [n * ksub * M];
        ScopeDeleter<float> del (dis_tables);
        /// 计算距离池
        compute_distance_tables (n, x, dis_tables);

        /// 根据距离池计算pq码
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            uint8_t * code = codes + i * code_size;
            const float * tab = dis_tables + i * ksub * M;
            compute_code_from_distance_table (tab, code);
        }
    }
}


/// 用 L2 距离计算各子向量到对应子空间的质心的距离
void ProductQuantizer::compute_distance_table (const float * x,
                                               float * dis_table) const
{
    size_t m;
    //计算各子向量到对应子空间的质心的距离
    for (m = 0; m < M; m++) {
        /// 用 L2 距离计算一个向量到ksub个质心的距离，并把格局里保存到dis_table中
        fvec_L2sqr_ny (dis_table + m * ksub,
                       x + m * dsub,
                       get_centroids(m, 0),
                       dsub,
                       ksub);
    }
}

/// 用 內积 距离计算各子向量到对应子空间的质心的距离
void ProductQuantizer::compute_inner_prod_table (const float * x,
                                                 float * dis_table) const
{
    size_t m;

    for (m = 0; m < M; m++) {
        fvec_inner_products_ny (dis_table + m * ksub,
                                x + m * dsub,
                                get_centroids(m, 0),
                                dsub,
                                ksub);
    }
}


/// 用 L2 距离计算数据集x各个子向量到对应子空间质心的距离
void ProductQuantizer::compute_distance_tables (
           size_t nx,
           const float * x,
           float * dis_tables) const
{

    if (dsub < 16) {

#pragma omp parallel for
        for (size_t i = 0; i < nx; i++) {
            compute_distance_table (x + i * d, dis_tables + i * ksub * M);
        }

    } else { // use BLAS

        for (int m = 0; m < M; m++) {
            pairwise_L2sqr (dsub,
                            nx, x + dsub * m,
                            ksub, centroids.data() + m * dsub * ksub,
                            dis_tables + ksub * m,
                            d, dsub, ksub * M);
        }
    }
}

/// 用 內积 距离计算数据集x各个子向量到对应子空间质心的距离
void ProductQuantizer::compute_inner_prod_tables (
           size_t nx,
           const float * x,
           float * dis_tables) const
{

    if (dsub < 16) {

#pragma omp parallel for
        for (size_t i = 0; i < nx; i++) {
            compute_inner_prod_table (x + i * d, dis_tables + i * ksub * M);
        }

    } else { // use BLAS

        // compute distance tables
        for (int m = 0; m < M; m++) {
            FINTEGER ldc = ksub * M, nxi = nx, ksubi = ksub,
                dsubi = dsub, di = d;
            float one = 1.0, zero = 0;

            sgemm_ ("Transposed", "Not transposed",
                    &ksubi, &nxi, &dsubi,
                    &one, &centroids [m * dsub * ksub], &dsubi,
                    x + dsub * m, &di,
                    &zero, dis_tables + ksub * m, &ldc);
        }

    }
}

template <typename CT, class C>
static void pq_knn_search_with_tables (
      const ProductQuantizer * pq,
      const float *dis_tables,
      const uint8_t * codes,
      const size_t ncodes,
      HeapArray<C> * res,
      bool init_finalize_heap)
{
    size_t k = res->k, nx = res->nh;
    size_t ksub = pq->ksub, M = pq->M;


#pragma omp parallel for
    for (size_t i = 0; i < nx; i++) {
        /* query preparation for asymmetric search: compute look-up tables */
        /* 非对称搜索的查询准备：计算查找表 */
        const float* dis_table = dis_tables + i * ksub * M;

        /* Compute distances and keep smallest values */
        /* 计算距离并保持最小值 */
        long * __restrict heap_ids = res->ids + i * k;
        float * __restrict heap_dis = res->val + i * k;

        if (init_finalize_heap) {
            heap_heapify<C> (k, heap_dis, heap_ids);
        }

        pq_estimators_from_tables<CT, C> (pq,
                                          (CT*)codes, ncodes,
                                          dis_table,
                                          k, heap_dis, heap_ids);
        if (init_finalize_heap) {
            heap_reorder<C> (k, heap_dis, heap_ids);
        }
    }
}

    /*
static inline void pq_estimators_from_tables (const ProductQuantizer * pq,
                                              const CT * codes,
                                              size_t ncodes,
                                              const float * dis_table,
                                              size_t k,
                                              float * heap_dis,
                                              long * heap_ids)
    */
/** perform a search (L2 distance)
 * @param x        查询向量, size nx * d
 * @param nx       查询向量集大小
 * @param codes    数据库码本, size ncodes * byte_per_idx
 * @param ncodes   数据库码本大小
 * @param res      存储结果的堆数组 (nh == nx)
 * @param init_finalize_heap  initialize heap (input) and sort (output)?
 */
void ProductQuantizer::search (const float * __restrict x,
                               size_t nx,
                               const uint8_t * codes,
                               const size_t ncodes,
                               float_maxheap_array_t * res,
                               bool init_finalize_heap) const
{
    FAISS_THROW_IF_NOT (nx == res->nh);
    ///计算查询向量集到质心的距离 
    float * dis_tables = new float [nx * ksub * M];
    ScopeDeleter<float> del(dis_tables);
    compute_distance_tables (nx, x, dis_tables);

    if (byte_per_idx == 1) {
        
        pq_knn_search_with_tables<uint8_t, CMax<float, long> > (
             this, dis_tables, codes, ncodes, res, init_finalize_heap);

    } else if (byte_per_idx == 2) {
        pq_knn_search_with_tables<uint16_t, CMax<float, long> > (
             this, dis_tables, codes, ncodes, res, init_finalize_heap);

    }

}



/** perform a search (inner product distance)
* @param x        查询向量, size nx * d
* @param nx       查询向量集大小
* @param codes    数据库码本, size ncodes * byte_per_idx
* @param ncodes   数据库码本大小
* @param res      存储结果的堆数组 (nh == nx)
* @param init_finalize_heap  initialize heap (input) and sort (output)?
*/
void ProductQuantizer::search_ip (const float * __restrict x,
                               size_t nx,
                               const uint8_t * codes,
                               const size_t ncodes,
                               float_minheap_array_t * res,
                               bool init_finalize_heap) const
{
    FAISS_THROW_IF_NOT (nx == res->nh);
    float * dis_tables = new float [nx * ksub * M];
    ScopeDeleter<float> del(dis_tables);
    compute_inner_prod_tables (nx, x, dis_tables);

    if (byte_per_idx == 1) {

        pq_knn_search_with_tables<uint8_t, CMin<float, long> > (
             this, dis_tables, codes, ncodes, res, init_finalize_heap);

    } else if (byte_per_idx == 2) {
        pq_knn_search_with_tables<uint16_t, CMin<float, long> > (
             this, dis_tables, codes, ncodes, res, init_finalize_heap);
    }

}


//求平方
static float sqr (float x) {
    return x * x;
}

// 计算质心之间的对称距离 如：
void ProductQuantizer::compute_sdc_table ()
{
    sdc_table.resize (M * ksub * ksub);

    for (int m = 0; m < M; m++) {

        const float *cents = centroids.data() + m * ksub * dsub;
        float * dis_tab = sdc_table.data() + m * ksub * ksub;

        // TODO optimize with BLAS
        for (int i = 0; i < ksub; i++) {
            const float *centi = cents + i * dsub;
            for (int j = 0; j < ksub; j++) {
                float accu = 0;
                const float *centj = cents + j * dsub;
                for (int k = 0; k < dsub; k++)
                    accu += sqr (centi[k] - centj[k]);
                dis_tab [i + j * ksub] = accu;   // dis += tab[bcode[m] + qcode[m] * ksub];
            }
        }
    }
}

/** perform a search (symetic distance)
* @param qcodes   查询向量码本, size nx * d
* @param nq       查询向量集大小
* @param bcodes    数据库码本, size ncodes * byte_per_idx
* @param ncodes   数据库码本大小
* @param res      存储结果的堆数组 (nh == nx)
* @param init_finalize_heap  initialize heap (input) and sort (output)?
*/
void ProductQuantizer::search_sdc (const uint8_t * qcodes,
                     size_t nq,
                     const uint8_t * bcodes,
                     const size_t nb,
                     float_maxheap_array_t * res,
                     bool init_finalize_heap) const
{
    FAISS_THROW_IF_NOT (sdc_table.size() == M * ksub * ksub);
    FAISS_THROW_IF_NOT (byte_per_idx == 1);
    size_t k = res->k;


#pragma omp parallel for
    for (size_t i = 0; i < nq; i++) {

        /* Compute distances and keep smallest values */
        long * heap_ids = res->ids + i * k;
        float *  heap_dis = res->val + i * k;
        const uint8_t * qcode = qcodes + i * code_size;

        if (init_finalize_heap)
            maxheap_heapify (k, heap_dis, heap_ids);

        const uint8_t * bcode = bcodes;
        for (size_t j = 0; j < nb; j++) {
            float dis = 0;
            const float * tab = sdc_table.data();
            for (int m = 0; m < M; m++) {
                dis += tab[bcode[m] + qcode[m] * ksub];  // dis_tab [i + j * ksub] = accu;
                tab += ksub * ksub;
            }
            if (dis < heap_dis[0]) {
                maxheap_pop (k, heap_dis, heap_ids);
                maxheap_push (k, heap_dis, heap_ids, dis, j);
            }
            bcode += code_size;
        }

        if (init_finalize_heap)
            maxheap_reorder (k, heap_dis, heap_ids);
    }

}


} // namespace faiss
