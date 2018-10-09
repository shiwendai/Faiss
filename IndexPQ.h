/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_INDEX_PQ_H
#define FAISS_INDEX_PQ_H

#include <stdint.h>

#include <vector>

#include "Index.h"
#include "ProductQuantizer.h"
#include "PolysemousTraining.h"

namespace faiss {


/** Index based on a product quantizer. Stored vectors are
 * approximated by PQ codes. */
/* 基于乘积量化的索引。 存储的向量由PQ码近似表示 */
struct IndexPQ: Index {

    /// The product quantizer used to encode the vectors
    /// 乘积量化器，用来给向量集进行pq编码
    ProductQuantizer pq;

    /// 码本  Codes. Size ntotal * pq.code_size(码字大小)
    std::vector<uint8_t> codes;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param M      number of subquantizers
     * @param nbits  number of bit per subvector index
     */
    IndexPQ (int d,                    ///< dimensionality of the input vectors
             size_t M,                 ///< number of subquantizers
             size_t nbits,             ///< number of bit per subvector index
             MetricType metric = METRIC_L2);

    IndexPQ ();

    void train(idx_t n, const float* x) override;

    void add(idx_t n, const float* x) override;

    void search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const override;

    void reset() override;

    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    void reconstruct(idx_t key, float* recons) const override;

    long remove_ids(const IDSelector& sel) override;

    /******************************************************
     * Polysemous codes implementation 多义代码实现
     ******************************************************/
    bool do_polysemous_training; ///< false = standard PQ

    /// parameters used for the polysemous training
    /// 用于多义训练的参数
    PolysemousTraining polysemous_training;

    /// how to perform the search in search_core
    /// 如何在search_core中执行搜索
    enum Search_type_t {
        ST_PQ,             ///< asymmetric product quantizer (default)
        ST_HE,             ///< Hamming distance on codes
        ST_generalized_HE, ///< nb of same codes
        ST_SDC,            ///< symmetric product quantizer (SDC)
        ST_polysemous,     ///< HE filter (using ht) + PQ combination
        ST_polysemous_generalize,  ///< Filter on generalized Hamming
    };

    Search_type_t search_type;

    // just encode the sign of the components, instead of using the PQ encoder
    // used only for the queries
    // 只编码组件的符号，而不是使用仅用于查询的PQ编码器
    bool encode_signs;

    /// Hamming threshold used for polysemy
    /// 用于多义词的汉明阈值
    int polysemous_ht;

    // actual polysemous search
    // 实际的多义搜索
    void search_core_polysemous (idx_t n, const float *x, idx_t k,
                               float *distances, idx_t *labels) const;

    /// prepare query for a polysemous search, but instead of
    /// computing the result, just get the histogram of Hamming
    /// distances. May be computed on a provided dataset if xb != NULL
    /// @param dist_histogram (M * nbits + 1)
    ///
    /// 准备查询多义搜索，但不是计算结果，只需获得汉明距离的直方图。 
    /// 如果xb！= NULL，则可以在提供的数据集上计算
    void hamming_distance_histogram (idx_t n, const float *x,
                                     idx_t nb, const float *xb,
                                     long *dist_histogram);

    /** compute pairwise distances between queries and database
     *  计算查询和数据库之间的成对距离
     *
     * @param n    nb of query vectors
     * @param x    query vector, size n * d
     * @param dis  output distances, size n * ntotal
     */
    void hamming_distance_table (idx_t n, const float *x,
                                 int32_t *dis) const;

};


/// statistics are robust to internal threading, but not if
/// IndexPQ::search is called by multiple threads
/// 统计信息对内部线程是健壮的，但如果多个线程调用IndexPQ :: search则不行
struct IndexPQStats {
    size_t nq;       // nb of queries run
    size_t ncode;    // nb of codes visited

    size_t n_hamming_pass; // nb of passed Hamming distance tests (for polysemy)

    IndexPQStats () {reset (); }
    void reset ();
};

extern IndexPQStats indexPQ_stats;



/** Quantizer where centroids are virtual: they are the Cartesian
 *  product of sub-centroids.
 *  质心是虚拟的量化器：它们是子质心的笛卡尔积。
 */
struct MultiIndexQuantizer: Index  {
    ProductQuantizer pq;

    MultiIndexQuantizer (int d,         ///< dimension of the input vectors
                         size_t M,      ///< number of subquantizers
                         size_t nbits); ///< number of bit per subvector index

    void train(idx_t n, const float* x) override;

    void search(
        idx_t n, const float* x, idx_t k,
        float* distances, idx_t* labels) const override;

    /// add and reset will crash at runtime
    void add(idx_t n, const float* x) override;
    void reset() override;

    MultiIndexQuantizer () {}

    void reconstruct(idx_t key, float* recons) const override;
};


/** MultiIndexQuantizer where the PQ assignmnet is performed by sub-indexes
 */
struct MultiIndexQuantizer2: MultiIndexQuantizer {

    /// M Indexes on d / M dimensions
    std::vector<Index*> assign_indexes;
    bool own_fields;

    MultiIndexQuantizer2 (
        int d, size_t M, size_t nbits,
        Index **indexes);

    MultiIndexQuantizer2 (
        int d, size_t nbits,
        Index *assign_index_0,
        Index *assign_index_1);

    void train(idx_t n, const float* x) override;

    void search(
        idx_t n, const float* x, idx_t k,
        float* distances, idx_t* labels) const override;

};


} // namespace faiss


#endif
