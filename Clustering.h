/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_CLUSTERING_H
#define FAISS_CLUSTERING_H
#include "Index.h"

#include <vector>

namespace faiss {


/** Class for the clustering parameters. Can be passed to the
 * constructor of the Clustering object.
 * 聚类参数类，
 */
struct ClusteringParameters {
    //聚类迭代次数（迭代次数是指：在一次聚类中，不断的聚类迭代以获取质心的过程）
    int niter;          ///< clustering iterations
    //聚类次数（聚类次数是指：每次聚类都会得到一个质心，最终从这nredo中去最优的质心的过程）
    int nredo;          ///< redo clustering this many times and keep best

    //是否打印日志
    bool verbose;
    //是否需要标准化的质心
    bool spherical;     ///< do we want normalized centroids?
    //在每次迭代中更新索引
    bool update_index;  ///< update index after each iteration?
    //用提供的质心作为输入，并且在迭代中不改变质心
    bool frozen_centroids;  ///< use the centroids provided as input and do not change them during iterations

    //每个聚类至少包含数据集大小
    int min_points_per_centroid; ///< otherwise you get a warning
    //每个聚类最多包含的数据集大小
    int max_points_per_centroid;  ///< to limit size of dataset

    //产生随机数的种子
    int seed; ///< seed for the random number generator

    /// sets reasonable defaults
    ClusteringParameters ();
};


/** clustering based on assignment - centroid update iterations
 *
 * The clustering is based on an Index object that assigns training
 * points to the centroids. Therefore, at each iteration the centroids
 * are added to the index.
 *
 * On output, the centoids table is set to the latest version
 * of the centroids and they are also added to the index. If the
 * centroids table it is not empty on input, it is also used for
 * initialization.
 *
 * To do several clusterings, just call train() several times on
 * different training sets, clearing the centroid table in between.
 */
struct Clustering: ClusteringParameters {
    /// typedef long idx_t; 所有索引都是这个类型
    typedef Index::idx_t idx_t;
    /// 向量维数
    size_t d;              ///< dimension of the vectors
    /// 质心的数量
    size_t k;              ///< nb of centroids

    /// 存储所有质心数据大小为 k * d
    std::vector<float> centroids;  /// vector centroids (k * d)

    /// objective values (sum of distances reported by index) over
    /// iterations
    /// 聚类时，每次迭代的距离值的和（所有距离之和）
    std::vector<float> obj;

    /// the only mandatory parameters are k and d
    /// 构造函数强制入参只有 k 和 d
    Clustering (int d, int k);
    Clustering (int d, int k, const ClusteringParameters &cp);

    /// Index is used during the assignment stage
    /** simplified interface
     *
     * @param n 待训练的数据集数量
     * @param x 训练集 (大小n * d)
     * @param index 被用来在训练阶段计算点到质心的距离
     */
    virtual void train (idx_t n, const float * x, faiss::Index & index);

    virtual ~Clustering() {}
};


/** simplified interface
 *
 * @param d dimension of the data
 * @param n nb of training vectors
 * @param k nb of output centroids
 * @param x training set (size n * d)
 * @param centroids output centroids (size k * d)
 * @return final quantization error
 */
float kmeans_clustering (size_t d, size_t n, size_t k,
                         const float *x,
                         float *centroids);



}


#endif
