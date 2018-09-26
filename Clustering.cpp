/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "Clustering.h"


#include <cmath>
#include <cstdio>
#include <cstring>

#include "utils.h"
#include "FaissAssert.h"
#include "IndexFlat.h"

namespace faiss {

ClusteringParameters::ClusteringParameters ():
    niter(25),
    nredo(1),
    verbose(false), spherical(false),
    update_index(false),
    frozen_centroids(false),
    min_points_per_centroid(39),
    max_points_per_centroid(256),
    seed(1234)
{}
// 39 corresponds to 10000 / 256 -> to avoid warnings on PQ tests with randu10k


Clustering::Clustering (int d, int k):
    d(d), k(k) {}

Clustering::Clustering (int d, int k, const ClusteringParameters &cp):
    ClusteringParameters (cp), d(d), k(k) {}



static double imbalance_factor (int n, int k, long *assign) {
    std::vector<int> hist(k, 0);
    for (int i = 0; i < n; i++)
        hist[assign[i]]++;

    double tot = 0, uf = 0;

    for (int i = 0 ; i < k ; i++) {
        tot += hist[i];
        uf += hist[i] * (double) hist[i];
    }
    uf = uf * k / (tot * tot);

    return uf;
}




void Clustering::train (idx_t nx, const float *x_in, Index & index) {
    FAISS_THROW_IF_NOT_FMT (nx >= k,
             "Number of training points (%ld) should be at least "
             "as large as number of clusters (%ld)", nx, k);

    double t0 = getmillisecs();

    // yes it is the user's responsibility, but it may spare us some
    // hard-to-debug reports.
    for (size_t i = 0; i < nx * d; i++) {
      FAISS_THROW_IF_NOT_MSG (finite (x_in[i]),
                        "input contains NaN's or Inf's");
    }

    const float *x = x_in;
    ScopeDeleter<float> del1;

    /// 如果训练集数量大于 k * max_points_per_centroid（即：质心数 * 每个聚类最大数量）时
    /// 随机选取里面的 k * max_points_per_centroid 个数据作为训练集
    if (nx > k * max_points_per_centroid) {
        if (verbose)
            printf("Sampling a subset of %ld / %ld for training\n",
                   k * max_points_per_centroid, nx);
        std::vector<int> perm (nx);
        rand_perm (perm.data (), nx, seed);
        nx = k * max_points_per_centroid;
        float * x_new = new float [nx * d];
        for (idx_t i = 0; i < nx; i++)
            memcpy (x_new + i * d, x + perm[i] * d, sizeof(x_new[0]) * d);
        x = x_new;
        del1.set (x);
    /// 如果训练集数量小于k * min_points_per_centroid（即：质心数 * 每个聚类至少数量）时
    /// 打印警告日志，但程序正常运行
    } else if (nx < k * min_points_per_centroid) {
        fprintf (stderr,
                 "WARNING clustering %ld points to %ld centroids: "
                 "please provide at least %ld training points\n",
                 nx, k, idx_t(k) * min_points_per_centroid);
    }

    /// 如果训练集大小等于k时，则直接把训练集数据拷贝给centroids作为质心
    if (nx == k) {
        if (verbose) {
            printf("Number of training points (%ld) same as number of "
                   "clusters, just copying\n", nx);
        }
        // this is a corner case, just copy training set to clusters
        centroids.resize (d * k);
        memcpy (centroids.data(), x_in, sizeof (*x_in) * d * k);
        index.reset();
        index.add(k, x_in);
        return;
    }


    if (verbose)
        printf("Clustering %d points in %ldD to %ld clusters, "
               "redo %d times, %d iterations\n",
               int(nx), d, k, nredo, niter);



    /// 用于保存在检索聚类过程中，检索集中每个数据点归属的类编号（即质心编号,如果质心数为255则编号为1只255）（因为检索集为nx，所以dis大小也为nx）
    idx_t * assign = new idx_t[nx];
    ScopeDeleter<idx_t> del (assign);
    /// 用于保存在检索聚类过程中，检索集中每个数据点到归属类的质心的距离（因为检索集为nx，所以dis大小也为nx）
    float * dis = new float[nx];
    ScopeDeleter<float> del2(dis);

    // for redo
    float best_err = HUGE_VALF;
    std::vector<float> best_obj;
    std::vector<float> best_centroids;

    // support input centroids

    FAISS_THROW_IF_NOT_MSG (
       centroids.size() % d == 0,
       "size of provided input centroids not a multiple of dimension");

    ///质心个数
    size_t n_input_centroids = centroids.size() / d;

    if (verbose && n_input_centroids > 0) {
        printf ("  Using %zd centroids provided as input (%sfrozen)\n",
                n_input_centroids, frozen_centroids ? "" : "not ");
    }

    double t_search_tot = 0;
    if (verbose) {
        printf("  Preprocessing in %.2f s\n",
               (getmillisecs() - t0)/1000.);
    }
    t0 = getmillisecs();

    
    /// 聚类算法核心代码
    /// 对训练集做redo次聚类，并取redo次聚类中最优的质心
    for (int redo = 0; redo < nredo; redo++) {

        if (verbose && nredo > 1) {
            printf("Outer iteration %d / %d\n", redo, nredo);
        }


        // initialize remaining centroids with random points from the dataset
        /// 从训练集中获取随机点（一个点即一个d维向量）初始化剩余的质心
        centroids.resize (d * k);
        std::vector<int> perm (nx);

        /// 
        rand_perm (perm.data(), nx, seed + 1 + redo * 15486557L);
        for (int i = n_input_centroids; i < k ; i++)
            memcpy (&centroids[i * d], x + perm[i] * d,
                    d * sizeof (float));

        /// 是否需要标准化的质心
        if (spherical) {
            fvec_renorm_L2 (d, k, centroids.data());
        }

        /// 清理index 索引数据
        if (index.ntotal != 0) {
            index.reset();
        }

        /// index 索引是否训练
        if (!index.is_trained) {
            index.train (k, centroids.data());
        }

        /// 向index索引对象中插入数据
        index.add (k, centroids.data());
        /// 计算检索集中各点到对相应质心距离和
        float err = 0;

        /// 聚类迭代,获取质心
        for (int i = 0; i < niter; i++) {
            double t0s = getmillisecs();
            /// 利用相似性搜索对nx个检索数据集x进行聚类，把数据集中的每个数据点的归属于某个类
            /// 并把该数据点归属于某个类编号（即质心序号）保存在assign中，该检索数据集大小为nx，所以assign大小为nx
            /// 该数据点到改质心的距离保存在dis中，该检索数据集大小为nx，所以dis大小为nx
            /// 该数据点到改质心的距离保存在dis中，
            index.search (nx, x, 1, dis, assign);
            t_search_tot += getmillisecs() - t0s;


            /// 计算检索集中各点到对相应质心距离和，并把他存入obj中
            err = 0;
            for (int j = 0; j < nx; j++)
                err += dis[j];
            obj.push_back (err);

            //根据聚类结果重新计算质心
            int nsplit = km_update_centroids (
                  x, centroids.data(),
                  assign, d, k, nx, frozen_centroids ? n_input_centroids : 0);

            if (verbose) {
                printf ("  Iteration %d (%.2f s, search %.2f s): "
                        "objective=%g imbalance=%.3f nsplit=%d       \r",
                        i, (getmillisecs() - t0) / 1000.0,
                        t_search_tot / 1000,
                        err, imbalance_factor (nx, k, assign),
                        nsplit);
                fflush (stdout);
            }

            /// 是否需要标准化的质心
            if (spherical)
                fvec_renorm_L2 (d, k, centroids.data());

            /// 重置并用最新的质心训练index索引对象
            index.reset ();
            if (update_index)
                index.train (k, centroids.data());

            assert (index.ntotal == 0);
            /// index训练完之后，在用质心初始化index
            index.add (k, centroids.data());
        }
        if (verbose) printf("\n");
        if (nredo > 1) {
            if (err < best_err) {
                if (verbose)
                    printf ("Objective improved: keep new clusters\n");
                best_centroids = centroids;
                best_obj = obj;
                best_err = err;
            }
            index.reset ();
        }
    }
    if (nredo > 1) {
        centroids = best_centroids;
        obj = best_obj;
        index.reset();
        index.add(k, best_centroids.data());
    }

}

float kmeans_clustering (size_t d, size_t n, size_t k,
                         const float *x,
                         float *centroids)
{
    Clustering clus (d, k);
    clus.verbose = d * n * k > (1L << 30);
    // display logs if > 1Gflop per iteration
    IndexFlatL2 index (d);
    clus.train (n, x, index);
    memcpy(centroids, clus.centroids.data(), sizeof(*centroids) * d * k);
    return clus.obj.back();
}

} // namespace faiss
