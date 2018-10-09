/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_POLYSEMOUS_TRAINING_INCLUDED
#define FAISS_POLYSEMOUS_TRAINING_INCLUDED


#include "ProductQuantizer.h"


namespace faiss {


/// parameters used for the simulated annealing method
/// 模拟退火算法参数
struct SimulatedAnnealingParameters {

    /*
        // set some reasonable defaults for the optimization
        // 为优化设置一些合理的默认值
        init_temperature = 0.7;
        // 0.9 的 1/500.幂
        temperature_decay = pow (0.9, 1/500.);
        // reduce by a factor 0.9 every 500 it
        // 每500次迭代减少0.9倍
        n_iter = 500000;
        n_redo = 2;
        seed = 123;
        verbose = 0;
        only_bit_flips = false;
        init_random = false;
    */

    // optimization parameters
    // 为优化设置一些合理的默认值 init_temperature = 0.7;
    double init_temperature;   // init probaility of accepting a bad swap
    
    // 温度衰减值temperature_decay = pow (0.9, 1/500.)，在每次迭代中，temp乘以此值
    double temperature_decay;  // at each iteration the temp is multiplied by this
    
    // 迭代次数 n_iter = 500000;
    int n_iter; // nb of iterations
    
    // 模拟次数   n_redo = 2;
    int n_redo; // nb of runs of the simulation
    
    // 随机数种子 seed = 123
    int seed;   // random seed
    
    // 日志开光
    int verbose;
    
    // 限制置换更改为位翻转    only_bit_flips = false;
    bool only_bit_flips; // restrict permutation changes to bit flips
    
    // 用随机排列初始化 init_random = false
    bool init_random; // intialize with a random permutation (not identity)

    // set reasonable defaults
    SimulatedAnnealingParameters ();

};


/// abstract class for the loss function
/// 损失函数的抽象类
struct PermutationObjective {

    //每个子量化索引的位数，用来定义每个子空间的聚类个数
    int n;

    // 计算排列 perm 的损失值 cost = 实际距离与汉明距离之间的方差
    virtual double compute_cost (const int *perm) const = 0;

    // what would the cost update be if iw and jw were swapped?
    // default implementation just computes both and computes the difference
    // 如果iw和jw被交换，成本更新会是什么？默认实现只计算两者并计算差异
    virtual double cost_update (const int *perm, int iw, int jw) const;

    virtual ~PermutationObjective () {}
};


struct ReproduceDistancesObjective : PermutationObjective {

    // 距离权重因子
    double dis_weight_factor;

    // 计算x的平方
    static double sqr (double x) { return x * x; }

    // weihgting of distances: it is more important to reproduce small
    // distances well
    // 距离加权：更好地再现小距离很重要重要
    double dis_weight (double x) const;
    
    // 源距离
    std::vector<double> source_dis; ///< "real" corrected distances (size n^2)
    // 想要的距离
    const double *      target_dis; ///< wanted distances (size n^2)
    // 每个距离的权重
    std::vector<double> weights;    ///< weights for each distance (size n^2)

    // 计算i和j的源距离
    double get_source_dis (int i, int j) const;

    // cost = quadratic difference between actual distance and Hamming distance
    // cost = 实际距离与汉明距离之间的方差
    double compute_cost(const int* perm) const override;

    // what would the cost update be if iw and jw were swapped?
    // computed in O(n) instead of O(n^2) for the full re-computation
    // 如果iw和jw被交换，成本更新会是什么？默认实现只计算两者并计算差异
    double cost_update(const int* perm, int iw, int jw) const override;

    ReproduceDistancesObjective (
           int n,
           const double *source_dis_in,
           const double *target_dis_in,
           double dis_weight_factor);

    static void compute_mean_stdev (const double *tab, size_t n2,
                                    double *mean_out, double *stddev_out);

    void set_affine_target_dis (const double *source_dis_in);

    ~ReproduceDistancesObjective() override {}
};

struct RandomGenerator;

/// Simulated annealing optimization algorithm for permutations.
/// 为排列的模拟退火优化算法。
 struct SimulatedAnnealingOptimizer: SimulatedAnnealingParameters {
    /// 损失函数的抽象类
    PermutationObjective *obj;
    // 排列的大小
    int n;         ///< size of the permutation
    // 记录成本函数的值
    FILE *logfile; /// logs values of the cost function

    SimulatedAnnealingOptimizer (PermutationObjective *obj,
                                 const SimulatedAnnealingParameters &p);
    // 随机数生成器
    RandomGenerator *rnd;

    /// remember intial cost of optimization
    /// 记住优化的初始成本
    double init_cost;

    // main entry point. Perform the optimization loop, starting from
    // and modifying permutation in-place
    // 主要入口
    double optimize (int *perm);

    // run the optimization and return the best result in best_perm
    // 运行优化并返回best_perm中的最佳结果
    double run_optimization (int * best_perm);

    virtual ~SimulatedAnnealingOptimizer ();
};




/// optimizes the order of indices in a ProductQuantizer
/// 优化ProductQuantizer中索引的顺序
struct PolysemousTraining: SimulatedAnnealingParameters {

    enum Optimization_type_t {
        OT_None,
        OT_ReproduceDistances_affine,  ///< default
        OT_Ranking_weighted_diff  /// same as _2, but use rank of y+ - rank of y-
    };
    Optimization_type_t optimization_type;

    // use 1/4 of the training points for the optimization, with
    // max. ntrain_permutation. If ntrain_permutation == 0: train on
    // centroids
    // 使用1/4的训练点进行优化，最大值ntrain_permutation。 如果ntrain_permutation == 0：在质心上训练
    int ntrain_permutation;
    // 加权距离损失的指数衰减
    double dis_weight_factor; // decay of exp that weights distance loss

    // filename pattern for the logging of iterations
    // 用于记录迭代的文件名模式
    std::string log_pattern;

    // sets default values
    PolysemousTraining ();

    /// reorder the centroids so that the Hamming distace becomes a
    /// good approximation of the SDC(symmetric product quantizer) distance (called by train)
    /// 重新排序质心，使汉明距离成为SDC距离的良好近似值
    void optimize_pq_for_hamming (ProductQuantizer & pq,
                                  size_t n, const float *x) const;

    /// called by optimize_pq_for_hamming
    void optimize_ranking (ProductQuantizer &pq, size_t n, const float *x) const;
    /// called by optimize_pq_for_hamming
    void optimize_reproduce_distances (ProductQuantizer &pq) const;

};


} // namespace faiss


#endif
