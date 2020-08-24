/// \file qn_optimizer.h
/// \brief Defines the qn_optimizer class.
#ifndef QN_OPTIMIZER_H
#define QN_OPTIMIZER_H

#include <vector>
#include <functional>
#include <eigen3/Eigen/Dense>

/// \brief Performs Quasi-Newton optimization.
class qn_optimizer
{
public:
    // CONSTRUCTORS
    /// \brief Creates a new qn_optimizer instance.
    /// \param n_dimensions The number of dimensions in the objective function.
    /// \param objective_function The objective function to minimize.
    /// \note This provides a default gradient function that estimates the gradient, but increases computational complexity proportional to the dimensionality.
    qn_optimizer(uint32_t n_dimensions, std::function<double(const Eigen::VectorXd&)> objective_function);
    /// \brief Creates a new qn_optimizer instance.
    /// \param n_dimensions The number of dimensions in the objective function.
    /// \param objective_function The objective function to minimize.
    /// \param objective_gradient The gradient function for the objective function.
    qn_optimizer(uint32_t n_dimensions, std::function<double(const Eigen::VectorXd&)> objective_function, std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> objective_gradient);

    // PARAMETERS
    /// \brief The starting step size to begin the step size search from. DEFAULT = 1.0
    double p_initial_step_size;
    /// \brief The multiplier (0 < tau < 1) to apply to the step size during step size search. DEFAULT = 0.75
    double p_tau;
    /// \brief Armijo rule constant (0 < c1 << c2 < 1). DEFAULT = 0.0001
    double p_c1;
    /// \brief Curvature rule constant (0 < c1 << c2 < 1). DEFAULT = 0.9
    double p_c2;
    /// \brief A threshold specifying when the gradient can be considered "0" and optimization is complete. DEFAULT = 0.00001
    /// \details The absolute value of the gradient in each and every dimension must fall below this threshold.
    double p_epsilon;
    /// \brief The amount to perturb each dimension when estimating the gradient of the objective function. DEFAULT = 0.0000000001
    /// \note This parameter is only used when a gradient function was not provided by the user.
    double p_perturbation;
    /// \brief The maximum number of optimization iterations before aborting. DEFAULT = infinity
    uint32_t p_max_iterations;
    /// \brief The maximum number of step size search iterations before aborting. DEFAULT = infinity
    uint32_t p_max_step_iterations;

    // METHODS
    /// \brief Performs the optimization routine.
    /// \param vector The vector to optimize. Beforehand, this should be set to an initial guess. Afterwards, it will contain the optimized values.
    /// \param final_score OPTIONAL Captures the final value of the objective function using the optimized values output in vector.
    /// \returns TRUE if the routine succeeded, otherwise FALSE if the route was aborted due to max iterations.
    bool optimize(Eigen::VectorXd& vector, double* final_score = nullptr);

    // PROPERTIES
    /// \brief Indicates the number of optimization and step size search iterations from the most recent optimize() call.
    /// \details The size of this vector indicates the number of optimization iterations. Each element in the vector represents
    /// the number of step size search iterations for each optimization iteration.
    /// \returns The number of optimization and step size search iterations.
    std::vector<uint32_t> iterations() const;

private:
    // MATRICES
    /// \brief The objective function gradient at the current optimization step, k.
    Eigen::VectorXd v_g_k;
    /// \brief The estimated inverse Hessian at the current optimization step, k.
    Eigen::MatrixXd m_h_k;
    /// \brief The step direction for the current optimization step, k.
    Eigen::VectorXd v_p_k;
    /// \brief The variable delta vector for the current optimization step, k.
    Eigen::VectorXd v_dx_k;
    /// \brief The transponse of the variable delta vector for the current optimization step, k.
    Eigen::MatrixXd v_dx_k_t;
    /// \brief The gradient delta vector for the current optimization step, k.
    Eigen::VectorXd v_y_k;
    /// \brief The transponse of the gradient delta vector for the current optimization step, k.
    Eigen::MatrixXd v_y_k_t;
    /// \brief The updated variable vector for the next optimization step, k+1.
    Eigen::VectorXd v_x_kp;
    /// \brief The updated objective function gradient for the next optimization step, k+1.
    Eigen::VectorXd v_g_kp;
    /// \brief The identity matrix for inverse Hessian updates.
    Eigen::MatrixXd m_i;
    /// \brief A temporary matrix for inverse Hessian updates.
    Eigen::MatrixXd m_t1;
    /// \brief A second temporary matrix for inverse Hessian updates.
    Eigen::MatrixXd m_t2;
    /// \brief Stores a perturbed version of the operating point in gradient estimation.
    Eigen::VectorXd v_xp;

    // FUNCTIONS
    /// \brief Stores the objective function to optimize.
    std::function<double(const Eigen::VectorXd&)> objective_function;
    /// \brief Stores the gradient function for the objective function.
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> objective_gradient;

    // COUNTERS
    /// \brief A counter for optimization and step size search iterations.
    std::vector<uint32_t> m_iterations;

    // METHODS
    /// \brief Initializes the class instance.
    /// \param n_dimensions The number of dimensions in the objective function.
    void initialize(uint32_t n_dimensions);
    /// \brief The default gradient function for approximating the objective function gradient through perturbation.
    /// \param operating_point The point at which to estimate the gradient.
    /// \param gradient Captures the calculated gradient.
    void gradient_approximator(const Eigen::VectorXd& operating_point, Eigen::VectorXd& gradient);
};

#endif