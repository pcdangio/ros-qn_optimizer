#ifndef QN_OPTIMIZER_H
#define QN_OPTIMIZER_H

#include <functional>
#include <eigen3/Eigen/Dense>

class qn_optimizer
{
public:
    qn_optimizer(uint32_t n_dimensions, std::function<double(const Eigen::Vector2d&)> objective_function);

    void set_objective_gradient(std::function<void(const Eigen::Vector2d&, Eigen::Vector2d&)> objective_gradient);

    void set_goal(double initial_step_size, double objective_threshold);
    void set_limits(uint32_t max_step_iterations, uint32_t max_optimization_iterations);
    void set_wolfe_constants(double c1, double c2);

    bool optimize(Eigen::Vector2d& vector, double* final_score = nullptr);

    uint32_t last_step_size_iterations();
    uint32_t last_optimization_iterations();

private:
    Eigen::Vector2d v_g_k;
    Eigen::Matrix2d m_h_k;
    Eigen::Vector2d v_p_k;
    Eigen::Vector2d v_dx_k;
    Eigen::Vector2d v_x_kp;
    Eigen::Vector2d v_g_kp;

    double p_initial_step_size;
    double p_objective_threshold;

    double p_c1;
    double p_c2;

    uint32_t p_max_step_iterations;
    uint32_t p_max_optimization_iterations;

    std::function<double(const Eigen::Vector2d&)> m_objective_function;
    std::function<void(const Eigen::Vector2d&, Eigen::Vector2d&)> m_objective_gradient;

    uint32_t m_iterations_step;
    uint32_t m_iterations_optimize;

    void reset();
    void gradient_approximator(const Eigen::Vector2d& operating_point, Eigen::Vector2d& gradient);
};

#endif