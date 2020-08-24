#ifndef QN_OPTIMIZER_H
#define QN_OPTIMIZER_H

#include <vector>
#include <functional>
#include <eigen3/Eigen/Dense>

class qn_optimizer
{
public:
    qn_optimizer(uint32_t n_dimensions, std::function<double(const Eigen::VectorXd&)> objective_function);

    double p_initial_step_size;
    double p_tau;
    double p_c1;
    double p_c2;
    double p_epsilon;
    double p_perturbation;
    uint32_t p_max_iterations;

    void set_objective_gradient(std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> objective_gradient);

    bool optimize(Eigen::VectorXd& vector, double* final_score = nullptr);

    std::vector<uint32_t> iterations();

private:
    Eigen::VectorXd v_g_k;
    Eigen::MatrixXd m_h_k;
    Eigen::VectorXd v_p_k;
    Eigen::VectorXd v_dx_k;
    Eigen::MatrixXd v_dx_k_t;
    Eigen::VectorXd v_y_k;
    Eigen::MatrixXd v_y_k_t;
    Eigen::VectorXd v_x_kp;
    Eigen::VectorXd v_g_kp;
    Eigen::MatrixXd m_i;
    Eigen::MatrixXd m_t1;
    Eigen::MatrixXd m_t2;
    Eigen::VectorXd v_xp;

    std::function<double(const Eigen::VectorXd&)> objective_function;
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> objective_gradient;

    std::vector<uint32_t> m_iterations;

    void gradient_approximator(const Eigen::VectorXd& operating_point, Eigen::VectorXd& gradient);
};

#endif