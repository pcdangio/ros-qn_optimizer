#include "qn_optimizer/qn_optimizer.h"

// CONSTRUCTORS
qn_optimizer::qn_optimizer(uint32_t n_dimensions, std::function<double(const Eigen::Vector2d&)> objective_function)
{
    // Store objective function.
    qn_optimizer::m_objective_function = objective_function;

    // Set default values.
    qn_optimizer::p_initial_step_size = 0.1;
    qn_optimizer::p_objective_threshold = 0;
    qn_optimizer::p_max_step_iterations = 10;
    qn_optimizer::p_max_optimization_iterations = 100;
    qn_optimizer::p_c1 = 0.0001;
    qn_optimizer::p_c2 = 0.9;

    // Initialize counters.
    qn_optimizer::m_iterations_step = 0;
    qn_optimizer::m_iterations_optimize = 0;

    // Set default objective gradient function to approximator.
    // NOTE: It may be replaced by an actual gradient function with set_objective_gradient.
    qn_optimizer::m_objective_gradient = std::bind(&qn_optimizer::gradient_approximator, this, std::placeholders::_1, std::placeholders::_2);
}

// INITIALIZATION
void qn_optimizer::set_objective_gradient(std::function<void(const Eigen::Vector2d&, Eigen::Vector2d&)> objective_gradient)
{
    // Store gradient function.
    qn_optimizer::m_objective_gradient = objective_gradient;
}
void qn_optimizer::set_goal(double initial_step_size, double objective_threshold)
{
    qn_optimizer::p_initial_step_size = initial_step_size;
    qn_optimizer::p_objective_threshold = objective_threshold;
}
void qn_optimizer::set_limits(uint32_t max_step_iterations, uint32_t max_optimization_iterations)
{
    qn_optimizer::p_max_step_iterations = max_step_iterations;
    qn_optimizer::p_max_optimization_iterations = max_optimization_iterations;
}
void qn_optimizer::set_wolfe_constants(double c1, double c2)
{
    qn_optimizer::p_c1 = c1;
    qn_optimizer::p_c2 = c2;
}

// OPTIMIZATION
bool qn_optimizer::optimize(Eigen::Vector2d& vector, double* final_score)
{
    // Reset all vectors/matrices.
    // set hessian to identity
    // set step size to initial step size
    double a_k = qn_optimizer::p_initial_step_size;
    // Reset notional values.
    double f_kp = 0;
    // Reset wolfe tracker to true (assuming 0 start)
    bool wolfe_k = true;

    // Calculate starting values for f_k, g_k.
    double f_k = qn_optimizer::m_objective_function(vector);
    qn_optimizer::m_objective_gradient(vector, qn_optimizer::v_g_k);
    

    // Loop until max iterations.
    for(uint32_t k = 0; k < qn_optimizer::p_max_optimization_iterations; ++k)
    {
        // Calculate p_k, the step direction.
        qn_optimizer::v_p_k.noalias() = qn_optimizer::m_h_k * qn_optimizer::v_g_k;
        qn_optimizer::v_p_k *= -1.0;

        // Calculate p_k' * g_k for step size calculation.
        // NOTE: vector_a' * vector_b = dot product between a and b.
        double ptg_k = qn_optimizer::v_p_k.dot(qn_optimizer::v_g_k);

        // Begin step size calculation.
        uint32_t h = 0;
        for(uint32_t i = 0; i < qn_optimizer::p_max_step_iterations; ++i)
        {
            // Calculate candidate dx_k.
            qn_optimizer::v_dx_k = a_k * qn_optimizer::v_p_k;
            // Calculate candidate x_k+1 using step direction and current step size.
            qn_optimizer::v_x_kp = qn_optimizer::v_dx_k + vector;
            // Calculate candidate f(x_k+1)
            f_kp = qn_optimizer::m_objective_function(qn_optimizer::v_x_kp);

            // Check Wolfe Conditions (selectively)
            bool wolfe = false;
            // Wolfe 1: Armijo
            if(f_kp <= (f_k + qn_optimizer::p_c1 * a_k * ptg_k))
            {
                // Calculate candidate g(x_k+1)
                qn_optimizer::m_objective_gradient(qn_optimizer::v_x_kp, qn_optimizer::v_g_kp);

                // Wolfe 2: Curvature
                if(qn_optimizer::v_p_k.dot(qn_optimizer::v_g_kp) >= qn_optimizer::p_c2 * ptg_k)
                {
                    wolfe = true;
                }
            }

            // Update step with either a hop or a jump.
            if(wolfe == wolfe_k)
            {
                // Perform a jump.
                if(wolfe)
                {
                    a_k += a_k;
                }
                
            }
            else
            {
                // Perform a hop.
            }

            // Update wolfe_k.
            wolfe_k = wolfe;
            
            // Increment step iterations.
            qn_optimizer::m_iterations_step++;
        }

        // CHECK SCORE HERE


        // Increment optimization iteration counter.
        qn_optimizer::m_iterations_optimize++;
    }
}


// PROPERTIES
uint32_t qn_optimizer::last_step_size_iterations()
{
    return qn_optimizer::m_iterations_step;
}
uint32_t qn_optimizer::last_optimization_iterations()
{
    return qn_optimizer::m_iterations_optimize;
}