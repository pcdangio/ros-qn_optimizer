#include "qn_optimizer/qn_optimizer.h"

// CONSTRUCTORS
qn_optimizer::qn_optimizer(uint32_t n_dimensions, std::function<double(const Eigen::VectorXd&)> objective_function)
{
    // Store objective function.
    qn_optimizer::objective_function = objective_function;

    // Set default values.
    qn_optimizer::p_initial_step_size = 1.0;
    qn_optimizer::p_tau = 0.75;
    qn_optimizer::p_c1 = 0.0001;
    qn_optimizer::p_c2 = 0.9;
    qn_optimizer::p_epsilon = 0.001;
    qn_optimizer::p_max_iterations = 100;

    // Set default objective gradient function to approximator.
    // NOTE: It may be replaced by an actual gradient function with set_objective_gradient.
    qn_optimizer::objective_gradient = std::bind(&qn_optimizer::gradient_approximator, this, std::placeholders::_1, std::placeholders::_2);

    // Initialize vector/matrix storage.
    qn_optimizer::v_g_k.setZero(n_dimensions);
    qn_optimizer::m_h_k.setZero(n_dimensions, n_dimensions);
    qn_optimizer::v_p_k.setZero(n_dimensions);
    qn_optimizer::v_dx_k.setZero(n_dimensions);
    qn_optimizer::v_dx_k_t.setZero(n_dimensions);
    qn_optimizer::v_y_k.setZero(n_dimensions);
    qn_optimizer::v_y_k_t.setZero(n_dimensions);
    qn_optimizer::v_x_kp.setZero(n_dimensions);
    qn_optimizer::v_g_kp.setZero(n_dimensions);
    qn_optimizer::m_i.setIdentity(n_dimensions, n_dimensions);
    qn_optimizer::m_t1.setZero(n_dimensions, n_dimensions);
    qn_optimizer::m_t2.setZero(n_dimensions, n_dimensions);
}

// USER METHODS
void qn_optimizer::set_objective_gradient(std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> objective_gradient)
{
    qn_optimizer::objective_gradient = objective_gradient;
}
std::vector<uint32_t> qn_optimizer::iterations()
{
    return qn_optimizer::m_iterations;
}


// OPTIMIZATION
bool qn_optimizer::optimize(Eigen::VectorXd& vector, double* final_score)
{
    // Initialize Hessian to identity.
    qn_optimizer::m_h_k.setIdentity();

    // Calculate starting values for f_k, g_k.
    double f_k = qn_optimizer::objective_function(vector);
    double f_kp = f_k;
    qn_optimizer::objective_gradient(vector, qn_optimizer::v_g_k);

    // Iterate until optimized or max iterations reached.
    while(true)
    {
        // Calculate p_k, the step direction.
        qn_optimizer::v_p_k.noalias() = qn_optimizer::m_h_k * qn_optimizer::v_g_k;
        qn_optimizer::v_p_k *= -1.0;

        // Calculate p_k' * g_k for step size calculation.
        // NOTE: vector_a' * vector_b = dot product between a and b.
        double ptg_k = qn_optimizer::v_p_k.dot(qn_optimizer::v_g_k);

        // Calculate step size using backtracking with Wolfe Conditions
        // Start at initial step size.
        double a_k = qn_optimizer::p_initial_step_size;
        uint32_t iterations_step_size = 0;
        while(true)
        {
            // Calculate dx_k.
            qn_optimizer::v_dx_k = a_k * qn_optimizer::v_p_k;
            // Calculate x_k+1 using step direction and current step size.
            qn_optimizer::v_x_kp = qn_optimizer::v_dx_k + vector;
            // Calculate f(x_k+1)
            f_kp = qn_optimizer::objective_function(qn_optimizer::v_x_kp);

            // Check Wolfe Conditions (selectively)
            // Wolfe 1: Armijo
            if(f_kp <= (f_k + qn_optimizer::p_c1 * a_k * ptg_k))
            {
                // Calculate candidate g(x_k+1)
                qn_optimizer::objective_gradient(qn_optimizer::v_x_kp, qn_optimizer::v_g_kp);

                // Wolfe 2: Curvature
                if(qn_optimizer::v_p_k.dot(qn_optimizer::v_g_kp) >= qn_optimizer::p_c2 * ptg_k)
                {
                    // Both Wolfe Conditions met.
                    break;
                }
            }

            // If this point reached, Wolfe Conditions have not been met.
            // Backtrack step size.
            a_k *= qn_optimizer::p_tau;

            // Increment step size iteration count.
            iterations_step_size++;
        }
        // Add step iterations to array.
        qn_optimizer::m_iterations.push_back(iterations_step_size);

        // Suitable step size found.
        // dx_k, x_k+1, f_kp, g_kp are already set.

        // Update vector to x_k+1.
        vector = qn_optimizer::v_x_kp;
        // Update f_k to f_k+1.
        f_k = f_kp;

        // Check if g_k+1 is less than epsilon.
        bool optimized = true;
        for(uint32_t i = 0; i < qn_optimizer::v_g_kp.size(); ++i)
        {
            if(qn_optimizer::v_g_kp(i) > qn_optimizer::p_epsilon)
            {
                optimized = false;
                break;
            }
        }

        // Check if optimization goal reached.
        if(optimized)
        {
            // Goal reached.
            // Capture final score if provided.
            if(final_score)
            {
                *final_score = f_k;
            }
            // Quit and indicate optimal result.
            return true;
        }
        else if(qn_optimizer::m_iterations.size() == qn_optimizer::p_max_iterations)
        {
            // Max iterations reached.
            // Capture final score if provided.
            if(final_score)
            {
                *final_score = f_k;
            }
            // Quit and indicate non-optimal result.
            return false;
        }

        // If this point is reached, optimization must continue.

        // Update g_k to g_k+1.
        qn_optimizer::v_g_k = qn_optimizer::v_g_kp;

        // Calculate y_k.
        qn_optimizer::v_y_k = qn_optimizer::v_g_kp - qn_optimizer::v_g_k;

        // Update Hessian with BFGS.
        double y_dot_dx = qn_optimizer::v_y_k.dot(qn_optimizer::v_dx_k);
        // (I-fraction_1)
        qn_optimizer::m_t1.noalias() = qn_optimizer::v_dx_k * qn_optimizer::v_y_k_t;
        qn_optimizer::m_t1 /= y_dot_dx;
        qn_optimizer::m_t1 = qn_optimizer::m_i - qn_optimizer::m_t1;
        // (I-fraction_1)*H
        qn_optimizer::m_t2.noalias() = qn_optimizer::m_t1 * qn_optimizer::m_h_k;
        // (I-fraction_2)
        qn_optimizer::m_t1.noalias() = qn_optimizer::v_y_k * qn_optimizer::v_dx_k_t;
        qn_optimizer::m_t1 /= y_dot_dx;
        qn_optimizer::m_t1 = qn_optimizer::m_i - qn_optimizer::m_t1;
        // (I-fraction_1)*H*(I-fraction_2)
        qn_optimizer::m_h_k.noalias() = qn_optimizer::m_t2 * qn_optimizer::m_t1;
        // (fraction_3)
        qn_optimizer::m_t1.noalias() = qn_optimizer::v_dx_k * qn_optimizer::v_dx_k_t;
        qn_optimizer::m_t1 /= y_dot_dx;
        // (I-fraction_1)*H*(I-fraction_2) + fraction_3
        qn_optimizer::m_h_k += qn_optimizer::m_t1;
    }
}