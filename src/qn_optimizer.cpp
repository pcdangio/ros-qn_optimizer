#include "qn_optimizer/qn_optimizer.h"

// CONSTRUCTORS
qn_optimizer::qn_optimizer(uint32_t n_dimensions, std::function<double(const Eigen::VectorXd&)> objective_function)
{
    // Initialize members.
    qn_optimizer::initialize(n_dimensions);

    // Store objective function.
    qn_optimizer::objective_function = objective_function;

    // Set default objective gradient function to approximator.
    qn_optimizer::objective_gradient = std::bind(&qn_optimizer::gradient_approximator, this, std::placeholders::_1, std::placeholders::_2);
}
qn_optimizer::qn_optimizer(uint32_t n_dimensions, std::function<double(const Eigen::VectorXd&)> objective_function, std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> objective_gradient)
{
    // Initialize members.
    qn_optimizer::initialize(n_dimensions);

    // Store objective function.
    qn_optimizer::objective_function = objective_function;

    // Store objective gradient function.
    qn_optimizer::objective_gradient = objective_gradient;
}
void qn_optimizer::initialize(uint32_t n_dimensions)
{
    // Set default values.
    qn_optimizer::p_initial_step_size = 1.0;
    qn_optimizer::p_tau = 0.75;
    qn_optimizer::p_c1 = 0.0001;
    qn_optimizer::p_c2 = 0.9;
    qn_optimizer::p_epsilon = 0.00001;
    qn_optimizer::p_perturbation = 0.0000000001;
    qn_optimizer::p_max_iterations = 100;
    qn_optimizer::p_max_step_iterations = 100;

    // Initialize vector/matrix storage.
    qn_optimizer::v_g_k.setZero(n_dimensions);
    qn_optimizer::m_h_k.setZero(n_dimensions, n_dimensions);
    qn_optimizer::v_p_k.setZero(n_dimensions);
    qn_optimizer::v_dx_k.setZero(n_dimensions);
    qn_optimizer::v_dx_k_t.setZero(1, n_dimensions);
    qn_optimizer::v_y_k.setZero(n_dimensions);
    qn_optimizer::v_y_k_t.setZero(1, n_dimensions);
    qn_optimizer::v_x_kp.setZero(n_dimensions);
    qn_optimizer::v_g_kp.setZero(n_dimensions);
    qn_optimizer::m_i.setIdentity(n_dimensions, n_dimensions);
    qn_optimizer::m_t1.setZero(n_dimensions, n_dimensions);
    qn_optimizer::m_t2.setZero(n_dimensions, n_dimensions);
    qn_optimizer::v_xp.setZero(n_dimensions);
}

// USER METHODS
std::vector<uint32_t> qn_optimizer::iterations()
{
    return qn_optimizer::m_iterations;
}

// OPTIMIZATION
bool qn_optimizer::optimize(Eigen::VectorXd& vector, double* final_score)
{
    // Initialize Hessian to identity.
    qn_optimizer::m_h_k.setIdentity();
    // Reset iteration count vector.
    qn_optimizer::m_iterations.clear();

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
            // Increment step size iteration count.
            iterations_step_size++;
            if(iterations_step_size > qn_optimizer::p_max_step_iterations)
            {
                break;
            }

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

            // Backtrack step size.
            a_k *= qn_optimizer::p_tau;
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
            if(std::abs(qn_optimizer::v_g_kp(i)) > qn_optimizer::p_epsilon)
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

        // Check if max iterations reached.
        if(qn_optimizer::m_iterations.size() == qn_optimizer::p_max_iterations)
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

        // Transpose dx_k.
        qn_optimizer::v_dx_k_t = qn_optimizer::v_dx_k.transpose();
        // Calculate y_k.
        qn_optimizer::v_y_k = qn_optimizer::v_g_kp - qn_optimizer::v_g_k;
        qn_optimizer::v_y_k_t = qn_optimizer::v_y_k.transpose();

        // Update g_k to g_k+1.
        qn_optimizer::v_g_k = qn_optimizer::v_g_kp;

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
void qn_optimizer::gradient_approximator(const Eigen::VectorXd& operating_point, Eigen::VectorXd& gradient)
{
    // Evaluate the objective function at the operating point.
    double f1 = qn_optimizer::objective_function(operating_point);

    // Iterate over each variable, perturbing it and calculating the derivative.
    for(uint32_t i = 0; i < operating_point.size(); ++i)
    {
        // Create the perturbed vector.
        qn_optimizer::v_xp = operating_point;
        qn_optimizer::v_xp(i) += qn_optimizer::p_perturbation;

        // Evaluate objective function at perturbed operating point.
        double f2 = qn_optimizer::objective_function(qn_optimizer::v_xp);

        // Calculate derivative and store in gradient vector.
        gradient(i) = (f2 - f1)/qn_optimizer::p_perturbation;
    }
}