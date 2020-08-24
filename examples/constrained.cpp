#include "qn_optimizer/qn_optimizer.h"

#include <iostream>

// TUNABLE PARAMETERS
const double phase_offset = 1.0;
const double left_bound = 2.0;
const double right_bound = 6.0;
const double initial_guess = 2.5;

// OBJECTIVE FUNCTION
double objective_function(const Eigen::VectorXd& variables)
{
    return std::sin(phase_offset + variables(0)) - 0.00001/(variables(0) - right_bound) + 0.00001/(variables(0) - left_bound);
}
void objective_gradient(const Eigen::VectorXd& operating_point, Eigen::VectorXd& gradient)
{
    gradient(0) = std::cos(phase_offset + operating_point(0)) + 0.00001/std::pow(operating_point(0) - right_bound,2.0) - 0.00001/std::pow(operating_point(0) - left_bound, 2.0);
}

int32_t main(int32_t argc, char** argv)
{
    //qn_optimizer qno(1, &objective_function);
    qn_optimizer qno(1, &objective_function, &objective_gradient);

    // Set up the variable vector and insert initial guess.
    Eigen::VectorXd variables;
    variables.setZero(1);
    variables(0) = initial_guess;

    // Run optimization.
    double score;
    bool result = qno.optimize(variables, &score);

    // Display result.
    if(result)
    {
        std::cout << "minimized value: " << std::endl << variables << std::endl;
    }
    else
    {
        std::cout << "optimization failed" << std::endl;
    }    

    // Display number of iterations.
    auto iterations = qno.iterations();
    std::cout << "total iterations: " << iterations.size() << std::endl;
    for(auto iteration = iterations.cbegin(); iteration != iterations.cend(); ++iteration)
    {
        std::cout << *iteration << std::endl;
    }
}