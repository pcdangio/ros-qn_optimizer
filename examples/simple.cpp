#include "qn_optimizer/qn_optimizer.h"

#include <iostream>

// TUNABLE PARAMETERS
const double min_a = -4.75;
const double min_b = 2.0;
const double initial_guess_a = 0.0;
const double initial_guess_b = 0.0;

double objective_function(const Eigen::VectorXd& variables)
{
    return std::pow(variables(0) - min_a, 2.0) + std::pow(variables(1) - min_b, 2.0);
}
void objective_gradient(const Eigen::VectorXd& operating_point, Eigen::VectorXd& gradient)
{
    gradient(0) = 2*(operating_point(0) - min_a);
    gradient(1) = 2*(operating_point(1) - min_b);
}

int32_t main(int32_t argc, char** argv)
{
    //qn_optimizer qno(2, &objective_function);
    qn_optimizer qno(2, &objective_function, &objective_gradient);

    // Set up the variable vector and insert initial guess.
    Eigen::VectorXd variables;
    variables.setZero(2);
    variables(0) = initial_guess_a;
    variables(1) = initial_guess_b;
    
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