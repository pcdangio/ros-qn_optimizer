#include "qn_optimizer/qn_optimizer.h"

#include <iostream>

double objective_function(const Eigen::VectorXd& variables)
{
    //return std::sin(1 + variables(0));// - 0.00001/(variables(0) - 6.0) - 0.00001/(2.0-variables(0));
    return std::pow(variables(0) + 4.75, 2.0) + std::pow(variables(1) - 2, 2.0);
}
void objective_gradient(const Eigen::VectorXd& operating_point, Eigen::VectorXd& gradient)
{
    gradient(0) = std::cos(1+operating_point(0));
    // gradient(0) = 2*(operating_point(0) - 4.75);
    // gradient(1) = 2*(operating_point(1) - 2);
}

int32_t main(int32_t argc, char** argv)
{
    qn_optimizer qno(2, &objective_function);

    Eigen::VectorXd variables;
    variables.setZero(2);
    double score;
    qno.optimize(variables, &score);

    std::cout << "minimized value: " << std::endl << variables << std::endl;

    auto iterations = qno.iterations();
    std::cout << "total iterations: " << iterations.size() << std::endl;
    for(auto iteration = iterations.cbegin(); iteration != iterations.cend(); ++iteration)
    {
        std::cout << *iteration << std::endl;
    }
}