#include <vector>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <chrono>
#include <iomanip>

// constant params
namespace data
{
    constexpr double T{0.1}, dx{1.0 / 512}, dy{1.0 / 512}, dt{8e-7}, alpha{1};
    constexpr int xMin{0}, xMax{1}, yMin{0}, yMax{1};
    const int numTimeSteps{static_cast<int>(T / dt)}, numXSteps{static_cast<int>((xMax - xMin) / dx) + 1}, numYSteps{static_cast<int>((yMax - yMin) / dy) + 1};
    const int Nx_int{numXSteps - 2}, Ny_int{numYSteps - 2};
    const double dtOverDxSq = data::dt / (data::dx * data::dx);
    const double dtOverDySq = data::dt / (data::dy * data::dy);
}

// u(x, y, 0) = sin(pi* x) * sin(pi * y)
double timeLowerBoundary(double x, double y)
{
    return sin(M_PI * (x * data::dx + data::xMin)) * sin(M_PI * (y * data::dy + data::yMin));
}

template <typename T>
std::ostream &operator<<(std::ostream &os, std::vector<std::vector<T>> &vect)
{
    for (int i = 0; i < vect.size(); i++)
    {
        for (int j = 0; j < vect[i].size(); j++)
        {
            os << std::setw(10) << std::fixed << std::setprecision(6) << vect[i][j] << ", ";
        }
        os << std::endl;
    }
    return os;
}

// return (np.e ** (-2 * (np.pi**2) * t)) * np.sin(np.pi*x * dx) * np.sin(np.pi*y * dy)
double calculateExactSol(int i, int j)
{
    return std::exp(-2 * (M_PI * M_PI) * data::T) * std::sin(M_PI * (i * data::dx + data::xMin)) * std::sin(M_PI * (j * data::dy + data::yMin));
}
// L^2 error
double computeError(const std::vector<std::vector<double>> &uNumeric)
{
    double ans = 0;
    for (int i = 0; i < data::numXSteps; i++)
    {
        for (int j = 0; j < data::numYSteps; j++)
        {
            double diff = calculateExactSol(i, j) - uNumeric[i][j];
            ans += (diff * diff) * data::dx * data::dy;
        }
    }
    return std::sqrt(ans);
}

// solves heat equation using explicit method
// solution = u(x, y, t)
inline void explicitBaseline()
{
    // start time
    auto t0 = std::chrono::steady_clock::now();
    // u(x, y, m*dt)
    std::vector<std::vector<double>> uXyPrev(data::numXSteps, std::vector<double>(data::numYSteps, 0));
    // set initial t=0 boundary conditions
    for (int n = 0; n < data::numXSteps; n++)
    {
        for (int v = 0; v < data::numYSteps; v++)
        {
            if (n == 0 || v == 0 || n == data::numXSteps - 1 || v == data::numYSteps - 1)
            {
                uXyPrev[n][v] = 0;
            }
            else
            {
                uXyPrev[n][v] = timeLowerBoundary(n, v);
            }
        }
    }
    // u(x, y, m+1*dt)
    std::vector<std::vector<double>> uXyCurr(data::numXSteps, std::vector<double>(data::numYSteps, 0));
    for (int m = 1; m <= data::numTimeSteps; m++)
    {
        for (int n = 1; n < data::numXSteps - 1; n++)
        {
            for (int v = 1; v < data::numYSteps - 1; v++)
            {
                // update according to explicit method
                uXyCurr[n][v] = (data::alpha * data::dtOverDxSq) * (uXyPrev[n + 1][v] + uXyPrev[n - 1][v]) +
                                uXyPrev[n][v] * (1 - 2 * data::alpha * data::dtOverDxSq - 2 * data::alpha * data::dtOverDySq) +
                                data::alpha * data::dtOverDySq * (uXyPrev[n][v + 1] + uXyPrev[n][v - 1]);
            }
        }
        // keep both vectors in. avalid state and set u(x, y, m) = u(x, y, m+1) for next iteration
        std::swap(uXyCurr, uXyPrev);
    }
    // end time point
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "baseline time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << std::endl;
    std::cout << "baseline L^2 error: " << computeError(uXyPrev) << std::endl;
}

int main()
{
    explicitBaseline();
    return 0;
}