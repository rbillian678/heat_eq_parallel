#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <omp.h>

// holds params
namespace data
{
    constexpr double T{0.1}, dx{1.0 / 512}, dy{1.0 / 512}, dt{1.5e-5}, alpha{1};
    constexpr int xMin{0}, xMax{1}, yMin{0}, yMax{1};
    const int numTimeSteps{static_cast<int>(T / dt)}, numXSteps{static_cast<int>((xMax - xMin) / dx) + 1}, numYSteps{static_cast<int>((yMax - yMin) / dy) + 1};
    const int Nx_int{numXSteps - 2}, Ny_int{numYSteps - 2};
    constexpr double lambda_x = -data::alpha * data::dt / (data::dx * data::dx);
    constexpr double lambda_y = -data::alpha * data::dt / (data::dy * data::dy);
    constexpr double a = 1 - 2 * data::lambda_x - 2 * data::lambda_y;
}

// u(x, y, 0) = sin(pi* x) * sin(pi * y)
double timeLowerBoundary(double x, double y)
{
    return sin(M_PI * (x * data::dx + data::xMin)) * sin(M_PI * (y * data::dy + data::yMin));
}
// compute Ax without building matrix
void applyAMatrix(const std::vector<double> &x,
                  std::vector<double> &y)
{
    const int Nx = data::Nx_int;
    const int Ny = data::Ny_int;

#pragma omp parallel for
    for (int j = 0; j < Ny; ++j)
    {
        const int row = j * Nx;
        for (int i = 0; i < Nx; ++i)
        {
            double sum = data::a * x[row + i];
            if (i > 0)
                sum += data::lambda_x * x[row + (i - 1)];
            if (i + 1 < Nx)
                sum += data::lambda_x * x[row + (i + 1)];
            if (j > 0)
                sum += data::lambda_y * x[(j - 1) * Nx + i];
            if (j + 1 < Ny)
                sum += data::lambda_y * x[(j + 1) * Nx + i];
            y[row + i] = sum;
        }
    }
}
// ans[i] = sum(x[i] * y[i]), for all i < n = x.size()
double dotProduct(const std::vector<double> &a, const std::vector<double> &b)
{
    double result = 0;
#pragma omp simd reduction(+ : result)
    for (int i = 0; i < a.size(); i++)
    {
        result += a[i] * b[i];
    }
    return result;
}
// ans[i] = x[i] * y[i], for all i < n = x.size()
void vectorMultiply(double a, const std::vector<double> &x, std::vector<double> &ans)
{
#pragma omp simd
    for (int i = 0; i < x.size(); i++)
    {
        ans[i] = a * x[i];
    }
}
// ans[i] = x[i] + y[i], for all i < n = x.size()
void vectorAdd(const std::vector<double> &x, const std::vector<double> &y, std::vector<double> &ans)
{
#pragma omp simd
    for (int i = 0; i < x.size(); i++)
    {
        ans[i] = x[i] + y[i];
    }
}
// sqrt(x_0^2 + x_1^2 + ... x_n^2)
double vectorNorm(const std::vector<double> &x)
{
    double sum = 0;
#pragma omp simd reduction(+ : sum)
    for (auto num : x)
    {
        sum += num * num;
    }
    return std::sqrt(sum);
}
// printing grid layers for debugging
template <typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> &vect)
{
    if (vect.size() == data::numXSteps * data::numYSteps)
    {
        for (int i = 0; i < data::numYSteps; i++)
        {
            for (int j = 0; j < data::numXSteps; j++)
            {
                os << std::setw(10) << std::fixed << std::setprecision(6) << vect[i * data::numXSteps + j] << ", ";
            }
            os << std::endl;
        }
    }
    else
    {
        for (int i = 0; i < data::Ny_int; i++)
        {
            for (int j = 0; j < data::Nx_int; j++)
            {
                os << std::setw(10) << std::fixed << std::setprecision(6) << vect[i * data::Nx_int + j] << ", ";
            }
            os << std::endl;
        }
    }
    return os;
}

// closed form solution
double calculateExactSol(int i, int j)
{
    // return (np.e ** (-2 * (np.pi**2) * t)) * np.sin(np.pi*x * dx) * np.sin(np.pi*y * dy)
    return std::exp(-2 * (M_PI * M_PI) * data::T) * std::sin(M_PI * i * data::dx) * std::sin(M_PI * j * data::dy);
}
// L^2 error
double computeError(const std::vector<double> &uNumeric)
{
    double ans = 0;
#pragma omp parallel for reduction(+ : ans)
    for (int i = 0; i < data::numXSteps; i++)
    {
#pragma omp simd
        for (int j = 0; j < data::numYSteps; j++)
        {
            double diff = calculateExactSol(i, j) - uNumeric[i * data::numXSteps + j];
            ans += (diff * diff) * data::dx * data::dy;
        }
    }
    return std::sqrt(ans);
}
// t=0 initialization
void initializeInitialGrid(std::vector<double> &uXyPrev)
{
#pragma omp parallel for
    for (int y = 0; y < data::numYSteps; y++)
    {
        for (int x = 0; x < data::numXSteps; x++)
        {
            if (y == 0 || x == 0 || x == data::numXSteps - 1 || y == data::numYSteps - 1)
            {
                uXyPrev[y * data::numXSteps + x] = 0;
            }
            else
            {
                uXyPrev[y * data::numXSteps + x] = timeLowerBoundary(x, y);
            }
        }
    }
}

void implicitOmp()
{
    // get start time
    auto t0 = std::chrono::steady_clock::now();
    // solution at time t-1
    std::vector<double> uXyPrev(data::numXSteps * data::numYSteps, 0);
    // set t=0 grid based on boundary behavior
    initializeInitialGrid(uXyPrev);
    // for Conjugate Gradient algo (CG)
    int maxIters = data::Nx_int * data::Ny_int;
    std::vector<double> rkPrev(data::Nx_int * data::Ny_int);
    std::vector<double> rk(rkPrev.size(), 0), bkTimesPk(rkPrev.size()), intermediateAPk(rkPrev.size(), 0);
    std::vector<double> xk(rkPrev.size(), 0), alphaKTimesPk(rkPrev.size(), 0), minusAlphakTimesInter(rkPrev.size(), 0);
    std::vector<double> pk(rkPrev.size(), 0);
    // outer loop from t=1 -> T
    for (int m = 1; m < data::numTimeSteps; m++)
    {
        // split the grid into pieces for each thread to process independently, copying t=0 boundary behavior to rkPrev
        // which does not include boundary
#pragma omp parallel for
        for (int i = 1; i < data::numYSteps - 1; i++)
        {
            for (int j = 1; j < data::numXSteps - 1; j++)
            {
                rkPrev[(i - 1) * data::Nx_int + (j - 1)] = uXyPrev[i * data::numXSteps + j];
            }
        }

        // initialize xk to 0
        std::fill(xk.begin(), xk.end(), 0);
        // pk starts at initial residual
        pk = rkPrev;

        double error = 1;
        int k = 0;
        // norm stays constant over entire t=m*dt
        double rkPrevDot = dotProduct(rkPrev, rkPrev);
        double bNorm = std::sqrt(rkPrevDot);

        // vect_a * vect_b == vect_a[i] * vect_b[i] for all i < vect_a.size() && vect_b.size()
        // solves Ax = b, A * U_m+1= U_m
        // Note: CG iteration count can be reduced significantly with a Jacobi (diagonal) preconditioner.
        while (error > 1e-3 && k < maxIters)
        {
            // intermediateAPk = A*pk
            applyAMatrix(pk, intermediateAPk);

            // alphaK = rkPrevDot / dot(pk, A*pk)
            double alphaK = rkPrevDot / dotProduct(pk, intermediateAPk);
            // alphaKTimesPk = alphaK * pk
            vectorMultiply(alphaK, pk, alphaKTimesPk);
            // xk = (rkPrevDot * pk) / dot(pk, A*pk) + xk
            vectorAdd(xk, alphaKTimesPk, xk);
            // minusAlphakTimesInter = -(rkPrevDot * A * pk) / dot(pk, Apk)
            vectorMultiply(-alphaK, intermediateAPk, minusAlphakTimesInter);
            // rk = -(rkPrevDot * Apk) / dot(pk, Apk) + rkPrev
            vectorAdd(rkPrev, minusAlphakTimesInter, rk);
            double rkDot = dotProduct(rk, rk);
            double bk = rkDot / rkPrevDot;
            vectorMultiply(bk, pk, bkTimesPk);
            // pk = rk + bk * pk
            vectorAdd(rk, bkTimesPk, pk);
            // swap O(1) and both vectors remain valid in state
            std::swap(rk, rkPrev);
            rkPrevDot = rkDot;
            // norm(rk) / norm(b)
            error = std::sqrt(rkDot) / bNorm;
            k++;
        }
        // copies solution to UxYPrev for interioir points
        for (int i = 1; i < data::numYSteps - 1; i++)
        {
            for (int j = 1; j < data::numXSteps - 1; j++)
            {
                uXyPrev[i * data::numXSteps + j] = xk[(i - 1) * data::Nx_int + (j - 1)];
            }
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "Implicit Omp time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << std::endl;
    std::cout << "Implicit Omp L^2 error: " << computeError(uXyPrev) << std::endl;
}

int main()
{
    implicitOmp();

    return 0;
}