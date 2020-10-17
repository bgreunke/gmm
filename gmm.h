#ifndef _GMM_H
#define _GMM_H

#include <random>
#include <cstring>
#include <cmath>
#include <limits>

class GaussianMixture {
public:
    /* Constructor
     * PARAMS:
     * d - the dimension of the distribution
     * m - the number of Gaussian components in the mixture
     * alpha_init, mu_init, Sigma_init - the initial values for the 
     *                                   distribution parameters
     */
    GaussianMixture(int d, int m, double* alpha_init, double* mu_init, double* Sigma_init);
    GaussianMixture(int d, int m) : GaussianMixture(d, m, nullptr, nullptr, nullptr) {}
    ~GaussianMixture();

    // Computes the log density at x
    double log_prob(double* x);
    /* Samples n points from the distribution from the random number generator
     * g and stores them in out
     * PARAMS:
     * g - a uniform random number generator ie. any of the generators in
     *     <random>
     * n - the number of points to sample
     * out - a buffer in which to store the sampled points
     */
    template<class URNG>
    void sample(URNG &g, int n, double* out);
    /* Both of these functions compute the maximum likelihood distribution
     * parameters given a set of points x by the expectation maximization
     * algorithm.
     * PARAMS:
     * x - the set of points of dimension d
     * n - the number of points in x
     * max_iters - the maximum number of iterations of the EM algorithm
     * eps - a threshold for convergence of the algorithm; termination occurs
     *       when the difference between the log likelihood of one step and the
     *       previous step is below this value
     * alpha_init, mu_init, Sigma_init - initial conditions for the algorithm
     */
    void MLE(double* x, int n, int max_iters, double eps, double* alpha_init, double* mu_init, double* Sigma_init);
    void MLE(double* x, int n, int max_iters, double eps) {this->MLE(x, n, max_iters, eps, nullptr, nullptr, nullptr);}

private:
    std::allocator<double> alloc;

    int m;
    int d;

    double* alpha;
    double* mu;
    double* Sigma;
};

extern "C" {
extern void dpotrf_(const char*,int*,double*,int*,int*);
extern void dpotrs_(const char*,int*,int*,double*,int*,double*,int*,int*);
extern double ddot_(int*,double*,int*,double*,int*);
extern void dtrmv_(const char*,const char*,const char*,int*,double*,int*,double*,int*);
extern void daxpy_(int*,double*,double*,int*,double*,int*);
extern double dasum_(int*,double*,int*);
extern void dsyr_(const char*,int*,double*,double*,int*,double*,int*);
extern void dscal_(int*,double*,double*,int*);
}

GaussianMixture::GaussianMixture(
    int d, int m, double* alpha_init, double* mu_init, double* Sigma_init
) {
    this->m = m;
    this->d = d;

    this->alloc = std::allocator<double>();

    this->alpha = this->alloc.allocate(m);
    this->mu = this->alloc.allocate(m*d);
    this->Sigma = this->alloc.allocate(m*d*d);

    // If the initial values are null, initialize the parameters with default
    // values alpha_i = 1/m, all mu are 0 vectors and all Sigma are the
    // identity covariance matrix
    if (alpha_init != nullptr) {
        std::memcpy(this->alpha, alpha_init, m*sizeof(double));
    }
    else {
        double alpha = 1.0 / (double) m;
        for (int i = 0; i < m; i++) {
            this->alpha[i] = alpha;
        }
    }

    if (mu_init != nullptr) {
        std::memcpy(this->mu, mu_init, m*d*sizeof(double));
    }
    else if (m <= 1) {
        for (int i = 0; i < d; i++) {
            mu[i] = 0.0;
        }
    }
    else {
        for (int k = 0; k < m; k++) {
            for (int i = 0; i < d; i++) {
                mu[k*d +i] = 0.0;
            }
        }
    }

    if (Sigma_init != nullptr) {
        std::memcpy(this->Sigma, Sigma_init, m*d*d*sizeof(double));
    }
    else {
        for (int k = 0; k < m; k++) {
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    this->Sigma[k*d*d + j*d + i] = (i == j) ? 1 : 0;
                }
            }
        }
    }
}

GaussianMixture::~GaussianMixture() {
    this->alloc.deallocate(this->alpha, m);
    this->alloc.deallocate(this->mu, m*d);
    this->alloc.deallocate(this->Sigma, m*d*d);
}

double GaussianMixture::log_prob(double* x) {
    int m = this->m;
    int d = this->d;

    double* work_mem = this->alloc.allocate(d*d + 2*d + m);

    double* cholesky = work_mem;
    double* center_x = &work_mem[d*d];
    double* inv_center_x = &work_mem[d*d + d];
    double* log_probs = &work_mem[d*d + 2*d];

    int one = 1;
    double log_coeff = -(((double)d)/2.0) * std::log(M_PI);

    for (int k = 0; k < m; k++) {
        for (int i = 0; i < d; i++) {
            center_x[i] = x[i] - this->mu[k*d + i];
        }

        std::memcpy(inv_center_x, center_x, d*sizeof(double));
        std::memcpy(cholesky, &this->Sigma[k*d*d], d*d*sizeof(double));

        int info;
        dpotrf_(
            "U",
            &d,
            cholesky,
            &d,
            &info
        );

        dpotrs_(
            "U",
            &d,
            &one,
            &cholesky[k*d*d],
            &d,
            inv_center_x,
            &one,
            &info
        );

        double expnt = ddot_(
            &d, 
            center_x,
            &one,
            inv_center_x,
            &one
        );

        double log_det = 0.0;
        for (int i = 0; i < d; i++) {
            log_det += std::log(cholesky[i*d + i]);
        }

        log_probs[k] = std::log(this->alpha[k]) + log_coeff 
                        - log_det - 0.5*expnt;
    }

    double prob = 0.0;
    for (int k = 0; k < d; k++) {
        prob += std::exp(log_probs[k]);
    }

    this->alloc.deallocate(work_mem, d*d + 2*d + m);

    return std::log(prob);
}

template<class URNG>
void GaussianMixture::sample(URNG &g, int n, double* out) {
    int m = this->m;
    int d = this->d;

    std::discrete_distribution<int> ag(&this->alpha[0], &this->alpha[m]);
    std::normal_distribution<double> bg;

    for (int i = 0; i < n*d; i++) {
        out[i] = bg(g);
    }

    double* cholesky = this->alloc.allocate(m*d*d);
    std::memcpy(cholesky, this->Sigma, m*d*d*sizeof(double));
    for (int k = 0; k < m; k++) {
        int info;
        dpotrf_(
            "U",
            &d,
            &cholesky[k*d*d],
            &d,
            &info
        );
    }

    double done = 1.0;
    int zone = 1;
    for (int i = 0; i < n; i++) {
        int j = ag(g);

        dtrmv_(
            "U",
            "N",
            "N",
            &d,
            &cholesky[j*d*d],
            &d,
            &out[i*d],
            &zone
        );

        daxpy_(
            &d,
            &done,
            &this->mu[j*d],
            &zone,
            &out[i*d],
            &zone
        );
    }
}

void GaussianMixture::MLE(
    double* x, 
    int n, 
    int max_iters, 
    double eps, 
    double* alpha_init, 
    double* mu_init, 
    double* Sigma_init
) {
    int d = this->d;
    int m = this->m;

    size_t work_size = m*d*d + m*d + m + 2*n*d + 2*m*n + n + m;
    size_t work_offset = 0;
    double* work_mem = this->alloc.allocate(work_size);

    double* work_alpha = &work_mem[work_offset];
    work_offset += m;
    double* work_mu = &work_mem[work_offset];
    work_offset += m*d;
    double* work_Sigma = &work_mem[work_offset];
    work_offset += m*d*d;

    double* work_x1 = &work_mem[work_offset];
    work_offset += n*d;
    double* work_x2 = &work_mem[work_offset];
    work_offset += n*d;

    double* probs = &work_mem[work_offset];
    work_offset += n;
    double* sub_probs = &work_mem[work_offset];
    work_offset += m*n;
    double* coeffs = &work_mem[work_offset];
    work_offset += m*n;
    double* coeffs_sum = &work_mem[work_offset]; 
    work_offset += n;

    double log_scale = -(((double)d)/2.0) * std::log(M_PI);
    double ll_curr, ll_prev;
    ll_curr = std::numeric_limits<double>::has_infinity ? 
        -std::numeric_limits<double>::infinity() : 
        -std::numeric_limits<double>::max();

    int zone = 1;
    int dxd = d*d;

    std::memcpy(
        work_alpha, 
        (alpha_init == nullptr) ? this->alpha : alpha_init, 
        m*sizeof(double)
    );

    std::memcpy(
        work_mu, 
        (mu_init == nullptr) ? this->mu : mu_init, 
        m*d*sizeof(double)
    );

    std::memcpy(
        work_Sigma, 
        (Sigma_init == nullptr) ? this->Sigma : Sigma_init,
        m*d*d*sizeof(double)
    );

    for (int it = 0; it < max_iters; it++) {
        // The expectation step of the EM algorithm
        for (int k = 0; k < m; k++) {
            int info;
            dpotrf_(
                "U",
                &d,
                &work_Sigma[k*d*d],
                &d,
                &info
            );

            double log_det = 0.0;
            for (int i = 0; i < d; i++) {
                log_det += std::log(work_Sigma[k*d*d + i*d + i]);
            }

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < d; j++) {
                    work_x1[i*d + j] = x[i*d + j] - work_mu[k*d + j];
                }
            }
            std::memcpy(work_x2, work_x1, n*d*sizeof(double));

            dpotrs_(
                "U",
                &d,
                &n,
                &work_Sigma[k*d*d],
                &d,
                work_x2,
                &d,
                &info
            );

            for (int i = 0; i < n; i++) {
                double expnt = ddot_(
                    &d,
                    &work_x1[i*d],
                    &zone,
                    &work_x2[i*d],
                    &zone
                );

                sub_probs[k*n + i] = work_alpha[k] * 
                    std::exp(log_scale - log_det - 0.5*expnt);
            }
        }

        // The maximization step of the EM algorithm
        for (int i = 0; i < n; i++)
            probs[i] = 0.0;

        for (int j = 0; j < m; j++) {
            for (int i = 0; i < n; i++) {
                probs[i] += sub_probs[j*n + i];
            }
        }

        for (int j = 0; j < m; j++) {
            for (int i = 0; i < n; i++) {
                coeffs[j*n + i] = sub_probs[j*n + i]/probs[i];
            }
        }

        for (int i = 0; i < m; i++) {
            coeffs_sum[i] = dasum_(
                &n,
                &coeffs[i*n],
                &zone
            );
            work_alpha[i] = coeffs_sum[i] / (double) n;
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < d; j++) {
                work_mu[i*d + j] = ddot_(
                    &n,
                    &x[j],
                    &d,
                    &coeffs[i*n],
                    &zone
                ) / coeffs_sum[i];
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < d; k++) {
                    work_x1[j*d + k] = x[j*d + k] - work_mu[i*d + k];
                }
            }

            for(int j = 0; j < d; j++) {
                for (int k = 0; k < d; k++) {
                    work_Sigma[i*d*d + k*d + j] = 0.0;
                }
            }

            for (int j = 0; j < n; j++) {
                dsyr_(
                    "U",
                    &d,
                    &coeffs[i*n + j],
                    &work_x1[j*d],
                    &zone,
                    &work_Sigma[i*d*d],
                    &d
                );
            }

            double scale = 1.0/coeffs_sum[i];
            dscal_(
                &dxd,
                &scale,
                &work_Sigma[i*d*d],
                &zone
            );
        }

        ll_prev = ll_curr;
        ll_curr = 0.0;
        for (int i = 0; i < n; i++) {
            ll_curr += std::log(probs[i]);
        }

        if (ll_curr - ll_prev < eps)
            break;

#ifdef PRINT_ITERS
        printf("Iteration %d\n", it);
        for (int k1 = 0; k1 < m; k1++) {
            printf("alpha[%d] = %f\n", k1, work_alpha[k1]);

            printf("mu[%d] = \n", k1);
            for (int i1 = 0; i1 < d; i1++) {
                printf("%f ", work_mu[k1*d + i1]);
            }
            printf("\n");

            printf("Sigma[%d] = \n", k1);
            for (int i1 = 0; i1 < d; i1++) {
                for (int j1 = 0; j1 < d; j1++) {
                    printf("%f ", work_Sigma[k1*d*d + j1*d + i1]);
                }
                printf("\n");
            }
            printf("\n");
        }
#endif
    }

    std::memcpy(this->alpha, work_alpha, m*sizeof(double));
    std::memcpy(this->mu, work_mu, m*d*sizeof(double));
    std::memcpy(this->Sigma, work_Sigma, m*d*d*sizeof(double));

    this->alloc.deallocate(work_mem, m*d*d+m*d+m+2*n*d+2*m*n+n+m);
}

#endif
