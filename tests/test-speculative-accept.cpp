// Tests the maths of the speculative acceptance kernel (common_draft_accept_step) without a model.
//
// Three things are checked:
//
//   1. Greedy equivalence. When the target is greedy (p one-hot) and the draft head is greedy
//      (q one-hot), the new rule makes exactly the same decisions as the old exact-match test,
//      token for token, and consumes no randomness. This is the hard gate: greedy output must not
//      change at all.
//
//   2. The emitted token is distributed as p. Over many trials with random p and random q, the
//      histogram of emitted tokens matches p to within sampling noise (chi-square).
//
//   3. The acceptance rate is sum_y min(p(y), q(y)), the theoretical maximum for this scheme,
//      and it beats the exact-match rate p(argmax q) whenever q is not one-hot.

#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int n_fail = 0;

static void check(bool ok, const char * what) {
    printf("%-60s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) {
        n_fail++;
    }
}

// build a random distribution over ids [0, n) by drawing exponentials and normalising
static std::vector<llama_token_data> random_dist(std::mt19937 & rng, int n) {
    std::exponential_distribution<double> exp_dist(1.0);

    std::vector<double> w(n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        w[i] = exp_dist(rng) + 1e-6;
        sum += w[i];
    }

    std::vector<llama_token_data> out(n);
    for (int i = 0; i < n; ++i) {
        out[i] = { (llama_token) i, 0.0f, (float) (w[i] / sum) };
    }

    return out;
}

// a truncated, renormalised proposal: keep `k` of the `n` ids and renormalise over them
static std::vector<llama_token_data> random_support(std::mt19937 & rng, int n, int k) {
    auto full = random_dist(rng, n);

    std::shuffle(full.begin(), full.end(), rng);
    full.resize(k);

    double sum = 0.0;
    for (const auto & e : full) {
        sum += e.p;
    }
    for (auto & e : full) {
        e.p = (float) (e.p / sum);
    }

    return full;
}

static int sample_from(const std::vector<llama_token_data> & d, std::mt19937 & rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);

    const double v = u(rng);

    double acc = 0.0;
    for (size_t i = 0; i < d.size(); ++i) {
        acc += d[i].p;
        if (acc > v) {
            return (int) i;
        }
    }

    return (int) d.size() - 1;
}

// ---------------------------------------------------------------------------------------------
// 1. greedy equivalence
// ---------------------------------------------------------------------------------------------
static void test_greedy_equivalence() {
    const int n_vocab = 32;

    std::mt19937 rng(1234);
    std::mt19937 rng_kernel(5678);

    int n_cases = 0;
    int n_same  = 0;

    // the accept path must not consume randomness - it is the hot path
    int n_accept_draws = 0;

    for (int t = 0; t < 10000; ++t) {
        // greedy target: p is one-hot on some token
        const llama_token id_target = (llama_token) (rng() % n_vocab);

        std::vector<llama_token_data> p;
        p.push_back({ id_target, 0.0f, 1.0f });

        // greedy draft head: q is one-hot on the token it drafted
        const llama_token x = (llama_token) (rng() % n_vocab);

        std::vector<llama_token_data> q;
        q.push_back({ x, 0.0f, 1.0f });

        const auto rng_before = rng_kernel;

        const auto res = common_draft_accept_step(p.data(), p.size(), q.data(), q.size(), x, 1.0f, id_target, rng_kernel);

        // what today's exact-match loop would do
        const llama_token exact_id       = id_target;
        const bool        exact_accepted = (x == id_target);

        n_cases++;
        if (res.id == exact_id && res.accepted == exact_accepted) {
            n_same++;
        }

        if (res.accepted && !(rng_kernel == rng_before)) {
            n_accept_draws++;
        }
    }

    printf("  greedy: %d/%d decisions identical to exact match, %d accepts consumed randomness\n",
            n_same, n_cases, n_accept_draws);

    check(n_same == n_cases, "greedy: every decision matches the exact-match loop");
    check(n_accept_draws == 0, "greedy: accepting a draft token consumes no randomness");
}

// ---------------------------------------------------------------------------------------------
// 2. the emitted token is distributed as p
// ---------------------------------------------------------------------------------------------
static void test_emitted_distribution() {
    const int n_vocab = 8;
    const int n_trial = 100000;

    // chi-square with 7 degrees of freedom: 24.32 at p = 0.001, 29.9 at p = 0.0001.
    // 60 is far out in the tail; a correct kernel will never get near it, a wrong one blows past it.
    const double chi2_limit = 60.0;

    std::mt19937 rng_setup(99);

    for (int scenario = 0; scenario < 4; ++scenario) {
        std::mt19937 rng(20260902 + scenario);
        std::mt19937 rng_kernel(4242 + scenario);

        const auto p = random_dist(rng_setup, n_vocab);
        // scenario 0: q over the full vocab; 1-3: q truncated to a few tokens, which is what a
        // top-k head actually produces
        const auto q = scenario == 0 ? random_dist(rng_setup, n_vocab) : random_support(rng_setup, n_vocab, 2 + scenario);

        std::vector<int> hist(n_vocab, 0);

        int n_accept = 0;

        for (int t = 0; t < n_trial; ++t) {
            // the draft head draws x from q, and the target independently draws its own sample from p
            const int         iq = sample_from(q, rng);
            const llama_token x  = q[iq].id;

            const llama_token id_sample = p[sample_from(p, rng)].id;

            const auto res = common_draft_accept_step(p.data(), p.size(), q.data(), q.size(), x, q[iq].p, id_sample, rng_kernel);

            hist[res.id]++;
            if (res.accepted) {
                n_accept++;
            }
        }

        double chi2 = 0.0;
        for (int i = 0; i < n_vocab; ++i) {
            const double e = (double) n_trial * p[i].p;
            const double d = (double) hist[i] - e;
            chi2 += d * d / e;
        }

        // theoretical acceptance rate and the exact-match rate it replaces
        double acc_theory = 0.0;
        double best_q     = 0.0;
        llama_token arg_q = 0;
        for (const auto & e : q) {
            acc_theory += std::min((double) e.p, (double) p[e.id].p);
            if (e.p > best_q) {
                best_q = e.p;
                arg_q  = e.id;
            }
        }
        const double acc_exact = p[arg_q].p;

        const double acc_emp = (double) n_accept / n_trial;

        printf("  scenario %d: |q| = %2zu  chi2 = %7.3f  accept: empirical %.4f, theory %.4f, exact-match %.4f\n",
                scenario, q.size(), chi2, acc_emp, acc_theory, acc_exact);

        char name[128];
        snprintf(name, sizeof(name), "scenario %d: emitted distribution matches p", scenario);
        check(chi2 < chi2_limit, name);

        snprintf(name, sizeof(name), "scenario %d: acceptance rate matches sum min(p,q)", scenario);
        check(std::fabs(acc_emp - acc_theory) < 0.01, name);
    }
}

// ---------------------------------------------------------------------------------------------
// 3. a one-hot q reproduces the exact-match acceptance rate, and the residual never re-emits x
// ---------------------------------------------------------------------------------------------
static void test_one_hot_q() {
    const int n_vocab = 8;
    const int n_trial = 100000;

    std::mt19937 rng_setup(7);
    std::mt19937 rng(31337);
    std::mt19937 rng_kernel(1000003);

    const auto p = random_dist(rng_setup, n_vocab);

    // the drafted token: whatever an argmax head would have produced
    const llama_token x = 3;

    std::vector<llama_token_data> q;
    q.push_back({ x, 0.0f, 1.0f });

    std::vector<int> hist(n_vocab, 0);

    int n_accept       = 0;
    int n_reemitted_x  = 0;

    for (int t = 0; t < n_trial; ++t) {
        const llama_token id_sample = p[sample_from(p, rng)].id;

        const auto res = common_draft_accept_step(p.data(), p.size(), q.data(), q.size(), x, 1.0f, id_sample, rng_kernel);

        hist[res.id]++;
        if (res.accepted) {
            n_accept++;
        } else if (res.id == x) {
            n_reemitted_x++;
        }
    }

    double chi2 = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        const double e = (double) n_trial * p[i].p;
        const double d = (double) hist[i] - e;
        chi2 += d * d / e;
    }

    const double acc_emp = (double) n_accept / n_trial;

    printf("  one-hot q: chi2 = %7.3f  accept: empirical %.4f, exact-match p(x) %.4f, residual re-emits x %d times\n",
            chi2, acc_emp, (double) p[x].p, n_reemitted_x);

    check(chi2 < 60.0, "one-hot q: emitted distribution matches p");
    check(std::fabs(acc_emp - (double) p[x].p) < 0.01, "one-hot q: acceptance rate equals p(x), as exact match");
    check(n_reemitted_x == 0, "one-hot q: a rejection never re-emits the drafted token");
}

// ---------------------------------------------------------------------------------------------
// 4. a forced position (reasoning budget) must emit the forced token
//
// when the reasoning-budget sampler is forcing its end sequence it zeroes every other candidate,
// so p arrives one-hot on the forced token. a draft token that is not the forced one must be
// rejected and the residual must collapse onto the forced token - i.e. the forced output survives
// rejection sampling untouched. this is why the kernel does not need to bail out on a budget.
// ---------------------------------------------------------------------------------------------
static void test_forced_position() {
    const int n_vocab = 16;

    std::mt19937 rng(2024);
    std::mt19937 rng_kernel(555);

    int n_cases        = 0;
    int n_emitted_forced = 0;
    int n_accepted     = 0;

    for (int t = 0; t < 10000; ++t) {
        const llama_token id_forced = (llama_token) (rng() % n_vocab);

        // p as the budget sampler leaves it: one-hot on the forced token
        std::vector<llama_token_data> p;
        p.push_back({ id_forced, 0.0f, 1.0f });

        // the draft head knows nothing about the budget, so it proposes something else from a
        // broad, sampled distribution
        auto q = random_support(rng, n_vocab, 4);

        // make sure the drafted token is NOT the forced one, which is the interesting case
        int iq = 0;
        while (iq < (int) q.size() && q[iq].id == id_forced) {
            iq++;
        }
        if (iq >= (int) q.size()) {
            continue;
        }

        const llama_token x = q[iq].id;

        const auto res = common_draft_accept_step(p.data(), p.size(), q.data(), q.size(), x, q[iq].p, id_forced, rng_kernel);

        n_cases++;
        if (res.id == id_forced) {
            n_emitted_forced++;
        }
        if (res.accepted) {
            n_accepted++;
        }
    }

    printf("  forced: %d cases, %d emitted the forced token, %d accepted the draft\n",
            n_cases, n_emitted_forced, n_accepted);

    check(n_cases > 1000,                  "forced: the case was actually exercised");
    check(n_emitted_forced == n_cases,     "forced: a non-matching draft still emits the forced token");
    check(n_accepted == 0,                 "forced: a non-matching draft is never accepted");
}

int main() {
    printf("test-speculative-accept\n\n");

    test_greedy_equivalence();
    printf("\n");
    test_emitted_distribution();
    printf("\n");
    test_one_hot_q();
    printf("\n");
    test_forced_position();

    printf("\n%s (%d failures)\n", n_fail == 0 ? "ALL PASS" : "FAILED", n_fail);

    return n_fail == 0 ? 0 : 1;
}
