import random


def sample_from_distribution(distribution, lo, hi, n):
    sample_prob = 1.0
    r0 = random.random()
    s = 0
    j = 0

    for i in range(n):
        s = s + distribution[i]
        if s >= r0:
            sample_prob = distribution[i]
            j = i
            break
    assert (j >= 0)
    assert (j <= n)
    r1 = random.random()
    delta = (hi - lo) / n
    sample_value = lo + (j + r1 - 1) * delta
    return sample_value, sample_prob


def choose_bernoulli_samples(input_ranges, sub_divs, distribution):
    no_inputs = len(input_ranges)
    sample = [0] * no_inputs
    for i in range(no_inputs):
        n = sub_divs[i]
        sample[i], sample_prob = sample_from_distribution(distribution[i], input_ranges[i][0], input_ranges[i][1], n)
    return sample


def stochastic_optimizer(function_min, cp_samples, *args, options):
    samples_per_round = options[0]
    n = options[1]
    no_inputs = len(cp_samples)
    distribution = [[1 / n for i in range(n)] for j in range(no_inputs)]
    sub_divs = [n] * len(cp_samples)
    inp_ranges = [cp_samples] * len(cp_samples)
    all_samples = []
    all_rob = []
    for i in range(samples_per_round):
        cur_sample = choose_bernoulli_samples(inp_ranges, sub_divs, distribution)
        robustness = function_min(cur_sample, *args)
        all_samples.append(cur_sample)
        all_rob.append(robustness)

    return all_samples, all_rob


