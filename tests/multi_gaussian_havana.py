# type:ignore
from numpy.typing import NDArray
import numpy as np
from symbolica import NumericalIntegrator, RandomNumberGenerator, Sample

# Params
eps = 1e-9
n_loops = 1
c_dim = 3*int(n_loops)
sigma = np.array(1.0)
sigma_disc = np.array([0.5, 1.0, 2.0])
offsets_disc = np.array([0.0, 0.5, 1.0])
num_sigma = sigma_disc.size
num_offsets = offsets_disc.size
niter = 10
neval = 10_000
n_integration_samples = 100_000
seed, stream_id = 42, 0
max_prob_ratio = 10.0
n_continuous_bins = 128
discrete_learning_rate = 1.5
continuous_learning_rate = 1.5


def multi_gaussian(sigma: NDArray, continuous: NDArray, discrete: NDArray | None = None, ) -> NDArray:
    # spherical param
    jac = (4*np.pi)**n_loops*np.ones((continuous.shape[0]))
    for i_loop in range(n_loops):
        start = 3*i_loop
        x1, x2, x3 = continuous.T[start:start+3, :]
        # x1 = x1 * (1 - 2*eps) + eps
        jac *= x1**2 / (1.0 - x1)**4

        r = x1/(1.0-x1)
        cos_az = (2*x2-1.0)
        sin_az = np.sqrt(1.0 - cos_az**2)
        m = ~np.isfinite(sin_az)
        if m.any():
            print(f"cos_az: {cos_az[m]}")
            print(f"sin_az: {sin_az[m]}")

        pol = 2*np.pi*x3

        continuous[:, start] = r * sin_az * np.cos(pol)
        continuous[:, start+1] = r * sin_az * np.sin(pol)
        continuous[:, start+2] = r * cos_az

    if discrete is None:
        sig = sigma
        offset = 0.0
    elif discrete.shape[1] == 1:
        sig = sigma[discrete[:, 0]]
        offset = 0.0
    else:
        sig = sigma[discrete[:, 0]]
        offset = offsets_disc[discrete[:, 1]][:, None]
        jac /= num_offsets

    norm_factor = np.sum((2*np.pi * sigma ** 2)**(continuous.shape[1]/2))
    res = jac * np.exp(-((continuous-offset)**2).sum(axis=1) / sig**2 / 2) / norm_factor
    res[np.isnan(res)] = 0.0

    return res


def eval_from_samples(samples: list[Sample], sigma: NDArray) -> NDArray:
    continuous = np.empty((len(samples), c_dim))
    d_dim = sigma.size
    discrete = np.empty((len(samples), len(samples[0].d)), dtype=np.uint64) if d_dim > 1 else None
    for i, sample in enumerate(samples):
        continuous[i] = sample.c
    if d_dim > 1:
        for i, sample in enumerate(samples):
            discrete[i] = sample.d

    return multi_gaussian(sigma, continuous, discrete)


def train_sampler(sampler: NumericalIntegrator, rng: RandomNumberGenerator, sigma: NDArray) -> None:
    for i in range(niter):
        samples = sampler.sample(neval, rng)
        sampler.add_training_samples(samples, eval_from_samples(samples, sigma))
        print("    it {}: {:.5f} +- {:.5f}, chi={:.3f}".format(i,
              *sampler.update(discrete_learning_rate, continuous_learning_rate)))


if __name__ == "__main__":
    sqrn = np.sqrt(n_integration_samples)
    continuous = np.random.uniform(eps, 1 - eps, (n_integration_samples, c_dim))
    single = multi_gaussian(sigma, continuous.copy(), )
    print("Uniform x-space samples:")
    print(f"    Single Gaussian: {single.mean():.5f} +- {single.std()/sqrn:.5f}, sigma:{sigma:.2f}")
    means = []
    errs = []
    for i, sig in enumerate(sigma_disc):
        disc = i * np.ones((continuous.shape[0], 1), dtype=np.uint64)
        multi = multi_gaussian(sigma_disc, continuous.copy(), disc)
        mean, err = multi.mean(), multi.std()/sqrn
        print(f"    Channel {i}: {mean:.5f} +- {err:.5f}, sigma:{sig:.2f}")
        means.append(mean)
        errs.append(err)
    print(f"    Sum: {sum(means)} +- {np.sqrt(sum([e**2 for e in errs]))}")

    # Initialize integrators
    havana_single = NumericalIntegrator.continuous(c_dim, n_continuous_bins)
    havana_disc = NumericalIntegrator.discrete([
        NumericalIntegrator.continuous(c_dim, n_continuous_bins)
        for _ in range(num_sigma)], max_prob_ratio)
    havana_double_disc = NumericalIntegrator.discrete([
        NumericalIntegrator.discrete([
            NumericalIntegrator.continuous(c_dim, n_continuous_bins)
            for k in range(num_offsets)], max_prob_ratio)
        for _ in range(num_sigma)], max_prob_ratio)
    havana_uniform = NumericalIntegrator.uniform(
        [num_sigma], NumericalIntegrator.continuous(c_dim, n_continuous_bins))
    havana_double_uniform = NumericalIntegrator.uniform(
        [num_sigma, num_offsets], NumericalIntegrator.continuous(c_dim, n_continuous_bins))
    rng = RandomNumberGenerator(seed, stream_id)

    print("Training Havana single gaussian")
    train_sampler(havana_single, rng, sigma)
    print("Training Havana uniform grids")
    train_sampler(havana_uniform, rng, sigma_disc)
    print("Training Havana double uniform grids")
    train_sampler(havana_double_uniform, rng, sigma_disc)
    print("Training Havana discrete grids")
    train_sampler(havana_disc, rng, sigma_disc)
    print("Training Havana double discrete grids")
    train_sampler(havana_double_disc, rng, sigma_disc)
    samples = havana_single.sample(n_integration_samples, rng)
    samples_disc = havana_disc.sample(n_integration_samples, rng)
    samples_double_disc = havana_double_disc.sample(n_integration_samples, rng)
    samples_uniform = havana_uniform.sample(n_integration_samples, rng)
    samples_double_uniform = havana_double_uniform.sample(n_integration_samples, rng)
    wgt = np.array([s.weights[0] for s in samples])
    wgt_disc = np.array([s.weights for s in samples_disc])
    wgt_double_disc = np.array([s.weights for s in samples_double_disc])
    wgt_uniform = np.array([s.weights[0] for s in samples_uniform])
    wgt_double_uniform = np.array([s.weights[0] for s in samples_double_uniform])
    res = eval_from_samples(samples, sigma) * wgt
    res_disc = eval_from_samples(samples_disc, sigma_disc)
    res_double_disc = eval_from_samples(samples_double_disc, sigma_disc)
    res_disc_disc = res_disc * wgt_disc[:, 0]
    res_disc_cont = res_disc * wgt_disc[:, 1]
    res_disc_prod = res_disc * wgt_disc.prod(axis=1)
    res_double_disc_0 = res_double_disc * wgt_double_disc[:, 0]
    res_double_disc_1 = res_double_disc * wgt_double_disc[:, 1]
    res_double_disc_2 = res_double_disc * wgt_double_disc[:, 2]
    res_double_disc_prod = res_double_disc * wgt_double_disc.prod(axis=1)
    res_uniform = eval_from_samples(samples_uniform, sigma_disc) * wgt_uniform
    res_double_uniform = eval_from_samples(samples_double_uniform, sigma_disc) * wgt_double_uniform
    print("Havana single:")
    print(f"    {res.mean():.5f} +- {res.std()/sqrn:.5f}")
    print("Havana uniform:")
    print(f"    {res_uniform.mean():.5f} +- {res_uniform.std()/sqrn:.5f}")
    print("Havana double uniform:")
    print(f"    {res_double_uniform.mean():.5f} +- {res_double_uniform.std()/sqrn:.5f}")
    print("Havana disc:")
    print(f"   disc wgts only: {res_disc_disc.mean():.5f} +- {res_disc_disc.std()/sqrn:.5f}")
    print(f"   cont wgts only: {res_disc_cont.mean():.5f} +- {res_disc_cont.std()/sqrn:.5f}")
    print(f"   prod of wgts  : {res_disc_prod.mean():.5f} +- {res_disc_prod.std()/sqrn:.5f}")
    print("Havana double disc:")
    print(f"   disc wgts 0 only: {res_double_disc_0.mean():.5f} +- {res_double_disc_0.std()/sqrn:.5f}")
    print(f"   disc wgts 1 only: {res_double_disc_1.mean():.5f} +- {res_double_disc_1.std()/sqrn:.5f}")
    print(f"   cont wgts only: {res_double_disc_2.mean():.5f} +- {res_double_disc_2.std()/sqrn:.5f}")
    print(f"   prod of wgts  : {res_double_disc_prod.mean():.5f} +- {res_double_disc_prod.std()/sqrn:.5f}")
