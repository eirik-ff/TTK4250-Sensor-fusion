# %% imports
import numpy as np
import scipy
import scipy.io
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt

from gaussparams import GaussParams
from mixturedata import MixtureParameters
import dynamicmodels
import measurementmodels
import imm
import ekf
import estimationstatistics as estats

# the given code gives warnings due to division by zero, which is handled.
# the below lines will mute these warnings
import warnings
warnings.filterwarnings("ignore")


# %% plot config check and style setup

# to see your plot config
print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

# set styles
try:
    # installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
    # gives quite nice plots
    plt_styles = ["science", "grid", "ieee", "bright", "no-latex"]
    plt.style.use(plt_styles)
    print(f"pyplot using style set {plt_styles}")
except Exception as e:
    print(e)
    print("setting grid and only grid and legend manually")
    plt.rcParams.update(
        {
            # setgrid
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "k",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": True,
            "legend.numpoints": 1,
        }
    )

# %% load data
use_pregen = True
# you can generate your own data if set to false
if use_pregen:
    data_filename = "data_for_imm.mat"
    loaded_data = scipy.io.loadmat(data_filename)
    Z = loaded_data["Z"].T
    K = loaded_data["K"].item()
    Ts = loaded_data["Ts"].item()
    Xgt = loaded_data["Xgt"].T
else:
    K = 100
    Ts = 2.5
    sigma_z = 2.25
    sigma_a = 0.7  # effective all the time
    sigma_omega = 5e-4 * np.pi  # effective only in turns

    init_x = [0, 0, 2, 0, 0]
    init_P = np.diag([25, 25, 3, 3, 0.0005]) ** 2
    # [Xgt, Z] = simulate_atc(q, r, K, init, false);
    raise NotImplementedError


fig1, ax1 = plt.subplots(num=1, clear=True)
ax1.scatter(*Z.T[:2], marker='o', c='m', s=8, label="Measurement")
ax1.plot(*Xgt.T[:2], c='c', linewidth=1.5, alpha=1, label="Ground truth")
ax1.legend()


# %% tune single filters

# parameters
## for measure model
sigma_z = 3

## for CV model
sigma_a_CV = 0.17

## for CT model
sigma_a_CT = 0.17
sigma_omega = 0.0001 * np.pi

# initial values
init_mean = np.array([0, 0, 2, 0, 0])
init_cov = np.diag([25, 25, 3, 3, 0.0005]) ** 2

init_state_CV = GaussParams(init_mean[:4], init_cov[:4, :4])  # get rid of turn rate
init_state_CT = GaussParams(init_mean, init_cov)  # same init otherwise
init_states = [init_state_CV, init_state_CT]

# create models
measurement_model_CV = measurementmodels.CartesianPosition(sigma_z)
measurement_model_CT = measurementmodels.CartesianPosition(sigma_z, state_dim=5)
CV = dynamicmodels.WhitenoiseAccelleration(sigma_a_CV)
CT = dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega)

# create filters
filters = []
filters.append(ekf.EKF(CV, measurement_model_CV))
filters.append(ekf.EKF(CT, measurement_model_CT))

# allocate
pred = []
upd = []
NIS = np.empty((2, K))
NEES_pred = np.empty((2, K))
NEES_upd = np.empty((2, K))
err_pred = np.empty((2, 2, K))  # (filters, vel/pos, time)
err_upd = np.empty((2, 2, K))  # (filters, vel/pos, time)

# per filter
for i, (ekf_filter, init) in enumerate(zip(filters, init_states)):
    # setup per filter
    updated = init
    ekfpred_list = []
    ekfupd_list = []

    # over time steps
    for k, (zk, x_gt_k) in enumerate(zip(Z, Xgt)):  # bypass any EKF.sequence problems
        # filtering
        predicted = ekf_filter.predict(updated, Ts)
        updated = ekf_filter.update(zk, predicted)

        # store per time
        ekfpred_list.append(predicted)
        ekfupd_list.append(updated)

        # measurement metric
        NIS[i, k] = ekf_filter.NIS(zk, predicted)

    # stor per filter
    pred.append(ekfpred_list)
    upd.append(ekfupd_list)

    # extract means and covs for metric processing
    x_bar = np.array([p.mean for p in ekfpred_list])
    x_hat = np.array([u.mean for u in ekfupd_list])

    P_bar = np.array([p.cov for p in ekfpred_list])
    P_hat = np.array([u.cov for u in ekfupd_list])

    # calculate metrics
    NEES_pred[i] = estats.NEES_sequence_indexed(x_bar, P_bar, Xgt, idxs=np.arange(4))
    NEES_upd[i] = estats.NEES_sequence_indexed(x_hat, P_hat, Xgt, idxs=np.arange(4))

    err_pred[i, 0] = estats.distance_sequence_indexed(x_bar, Xgt, np.arange(2))
    err_pred[i, 1] = estats.distance_sequence_indexed(x_bar, Xgt, np.arange(2, 4))
    err_upd[i, 0] = estats.distance_sequence_indexed(x_hat, Xgt, np.arange(2))
    err_upd[i, 1] = estats.distance_sequence_indexed(x_hat, Xgt, np.arange(2, 4))


# errors
RMSE_pred = err_pred.mean(axis=2)
RMSE_upd = err_upd.mean(axis=2)

# measurement consistency
ANIS = NIS.mean(axis=1)
CINIS = np.array(scipy.stats.chi2.interval(0.9, 2))
CIANIS = np.array(scipy.stats.chi2.interval(0.9, K * 2)) / K
print(f"ANIS={ANIS} with CIANIS={CIANIS}")


# plot individual estimates
fig2, axs2 = plt.subplots(2, 2, num=2, clear=True)
for axu, axl, u_s, rmse_pred, rmse_upd in zip(
    axs2[0], axs2[1], upd, RMSE_pred, RMSE_upd
):
    # ax.scatter(*Z.T)
    x = np.array([data.mean for data in u_s])
    axu.plot(*x.T[:2])
    rmsestr = ", ".join(f"{num:.3f}" for num in (*rmse_upd, *rmse_pred))
    axu.set_title(f"RMSE(p_u, v_u, p_pr, v_pr)\n{rmsestr}", fontsize=8)
    #axu.axis("equal")
    axu.set_xlabel('X')
    axu.set_ylabel('Y', rotation=0)

    if x.shape[1] >= 5:
        axl.plot(np.arange(K) * Ts, x.T[4])
    axl.plot(np.arange(K) * Ts, Xgt[:, 4])
    axl.set_xlabel('time step')

axs2[1, 0].set_ylabel(r"$\omega$", rotation=0)
axs2[0, 0].legend(['CV'])
axs2[0, 1].legend(['CT'])
fig2.tight_layout(w_pad=0.5, h_pad=1.0)


# plot predicted vs ground truth
fig7, axs7 = plt.subplots(1, 2, num=7, clear=True)
fig7.suptitle("Model predictions vs ground truth")

axs7[0].plot(*Xgt.T[:2], label="Ground truth")
x_CV = np.array([data.mean for data in upd[0]])
axs7[0].plot(*x_CV.T[:2], '--r', label="CV model")
axs7[0].legend(fontsize=6)

axs7[1].plot(*Xgt.T[:2], label="Ground truth")
x_CT = np.array([data.mean for data in upd[1]])
axs7[1].plot(*x_CT.T[:2], '--r', label="CT model")
axs7[1].legend(fontsize=6)



# plot errors
fig3, axs3 = plt.subplots(1, 3, num=3, clear=True)
fig3.subplots_adjust(wspace=0.4)

axs3[0].plot(np.arange(K) * Ts, NIS[0], label='CV')
axs3[0].plot(np.arange(K) * Ts, NIS[1], label='CT')
for ci, anis, lab in zip(CIANIS, ANIS, ['CV', 'CT']):
    axs3[0].plot(np.arange(K) * Ts, ci * np.ones((K,)), '--r', label=f'CI_{lab}')
    axs3[0].plot(np.arange(K) * Ts, anis * np.ones((K,)), '--g', label=f"ANIS_{lab}")
axs3[0].set_title("NIS")
axs3[0].set_xlabel('Time step')
axs3[0].legend(fontsize=6, bbox_to_anchor=(-0.3,1))

axs3[1].plot(np.arange(K) * Ts, err_upd[:, 0].T)
# axs3[1].plot(np.arange(K) * Ts, err_upd[1, :, 0])
axs3[1].set_title("pos error (gt)")
axs3[1].set_xlabel('Time step')

axs3[2].plot(np.arange(K) * Ts, err_upd[:, 1].T)
# axs3[2].plot(np.arange(K) * Ts, err_upd[1, :, 1])
axs3[2].set_title("vel error (gt)")
axs3[2].set_xlabel('Time step')

# %% tune IMM by only looking at the measurements
sigma_z = 1.5
sigma_a_CV = 0.1
sigma_a_CT = 0.3
sigma_omega = 0.0001 * np.pi
PI = np.array([[0.95, 0.05], [0.05, 0.95]])
assert np.allclose(PI.sum(axis=1), 1), "rows of PI must sum to 1"

# make model
measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim=5)
CV = dynamicmodels.WhitenoiseAccelleration(sigma_a_CV, n=5)
CT = dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega)
ekf_filters = []
ekf_filters.append(ekf.EKF(CV, measurement_model))
ekf_filters.append(ekf.EKF(CT, measurement_model))
imm_filter = imm.IMM(ekf_filters, PI)

init_weights = np.array([0.5] * 2)
init_mean = [0] * 5
# HAVE TO BE DIFFERENT: use intuition, eg. diag guessed distance to true values squared.
init_cov = np.diag([1, 1, 1, 1, 1])*100
init_mode_states = [GaussParams(init_mean, init_cov)] * 2  # copy of the two modes
init_immstate = MixtureParameters(init_weights, init_mode_states)

imm_preds = []
imm_upds = []
imm_ests = []
updated_immstate = init_immstate
for zk in Z:
    predicted_immstate = imm_filter.predict(updated_immstate, Ts)
    updated_immstate = imm_filter.update(zk, predicted_immstate)
    estimate = imm_filter.estimate(updated_immstate)

    imm_preds.append(predicted_immstate)
    imm_upds.append(updated_immstate)
    imm_ests.append(estimate)

x_est = np.array([est.mean for est in imm_ests])
prob_est = np.array([upds.weights for upds in imm_upds])

# consistency
NISes_comb = [imm_filter.NISes(zk, pred_k) for zk, pred_k in zip(Z, imm_preds)]
NIS = np.array([n[0] for n in NISes_comb])
NISes = np.array([n[1] for n in NISes_comb])
ANIS = NIS.mean()
CINIS = np.array(scipy.stats.chi2.interval(0.9, 2))
CIANIS = np.array(scipy.stats.chi2.interval(0.9, 2 * K)) / K
print(f"ANIS={ANIS} with CIANIS={CIANIS}")

# plot imm
fig4, axs4 = plt.subplots(2, 2, num=4, clear=True)

axs4[0, 0].plot(*x_est.T[:2], label="est", color="C0")
axs4[0, 0].scatter(*Z.T, label="z", color="C1", s=4)
axs4[0, 0].legend(fontsize=8)

axs4[0, 1].plot(np.arange(K) * Ts, x_est[:, 4], label=r"$\omega$")
axs4[0, 1].legend(fontsize=8)

axs4[1, 0].plot(np.arange(K) * Ts, prob_est, label=r"$Pr(s)$")
axs4[1, 0].legend([r"$Pr(CV)$", r"$Pr(CT)$"], fontsize=4)

axs4[1, 1].plot(np.arange(K) * Ts, NIS, label="NIS")
axs4[1, 1].plot(np.arange(K) * Ts, NISes)

ratio_in_CI = np.sum(np.less_equal(CINIS[0], NIS) * np.less_equal(NIS, CINIS[1])) / K
CI_LABELS = ["CI0", "CI1"]
for ci, cilbl in zip(CINIS, CI_LABELS):
    axs4[1, 1].plot([1, K * Ts], np.ones(2) * ci, "--r", label=cilbl)
axs4[1, 1].text(K * Ts * 1.1, 1, f"{ratio_in_CI} inside CI", rotation=90)
axs4[1, 1].legend(fontsize=4)

fig4.subplots_adjust(wspace=0.25, hspace=0.4)

# plot predicted vs ground truth
fig8, ax8 = plt.subplots(1, num=8, clear=True)
fig8.suptitle("IMM predictions vs ground truth")

ax8.plot(*Xgt.T[:2], label="Ground truth")
ax8.plot(*x_est.T[:2], '--r', label="IMM model")
ax8.legend(fontsize=6)

# %% tune IMM by looking at ground truth
sigma_z = 1.5
sigma_a_CV = 0.4
sigma_a_CT = 0.1
sigma_omega = 0.002 * np.pi
PI = np.array([[0.95, 0.05], [0.05, 0.95]])
assert np.allclose(PI.sum(axis=1), 1), "rows of PI must sum to 1"

# make model
measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim=5)
CV = dynamicmodels.WhitenoiseAccelleration(sigma_a_CV, n=5)
CT = dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega)
ekf_filters = []
ekf_filters.append(ekf.EKF(CV, measurement_model))
ekf_filters.append(ekf.EKF(CT, measurement_model))
imm_filter = imm.IMM(ekf_filters, PI)

init_weights = np.array([0.5] * 2)
init_mean = [0] * 5
# HAVE TO BE DIFFERENT: use intuition, eg. diag guessed distance to true values squared.
init_cov = np.diag([1, 1, 1, 1, 0.0001])*100**2
init_mode_states = [GaussParams(init_mean, init_cov)] * 2  # copy of the two modes
init_immstate = MixtureParameters(init_weights, init_mode_states)

imm_preds = []
imm_upds = []
imm_ests_pred = []
imm_ests_upd = []

updated_immstate = init_immstate
for zk in Z:
    predicted_immstate = imm_filter.predict(updated_immstate, Ts)
    updated_immstate = imm_filter.update(zk, predicted_immstate)

    estimate_pred = imm_filter.estimate(predicted_immstate)
    estimate_upd = imm_filter.estimate(updated_immstate)

    imm_preds.append(predicted_immstate)
    imm_upds.append(updated_immstate)

    imm_ests_pred.append(estimate_pred)
    imm_ests_upd.append(estimate_upd)

# extract all means and covs
x_bar = np.array([est.mean for est in imm_ests_pred])
P_bar = np.array([est.cov for est in imm_ests_pred])

x_hat = np.array([est.mean for est in imm_ests_upd])
P_hat = np.array([est.cov for est in imm_ests_upd])

x_bar_modes = np.array([[comp.mean for comp in pr.components] for pr in imm_preds])
P_bar_modes = np.array([[comp.cov for comp in pr.components] for pr in imm_preds])

x_hat_modes = np.array([[comp.mean for comp in pr.components] for pr in imm_upds])
P_hat_modes = np.array([[comp.cov for comp in pr.components] for pr in imm_upds])

mode_prob = np.array([upds.weights for upds in imm_upds])

# consistency: NIS
NISes_comb = (imm_filter.NISes(zk, pred_k) for zk, pred_k in zip(Z, imm_preds))
NIS, NISes = [np.array(n) for n in zip(*NISes_comb)]
ANIS = NIS.mean()
CINIS = np.array(scipy.stats.chi2.interval(0.9, 2))
CIANIS = np.array(scipy.stats.chi2.interval(0.9, 2 * K)) / K

# consistency: NEES
NEES_pred = estats.NEES_sequence_indexed(x_bar, P_bar, Xgt, idxs=np.arange(4))
NEESes_pred = np.array(
    [
        estats.NEES_sequence_indexed(x, P, Xgt, idxs=np.arange(4))
        for x, P in zip(x_bar_modes, P_bar_modes)
    ]
)
NEES_upd = estats.NEES_sequence_indexed(x_hat, P_hat, Xgt, idxs=np.arange(4))
NEESes_upd = np.array(
    [
        estats.NEES_sequence_indexed(x, P, Xgt, idxs=np.arange(4))
        for x, P in zip(x_hat_modes, P_hat_modes)
    ]
)

ANEES_pred = NEES_pred.mean()
ANEES_upd = NEES_upd.mean()

CINEES = np.array(scipy.stats.chi2.interval(0.9, 4))
CIANEES = np.array(scipy.stats.chi2.interval(0.9, 4 * K)) / K
print(f"ANIS={ANIS} and CIANIS={CIANIS}")
print(f"ANEES_upd={ANEES_upd}, ANEES_pred={ANEES_pred} and CIANEES={CIANEES}")

#  errors
pos_err = estats.distance_sequence_indexed(x_hat, Xgt, idxs=np.arange(2))
# np.sqrt(np.sum((x_est[:, :2] - Xgt[:, :2]) ** 2, axis=1))
vel_err = estats.distance_sequence_indexed(x_hat, Xgt, idxs=np.arange(2, 4))
# np.sqrt(np.sum((x_est[:, 2:4] - Xgt[:, 2:4]) ** 2, axis=1))
pos_RMSE = np.sqrt(
    np.mean(pos_err ** 2)
)  # not true RMSE (which is over monte carlo simulations)
vel_RMSE = np.sqrt(
    np.mean(vel_err ** 2)
)  # not true RMSE (which is over monte carlo simulations)
pos_peak_deviation = pos_err.max()
vel_peak_deviation = vel_err.max()

rmsestr = ", ".join(f"{num:.3f}" for num in (pos_RMSE, vel_RMSE))
devstr = ", ".join(f"{num:.3f}" for num in (pos_peak_deviation, vel_peak_deviation))
# plot
fig5, axs5 = plt.subplots(2, 2, num=5, clear=True)
axs5[0, 0].plot(*x_hat.T[:2], label="est", color="C0")
axs5[0, 0].scatter(*Z.T, label="z", color="C1")
axs5[0, 0].legend(fontsize=8)
axs5[0, 0].set_title(f"RMSE(p, v) = {rmsestr}\npeak_dev(p, v) = {devstr}.0")
axs5[0, 1].plot(np.arange(K) * Ts, x_hat[:, 4], label=r"$\hat{\omega}$")
axs5[0, 1].plot(np.arange(K) * Ts, Xgt[:, 4], label=r"$\omega_{gt}$")
axs5[0, 1].legend(fontsize=8)
for s in range(len(ekf_filters)):
    axs5[1, 0].plot(np.arange(K) * Ts, prob_est[:, s], label=rf"$Pr({['CV','CT'][s]})$")
axs5[1, 0].legend(fontsize=4)
axs5[1, 1].plot(np.arange(K) * Ts, NIS, label="NIS")
axs5[1, 1].plot(np.arange(K) * Ts, NISes)

ratio_in_CI = np.sum(np.less_equal(CINIS[0], NIS) * np.less_equal(NIS, CINIS[1])) / K
CI_LABELS = ["CI0", "CI1"]
for ci, cilbl in zip(CINIS, CI_LABELS):
    axs5[1, 1].plot([1, K * Ts], np.ones(2) * ci, "--r", label=cilbl)
axs5[1, 1].text(K * Ts * 1.1, 1, f"{ratio_in_CI} inside CI", rotation=90)
axs5[1, 1].legend(fontsize=6)

fig6, axs6 = plt.subplots(2, 2, sharex=True, num=6, clear=True)
axs6[0, 0].plot(np.arange(K) * Ts, pos_err)
axs6[0, 0].set_ylabel("position error")
axs6[0, 1].plot(np.arange(K) * Ts, vel_err)
axs6[0, 1].yaxis.set_label_position("right")
axs6[0, 1].set_ylabel("velocity error")
axs6[1, 0].plot(np.arange(K) * Ts, NIS)
axs6[1, 0].plot(np.arange(K) * Ts, NISes)
ratio_in_CI = np.mean(np.less_equal(CINIS[0], NIS) * np.less_equal(NIS, CINIS[1]))
axs6[1, 0].set_ylabel(f"NIS: {ratio_in_CI}% in CI")
axs6[1, 0].plot([0, Ts * (K - 1)], np.repeat(CINIS[None], 2, 0), "r--")
# axs6[1, 0].text(K * Ts * 1.1, -2, f"{ratio_in_CI}% inside CI", rotation=90)
axs6[1, 0].set_ylim([0, 2 * CINIS[1]])

axs6[1, 1].plot(np.arange(K) * Ts, NEES_pred)
axs6[1, 1].plot(np.arange(K) * Ts, NEES_upd)
# axs6[1, 1].plot(np.arange(K) * Ts, NISes)
ratio_in_CI_nees = np.mean(
    np.less_equal(CINEES[0], NEES_upd) * np.less_equal(NEES_upd, CINEES[1])
)
# axs6[1, 1].text(K * Ts * 1.1, -2, f"{ratio_in_CI_nees}% inside CI", rotation=90)
axs6[1, 1].yaxis.set_label_position("right")
axs6[1, 1].set_ylabel(f"NEES: {ratio_in_CI_nees}% in CI")
axs6[1, 1].plot([0, Ts * (K - 1)], np.repeat(CINEES[None], 2, 0), "r--")
axs6[1, 1].set_ylim([0, 2 * CINEES[1]])
# axs6[1, 1].text(K * Ts * 1.1, -2, f"{ratio_in_CI_nees}% inside CI", rotation=90)


# plot predicted vs ground truth
fig9, ax9 = plt.subplots(1, num=9, clear=True)
fig9.suptitle("IMM predictions vs ground truth")

ax9.plot(*Xgt.T[:2], label="Ground truth")
ax9.plot(*x_hat.T[:2], '--r', label="IMM model")
ax9.legend(fontsize=10)


# %%

# %%
