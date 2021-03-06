# %% Imports
from gaussparams import GaussParams
import measurmentmodels
import dynamicmodels
import ekf
import scipy
import scipy.stats
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# to see your plot config
print(f'matplotlib backend: {matplotlib.get_backend()}')
print(f'matplotlib config file: {matplotlib.matplotlib_fname()}')
print(f'matplotlib config dir: {matplotlib.get_configdir()}')
plt.close('all')

# set styles
try:
    # installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
    # gives quite nice plots
    plt_styles = ['science', 'ieee', 'grid', 'bright']
    plt.style.use(plt_styles)
    print(f'pyplot using style set {plt_styles}')
except Exception as e:
    print(e)
    print('setting grid and only grid and legend manually')
    plt.rcParams.update({
        # set grid
        'axes.grid': True,
        'grid.linestyle': ':',
        'grid.color': 'k',
        'grid.alpha': 0.5,
        'grid.linewidth': 0.5,
        # Legend
        'legend.frameon': True,
        'legend.framealpha': 1.0,
        'legend.fancybox': True,
        'legend.numpoints': 1,
    })


# %% get and plot the data

# TODO: choose this for the last task
usePregen = True  # choose between own generated data and pre generated

if usePregen:
    data_path = 'data_for_ekf.mat'
    loadData: dict = scipy.io.loadmat(data_path)
    K: int = int(loadData['K'])  # The number of time steps
    Ts: float = float(loadData['Ts'])  # The sampling time
    Xgt: np.ndarray = loadData['Xgt'].T  # ground truth
    Z: np.ndarray = loadData['Z'].T  # the measurements
else:
    from sample_CT_trajectory import sample_CT_trajectory
    np.random.seed(10)  # random seed can be set for repeatability

    # initial state distribution
    x0 = np.array([0, 0, 1, 1, 0])
    P0 = np.diag([50, 50, 10, 10, np.pi/4]) ** 2

    # model parameters to sample from
    # TODO for toying around
    sigma_a_true = 0.25
    sigma_omega_true = np.pi/15
    sigma_z_true = 3

    # sampling interval a length
    K = 1000
    Ts = 0.1

    # get data
    Xgt, Z = sample_CT_trajectory(
        K, Ts, x0, P0, sigma_a_true, sigma_omega_true, sigma_z_true)

# show ground truth and measurements
fig, ax = plt.subplots(num=1, clear=True)
ax.plot(*Xgt.T[:2], color='k', alpha=0.8, label='Ground truth')
ax.scatter(*Z.T, s=1.5, color='C1', marker='.', label='Measurement')
ax.set_title('Ground truth and measurements')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()

# show turn rate
fig2, ax2 = plt.subplots(num=2, clear=True)
ax2.plot(Xgt.T[4])
ax2.set_title('Turn rate')
ax2.set_xlabel('time step')
ax2.set_ylabel('turn rate')


# %% a: tune by hand and comment

# set parameters
#sigma_a, sigma_z = 5, 4
sigma_a, sigma_z = 3.1, 2.6


# create the model and estimator object
dynmod = dynamicmodels.WhitenoiseAccelleration(sigma_a)
measmod = measurmentmodels.CartesianPosition(sigma_z)
ekf_filter = ekf.EKF(dynmod, measmod)
print(ekf_filter)  # make use of the @dataclass automatic repr

# initialize mean and covariance
x_bar_init = np.array([0,0,0,0])
P_bar_init = np.diag([1,1,1,1])*100**2

# initialize mean and covariance based on the two first measurements
# as derived in task 2
Kp1 = np.eye(2);
Kp0 = np.zeros((2,2))
Ku1 = -1/Ts * np.eye(2)
Ku0 = 1/Ts * np.eye(2)
Kx = np.block([[Kp1, Kp0], [Ku1, Ku0]])
z_init = np.reshape(Z[0:2, :], (4,1))
x_bar_init = (Kx @ z_init).squeeze()

tmp = np.block([np.eye(2), -Ts*np.eye(2)])
Q = dynmod.Q(x_bar_init, Ts)
R = measmod.R(x_bar_init)
zeros = np.zeros(R.shape)
P22 = tmp @ Q @ tmp.T + R
P_bar_init = Kx @ np.block([[R, zeros], [zeros, P22]]) @ Kx.T

init_ekfstate = ekf.GaussParams(x_bar_init, P_bar_init)

# estimate
ekfpred_list, ekfupd_list = \
    ekf_filter.estimate_sequence(Z, init_ekfstate, Ts)

# get statistics:
# TODO: see that you sort of understand what this does
stats = ekf_filter.performance_stats_sequence(
    K, Z=Z, ekfpred_list=ekfpred_list, ekfupd_list=ekfupd_list,
    X_true=Xgt[:, :4], norm_idxs=[[0, 1], [2, 3]], norms=[2, 2]
)

print(f'keys in stats is {stats.dtype.names}')

# %% Calculate average performance metrics
# stats['dists_pred'] contains 2 norm of position and speed for each time index
# same for 'dists_upd'
# square stats['dists_pred'] -> take its mean over time -> take square root
RMSE_pred = np.sqrt(np.mean(stats['dists_pred']**2, axis=0))
RMSE_upd = np.sqrt(np.mean(stats['dists_upd']**2, axis=0))

fig3, ax3 = plt.subplots(num=3, clear=True)

ax3.plot(*Xgt.T[:2], color='k', linewidth=1, label='Ground truth')
ax3.plot(*ekfupd_list.mean.T[:2], color='g', linewidth=1, label='EKF')
RMSEs_str = ", ".join(f"{v:.2f}" for v in (*RMSE_pred, *RMSE_upd))
ax3.set_title(rf'$\sigma_a = {sigma_a}$, $\sigma_z= {sigma_z}$,' + \
              f'\nRMSE(p\_p, p\_v, u\_p, u\_v) = ({RMSEs_str})')
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.legend()


# import sys
# sys.exit(0)

# %% Task 5 b and c

# % parameters for the parameter grid
# TODO: pick reasonable values for grid search
# n_vals = 20
# is Ok, try lower to begin with for more speed (20*20*1000 = 400 000 KF steps)
n_vals = 10
sigma_a_low = 2
sigma_a_high = 8
sigma_z_low = 2
sigma_z_high = 8

# % set the grid on logscale(not mandatory)
sigma_a_list = np.logspace(
    np.log10(sigma_a_low), np.log10(sigma_a_high), n_vals, base=10
)
sigma_z_list = np.logspace(
    np.log10(sigma_z_low), np.log10(sigma_z_high), n_vals, base=10
)

dtype = stats.dtype  # assumes the last cell has been run without faults
stats_array = np.empty((n_vals, n_vals, K), dtype=dtype)

# %% run through the grid and estimate
# ? Should be more or less a copy of the above
for i, sigma_a in enumerate(sigma_a_list):
    dynmod = dynamicmodels.WhitenoiseAccelleration(sigma_a)
    for j, sigma_z in enumerate(sigma_z_list):
        measmod = measurmentmodels.CartesianPosition(sigma_z)
        ekf_filter = ekf.EKF(dynmod, measmod)

        ekfpred_list, ekfupd_list = \
            ekf_filter.estimate_sequence(Z, init_ekfstate, Ts)
        stats_array[i, j] = ekf_filter.performance_stats_sequence(
                K, Z=Z, ekfpred_list=ekfpred_list, ekfupd_list=ekfupd_list,
                X_true=Xgt[:, :4], norm_idxs=[[0, 1], [2, 3]], norms=[2, 2]
            )

# %% calculate averages

# remember to use axis argument, see eg. stats_array['dists_pred'].shape
# The way done below is computationally heavy since it calculates the square
# and then the square root of the same number. i didn't find an easier way to
# just take the square/sqrt along a single axis
RMSE_pred = np.sqrt(np.mean(np.square(stats_array['dists_pred']), axis=2))
RMSE_upd = np.sqrt(np.mean(np.square(stats_array['dists_upd']), axis=2))
ANEES_pred = np.mean(stats_array['NEESpred'], axis=2)
ANEES_upd = np.mean(stats_array['NEESupd'], axis=2)
ANIS = np.mean(stats_array['NIS'], axis=2)


# %% find confidence regions for NIS and plot
confprob = 0.95  # number to use for confidence interval, 95% conf int
# confidence intervall for NIS, hint: scipy.stats.chi2.interval
CINIS = np.asarray(
    scipy.stats.chi2.interval(confprob, ekf_filter.sensor_model.m*K)) / K
print("CINIS:", CINIS)

CINIS_gamma = np.asarray(scipy.stats.gamma.interval(confprob, K*2/2) ) / K * 2
print("CINIS_gamma", CINIS_gamma)

# plot
fig4 = plt.figure(4, clear=True)
ax4 = plt.gca(projection='3d')
ax4.plot_surface(*np.meshgrid(sigma_a_list, sigma_z_list),
                 ANIS, alpha=0.8)
CS = ax4.contour(*np.meshgrid(sigma_a_list, sigma_z_list),
            ANIS, [1, 1.5, *CINIS, 2.5, 3], offset=0)  # , extend3d=True, colors='yellow')
ax4.set_xlabel(r'$\sigma_a$', labelpad=-6)
ax4.set_ylabel(r'$\sigma_z$', labelpad=-6)
ax4.set_zlabel('ANIS', labelpad=-6, rotation=90)
ax4.set_zlim(0, 10)
ax4.view_init(30, 40)
ax4.tick_params(axis='both', pad=-4)

CS_legends = ['1', '1.5', f'Low: {CINIS[0]:.2f}', f'High: {CINIS[1]:.2f}', '2.5', '3']
for i, label in enumerate(CS_legends):
    CS.collections[i].set_label(label)
ax4.legend(loc='upper center', ncol=3, prop={'size': 5})

# %% find confidence regions for NEES and plot
confprob = 0.95
CINEES = np.asarray(
    scipy.stats.chi2.interval(confprob, ekf_filter.dynamic_model.n*K)) / K
print("CINEES:", CINEES)

# plot
fig5 = plt.figure(5, clear=True)
subplot_layout = (2,1)
ax5s = [fig5.add_subplot(*subplot_layout, 1, projection='3d'),
        fig5.add_subplot(*subplot_layout, 2, projection='3d')]
ax5s[0].plot_surface(*np.meshgrid(sigma_a_list, sigma_z_list),
                     ANEES_pred, alpha=0.8)
ax5s[0].contour(*np.meshgrid(sigma_a_list, sigma_z_list),
                ANEES_pred, [3, 3.5, *CINEES, 4.5, 5], offset=0)
ax5s[0].set_xlabel(r'$\sigma_a$', labelpad=-7)
ax5s[0].set_ylabel(r'$\sigma_z$', labelpad=-7)
ax5s[0].set_zlabel('ANEES\_pred', labelpad=-7, rotation=90)
ax5s[0].set_zlim(0, 20)
ax5s[0].view_init(40, 30)
ax5s[0].tick_params(axis='both', pad=-4)
ax5s[0].locator_params(nbins=4, integer=True, min_n_ticks=3)

ax5s[1].plot_surface(*np.meshgrid(sigma_a_list, sigma_z_list),
                     ANEES_upd, alpha=0.8)
ax5s[1].contour(*np.meshgrid(sigma_a_list, sigma_z_list),
                ANEES_upd, [3, 3.5, *CINEES, 4.5, 5], offset=0)
ax5s[1].set_xlabel(r'$\sigma_a$', labelpad=-7)
ax5s[1].set_ylabel(r'$\sigma_z$', labelpad=-7)
ax5s[1].set_zlabel('ANEES\_upd',  labelpad=-7, rotation=90)
ax5s[1].set_zlim(0, 20)
ax5s[1].view_init(40, 30)
ax5s[1].tick_params(axis='both', pad=-4)
ax5s[1].locator_params(nbins=4, integer=True, min_n_ticks=3)

fig5.tight_layout(h_pad=5.0)

# %% see the intersection of NIS and NEESes
fig6, ax6 = plt.subplots(num=6, clear=True)
cont_upd = ax6.contour(*np.meshgrid(sigma_a_list, sigma_z_list),
                       ANEES_upd, CINEES, colors=['C0', 'C1'])
cont_pred = ax6.contour(*np.meshgrid(sigma_a_list, sigma_z_list),
                        ANEES_pred, CINEES, colors=['C2', 'C3'])
cont_nis = ax6.contour(*np.meshgrid(sigma_a_list, sigma_z_list),
                       ANIS, CINIS, colors=['C4', 'C5'])

for cs, l in zip([cont_upd, cont_pred, cont_nis], ['NEESupd', 'NEESpred', 'NIS']):
    for c, hl in zip(cs.collections, ['low', 'high']):
        c.set_label(l + '\_' + hl)
ax6.legend()
ax6.set_xlabel(r'$\sigma_a$')
ax6.set_ylabel(r'$\sigma_z$')

# %% show all the plots
plt.show()
