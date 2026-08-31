import numpy as np
import matplotlib.pyplot as plt
import os
from args_parser import parse_args
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, ttest_rel

## ====================================================================
## ====================== Loading and Parameters ======================
## ====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

args = parse_args()

data_path = os.path.join(DATA_DIR, 'SpikesX10U12D.npy')
spikes_data = np.load(data_path, allow_pickle=True)

## ====================================================================
## ===================== Q1: Firing rate statistics ===================
## ====================================================================

unit_index = 0
direction_index = 0
trial_duration = 1.28

#slicing the relevant data
unit_0_diraction_0 = spikes_data[unit_index, direction_index, :]

def count_in_trial_duration(spike_time):
    spike_array = np.array(spike_time, dtype=float)
    return np.sum(spike_array<=trial_duration)

v_count = np.vectorize(count_in_trial_duration, otypes=[int])
spike_count = v_count(unit_0_diraction_0)

fire_rate = spike_count / trial_duration

mean_fire_rate = np.mean(fire_rate)
median_fire_rate = np.median(fire_rate)
std_fire_rate = np.std(fire_rate)

print(f"Unit 1, direction (0°) statistics:")
print(f"mean: {mean_fire_rate:.3f} Hz")
print(f"median: {median_fire_rate:.3f} Hz")
print(f"std: {std_fire_rate:.3f} Hz")

## ====================================================================
## ================ Q2: PSTH calculations and display =================
## ====================================================================

#bin setting
bin_width = trial_duration / args.n_bins
bin_edges = np.linspace(0, trial_duration, args.n_bins + 1)

#zero mat for results
spikes_hist_counts = np.zeros((args.n_units, args.n_directions, args.n_repetitions, args.n_bins))

#the function get as input a list of spikes time of a single trial and count the spikes in each bin
def calc_trial_histogram(spike_times):
    spike_array = np.asarray(spike_times, dtype= float)
    count, _ = np.histogram(spike_array, bins = bin_edges)
    return count

#create a new vector of all of the calc with the help of the function above
v_histogram = np.vectorize(calc_trial_histogram, otypes=[object])

flat_trials = v_histogram(spikes_data.ravel())

#fill in the histogram data and reshaping the matrix

spikes_hist_counts = np.vstack(flat_trials).reshape(args.n_units, args.n_directions, args.n_repetitions, args.n_bins)

#calc the average of the 200 trials
psth_average = np.mean(spikes_hist_counts, axis=2)

psth_hz_calc = psth_average/bin_width

chosen_unit_idx = 4
directions = np.arange(0,360,30)

#creating the graph
fig, axes = plt.subplots(2,6,figsize=(15,10), sharex=True, sharey=True)
fig.suptitle(f'Unit #{chosen_unit_idx + 1} - PSTH per direction', fontsize = 16)

bin_centers = bin_edges[:-1]+bin_width/2

for i in range(6):
    axes[0,i].bar(bin_centers, psth_hz_calc[chosen_unit_idx, i], width=bin_width)
    axes[0,i].set_title(f'{directions[i]} deg', fontsize=15)

    axes[1,i].bar(bin_centers, psth_hz_calc[chosen_unit_idx, i+6], width=bin_width)
    axes[1,i].set_title(f'{directions[i+6]} deg', fontsize=15)

fig.text(0.05, 0.5, 'Firing Rate(Hz)', va='center', rotation='vertical', fontsize=15)
fig.text(0.5, 0.05, 'Time(sec)', ha='center', fontsize=15)

plt.show()

## ====================================================================
## ============== Q3: Orientation and direction tuning ================
## ====================================================================
#data prep
spikes_per_trial = np.sum(spikes_hist_counts, axis=3)
fire_rate_per_trial = spikes_per_trial / trial_duration
mean_values = np.mean(fire_rate_per_trial, axis=2)
std_values = np.std(fire_rate_per_trial, axis=2)
x_rad = np.radians(directions)
x_vec = np.linspace(0,2*np.pi, 500)

#helper functions:

#general fitting function:
def fit_and_evaluate(model_function, x_data, y_data, p0):
    popt, _ = curve_fit(model_function, x_data, y_data, p0, maxfev=10000)
    y_pred = model_function(x_data, *popt)
    rsme = np.sqrt(np.mean((y_data - y_pred)**2))
    return popt, rsme

#plot tuning fits for all 10 units.
def plot_tuning_fits(units_data, std_data, directions, best_models, title, x_vec):
    fig, axes = plt.subplots(2, 5, figsize=(18, 12), sharex=True)
    fig.suptitle(title, fontsize=20)

    x_deg_fine = np.degrees(x_vec)

    for unit in range(len(best_models)):
        ax = axes[unit//5, unit%5]

        ax.errorbar(directions, units_data[unit], yerr=std_data[unit], fmt='o', mfc='none', label='Data')

        y_fit = best_models[unit](x_vec)
        ax.plot(x_deg_fine, y_fit, label='Model Fit')
        ax.set_title(f'Unit #{unit+1}', fontsize=16)
        ax.set_xticks([0, 90, 180, 270, 360])

        if unit % 5 == 0:
            ax.set_ylabel("Rate [Hz]", fontsize=15)
        if unit >= 5:
            ax.set_xlabel("Direction [deg]", fontsize=15)

    handles = [plt.Line2D([], [], marker='o', linestyle='None', mfc='none', mec='C0', label='Data'),plt.Line2D([], [], color='C1', label='Model Fit')]
    fig.legend(handles = handles, loc='upper center',   bbox_to_anchor=(0.5, 0.93), ncol=2, fontsize=15, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    return fig

def run_tuning_analysis(unit_means, unit_stds, x_data, directions, direction_function, orientation_function, p0_dir, p0_ori, model_name, x_vec):
    best_models = []
    rmse_total = []
    print(f"\n {model_name} Fitting results")

    for unit in range(unit_means.shape[0]):
        y = unit_means[unit]

        p_direction, rmse_direction = fit_and_evaluate(direction_function, x_data, y, p0_dir)
        p_orientation, rmse_orientation = fit_and_evaluate(orientation_function, x_data, y, p0_ori)

        print(f"Unit {unit + 1}: Direction RMSE = {rmse_direction:.3f}, Orientation RMSE = {rmse_orientation:.3f}")

        if rmse_direction<rmse_orientation:
            best_function = lambda x, p=p_direction:direction_function(x, *p)
            rmse = rmse_direction
        else:
            best_function = lambda x, p=p_orientation:orientation_function(x, *p)
            rmse = rmse_orientation

        best_models.append(best_function)
        rmse_total.append(rmse)

    avg_rmse = np.mean(rmse_total)
    print(f"Total Average {model_name} RMSE: {avg_rmse:.3f}")
    plot_tuning_fits(unit_means, unit_stds, directions, best_models,f"Selectivity - {model_name} Fit", x_vec)

    plt.show()
    return avg_rmse

#model functions

#von mises models:
def von_mises_direction(x, A, k, PO):
    return A * np.exp(k * np.cos(x - PO))

def von_mises_orientation(x, A, k, PO):
    return A * np.exp(k * np.cos(2 * (x - PO)))

# Gaussian models:
def circular_dist(x, mu):
    return np.angle(np.exp(1j*(x-mu)))

def gaussian_direction(x, a, mu, sigma, offset):
    return a * np.exp(-circular_dist(x, mu)**2 / (2* sigma**2)) + offset

#2 peaks at 180 degree apart
def gaussian_orientation(x, a, mu, sigma, offset):
    peak1 = np.exp(-circular_dist(x, mu)**2 / (2 * sigma**2))
    peak2 = np.exp(-circular_dist(x, mu + np.pi)**2 / (2 * sigma**2))
    return a * (peak1 + peak2) + offset

#bonus models
def fuorier_direction(x, a0, a1, a2, a3, a4):
    return a0 + a1*np.cos(x) + a2*np.sin(x) + a3*np.cos(2*x) + a4*np.sin(2*x)

def fuorier_orientation(x, a0, a1, a2, a3, a4):
    return a0 + a1 * np.cos(2*x) + a2 * np.sin(2*x) + a3 * np.cos(4 * x) + a4 * np.sin(4 * x)

## PART A: Von Mises
p0_vm = [10, 1, np.pi]
avg_rmse_vm = run_tuning_analysis(mean_values, std_values, x_rad, directions, von_mises_direction, von_mises_orientation, p0_vm, p0_vm, "Von Mises", x_vec)

##PART B: Gaussian
p0_gauss = [10, np.pi, 0.5, 0]
avg_rmse_gauss = run_tuning_analysis(mean_values, std_values, x_rad, directions,gaussian_direction, gaussian_orientation, p0_gauss, p0_gauss, "Gaussian", x_vec)

##Bonus
po_fourier_bonus = [10, 1, 1, 0.5, 0.5]
avg_rmse_fourier = run_tuning_analysis(mean_values, std_values, x_rad, directions, fuorier_direction, fuorier_orientation, po_fourier_bonus, po_fourier_bonus, "Fourier", x_vec)

print(f"\nComparison: Von Mises RMSE = {avg_rmse_vm:.3f} vs Gaussian RMSE = {avg_rmse_gauss:.3f}")

## ====================================================================
## ==================== Q4: Statistical Analysis ======================
## ====================================================================

## PART A: Correlation Between Tuning Strength and Variability
print("\nQ4A: Pearson Correlation Results (Mean vs. STD)")

correlation_results = []

for unit in range(mean_values.shape[0]):
    mean_per_direction = mean_values[unit]
    std_per_direction = std_values[unit]

    r, p_val = pearsonr(mean_per_direction, std_per_direction)
    correlation_results.append((unit + 1, r, p_val))

    print(f"Unit {unit+1}: r = {r:.3f}, p-value = {p_val:.5f}")


## PART B: Hypotheses testing
print("\nQ4B: Paired t-test Results")

unit_idx = 4   # Unit 5, because it shows clear direction selectivity
dir1_idx = 2   # 60 degrees, strong response in Unit 5
dir2_idx = 8   # 240 degrees, opposite direction


data1 = fire_rate_per_trial[unit_idx, dir1_idx, :]
data2 = fire_rate_per_trial[unit_idx, dir2_idx, :]

# Perform paired t-test
t_stat, p_val_t = ttest_rel(data1, data2)

print(f"Comparing Unit #{unit_idx + 1}: {directions[dir1_idx]}° vs {directions[dir2_idx]}°")
print(f"t-statistic = {t_stat:.3f}, p-value = {p_val_t:.5f}")

if p_val_t < 0.05:
    print("Result is statistically significant (p < 0.05)")
else:
    print("Result is not statistically significant (p >= 0.05)")

