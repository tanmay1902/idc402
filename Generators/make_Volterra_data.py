"""
File to generate Volterra data series. 

Adapted from Hauser et al.
"""
from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from alive_progress import alive_bar

# Define the Gaussian kernel
def gauss_kernel_2D(time_step, t1, t2, mu, sigma):
    """
    Compute a 2D Gaussian kernel value at the given time coordinates.

    The 2D Gaussian is centered at (mu[0], mu[1]) with standard deviations 
    sigma[0] and sigma[1] along the t1 and t2 dimensions respectively.

    Formula:
        G(t1, t2) = exp(-[ (t1 - mu1)^2 / (2*sigma1^2) + (t2 - mu2)^2 / (2*sigma2^2) ])

    Args:
        time_step (float): Time step of the system (not used directly in calculation, kept for compatibility).
        t1 (float or np.ndarray): Time coordinate in the first dimension.
        t2 (float or np.ndarray): Time coordinate in the second dimension.
        mu (tuple or list of float): Mean values (mu1, mu2) for the Gaussian center.
        sigma (tuple or list of float): Standard deviations (sigma1, sigma2) along each dimension.

    Returns:
        float or np.ndarray: The computed 2D Gaussian kernel value(s).
    """
    
    return np.exp(-((t1 - mu[0])**2 / (2 * sigma[0]**2) + (t2 - mu[1])**2 / (2 * sigma[1]**2)))



def make_volterra_data(mu=[0.1, 0.1],sigma=[0.05, 0.05],dosave=False,filename=None,show=False):
    """
    Generate a nonlinear Volterra time series using a 2D Gaussian kernel.

    This function constructs a second-order Volterra series by convolving a kernel `h2`
    (generated using a 2D Gaussian function) with an input signal composed of a product
    of three sinusoids. It optionally saves the resulting dataset and plots.

    Args:
        mu (list of float, optional): Mean (μ₁, μ₂) of the 2D Gaussian kernel. Defaults to [0.1, 0.1].
        sigma (list of float, optional): Standard deviation (σ₁, σ₂) of the 2D Gaussian kernel. Defaults to [0.05, 0.05].
        dosave (bool, optional): Whether to save the dataset as a `.mat` file. Defaults to False.
        filename (str, optional): Base filename for saving `.mat` and `.png` files. Required if `dosave` or `show` is True.
        show (bool, optional): Whether to display and save plots of the kernel and signals. Defaults to False.

    Returns:
        dict: A dictionary containing the generated signals:
            - 'info': Description of the input signal.
            - 'u': Input signal (scaled and interpolated).
            - 'y': Raw output of Volterra convolution.
            - 'yn': Normalized output (zero mean, unit variance).

    Notes:
        - The kernel length is 0.2 seconds, and the signal length is 500 seconds.
        - Time step is 0.001 s for both kernel and signal.
        - Saves the 3D kernel plot and input/output signal if `show` is True.
        - Requires `filename` if saving or showing results.
    """
    
    # Parameters
    time_step = 0.001#0.001
    len_h = 0.2  # Length of the kernel in seconds
    t = np.linspace(0, len_h, int(len_h / time_step))

    h2 = np.zeros((len(t), len(t)))

    # Producing a discretized version of the kernel
    with alive_bar(len(t)**2) as tt:
        for i in range(len(t)):
            for j in range(len(t)):
                h2[i, j] = gauss_kernel_2D(time_step, t[i], t[j], mu, sigma)
                tt()

    # Plotting the kernel
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(t, t)
    ax.plot_surface(X, Y, h2, cmap='viridis')
    plt.savefig(f"gauss_kernel_{filename}.png",dpi=600)
    # plt.show()

    # Convolution with the kernel
    len_ = 500  # Length of the input signal in seconds
    t = np.linspace(0, len_, int(len_ / time_step))
    f1 = 2.11  # Hz
    f2 = 3.73
    f3 = 4.33

    # Input signal: product of sinusoids
    x = np.sin(2 * np.pi * f1 * t) * np.sin(2 * np.pi * f2 * t) * np.sin(2 * np.pi * f3 * t)

    # Add zeros to avoid index issues
    x = np.concatenate([np.zeros(int(len_h / time_step)), x])

    y = np.zeros(len(x))

    # Volterra loop
    leng = range(int(len_h / time_step),len(x))
    print(f'MU: {mu} | SG: {sigma}')
    with alive_bar(len(leng)) as bb:
        for i in leng:
            sum_1 = 0
            for m1 in range(int(len_h / time_step)):
                sum_2 = 0
                for m2 in range(int(len_h / time_step)):
                    sum_2 += h2[m1, m2] * x[i - m1] * x[i - m2]
                sum_1 += sum_2
            y[i] = sum_1
            bb()

    # Interpolate to get a time step of 1 ms
    interp_factor = int(time_step / 0.001)
    interp_func = interp1d(np.arange(len(y)), y, kind='linear')
    y_ = interp_func(np.linspace(0, len(y) - 1, len(y) * interp_factor))
    x_ = 0.2 * interp_func(np.linspace(0, len(x) - 1, len(x) * interp_factor))

    # Normalize output data
    scaler = StandardScaler()
    yn_ = scaler.fit_transform(y_.reshape(-1, 1)).flatten()

    # Pack data into a dictionary
    dat = {
        'info': 'three sinus',
        'u': x_,
        'y': y_,
        'yn': scaler.fit_transform(y_.reshape(-1, 1)).flatten()
    }

    
    if dosave:
        savemat(file_name=f"{filename}.mat",mdict=dat)
        print(f"{filename}.mat"," SAVED")

    if show:
        # Plot results
        t_ = np.linspace(0, len(x_) * 0.001, len(x_))
        plt.figure()
        plt.plot(t_, x_, label='input')
        plt.plot(t_, y_[:len(x_)], 'r', label='output')
        plt.title('Volterra series')
        plt.legend()
        plt.xlabel('time [s]')
        plt.ylabel('[ ]')
        plt.xlim([0, 200])
        plt.grid(True)
        plt.savefig(f"{filename}.png")
        print(f"{filename}.png SAVED")

import argparse

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate Volterra data with specified parameters.")
    
    # Define arguments
    parser.add_argument("--mu", type=float, nargs=2, default=[0.1, 0.1],
                        help="Mean values for data generation (two values).")
    parser.add_argument("--sigma", type=float, nargs=2, default=[0.05, 0.05],
                        help="Standard deviation values (two values).")
    parser.add_argument("--dosave", action="store_true",
                        help="Flag to save the data.")
    parser.add_argument("--filename", type=str, default=None,
                        help="Filename to save the data.")
    parser.add_argument("--show", action="store_true",
                        help="Flag to show the data visualization.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the function with parsed arguments
    make_volterra_data(mu=args.mu, sigma=args.sigma, dosave=args.dosave, 
                       filename=args.filename, show=args.show)
