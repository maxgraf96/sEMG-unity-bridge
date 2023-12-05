import numpy as np
import pywt
import scipy.signal as signal
import zmq

from hyperparameters import DATA_LEN

# Bandpass self.data from 10-25 Hz
fs = 50  # Sampling frequency
fmin = 20  # Minimum frequency to pass
fmax = 25  # Maximum frequency to pass
nyq = 0.5 * fs  # Nyquist frequency

# Define filter parameters
# b, a = signal.butter(4, [fmin / nyq, fmax / nyq], btype='band')


def power_spectral_density(emg_data, fs=1000):
    psd = []
    for channel_data in emg_data.T:
        f, pxx = signal.welch(channel_data, fs, nperseg=50)
        psd.append(pxx)
    return np.array(psd)


def median_frequency(psd, f):
    mdf = []
    for channel_psd in psd:
        cumsum = np.cumsum(channel_psd)
        median_idx = np.where(cumsum >= cumsum[-1] / 2)[0][0]
        mdf.append(f[median_idx])
    return np.array(mdf)


def mean_frequency(psd, f):
    mnf = []
    for channel_psd in psd:
        mean_freq = np.sum(channel_psd * f) / np.sum(channel_psd)
        mnf.append(mean_freq)
    return np.array(mnf)


def peak_frequency(psd, f):
    pf = []
    for channel_psd in psd:
        max_idx = np.argmax(channel_psd)
        pf.append(f[max_idx])
    return np.array(pf)


def extract_frequency_features(emg_data, fs=1000):
    psd = power_spectral_density(emg_data, fs)
    f = np.linspace(0, fs / 2, len(psd[0]))

    mdf = median_frequency(psd, f)
    mnf = mean_frequency(psd, f)
    pf = peak_frequency(psd, f)

    return mdf, mnf, pf


def preprocess_sample(sample, wrist_angles=None):
    # =============================================================================
    # Preprocessing
    # =============================================================================

    coefficients_level4 = pywt.wavedec(sample, 'db2', 'smooth', level=4, axis=0)
    cA, cD = coefficients_level4[0], coefficients_level4[1]
    # Concat cA and cD
    wavelet_features = np.concatenate((cA, cD), axis=0)

    # Get features
    # Time domain features
    # Mean absolute value
    mean_abs = np.mean(np.abs(sample), axis=0).reshape(1, -1)
    # Root mean square
    rms = np.sqrt(np.mean(np.square(sample), axis=0)).reshape(1, -1)
    # Variance
    var = np.var(sample, axis=0).reshape(1, -1)

    mdf, mnf, pf = extract_frequency_features(sample, fs=50)
    mdf = mdf.reshape(1, -1)
    mnf = mnf.reshape(1, -1)
    pf = pf.reshape(1, -1)

    # Prepend features to sample
    # This way for label only works in finetuning mode!!
    wrist_angle = np.zeros(sample.shape)
    if wrist_angles is not None:
        # Make array of zeros for wrist angle
        wrist_angle[:, :3] = wrist_angles

    sample = np.concatenate((mean_abs, rms, var, mdf, mnf, pf, wavelet_features, sample, wrist_angle), axis=0)

    # replace all nan values with 0
    sample = np.nan_to_num(sample, copy=False)

    # Convert to float32
    sample = sample.astype(np.float32)

    return sample


def my_function():
    print("Python function called", flush=True)


if __name__ == '__main__':
    dummy = np.random.rand(DATA_LEN, 8)

    # Create a ZeroMQ context
    context = zmq.Context()

    # Create a socket and bind it to an address
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    while True:
        # Wait for a request
        request = socket.recv()

        # Process the request
        arg = request.decode()
        ls = eval(arg)
        arr = np.array(ls).reshape(-1, 8)

        out = str(preprocess_sample(arr).tolist())

        # Send the response back
        socket.send(out.encode())
