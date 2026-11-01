import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_control_mean_var_per_class(c_mean, c_var, classes, cdim, figsize=(8, 6)):
    c_std = c_var.sqrt()

    c_mean = c_mean.detach().cpu().numpy()
    c_std = c_std.detach().cpu().numpy()
    classes = classes.detach().cpu().numpy()

    num_classes = classes.shape[0]
    cmap = plt.get_cmap('winter', num_classes)
    colors = [cmap(i) for i in range(num_classes)]

    f, ax = plt.subplots(1, figsize=figsize)

    for i in range(classes.shape[0]):
        idx = classes[i].item()
        ax.errorbar(x=c_mean[idx], y=idx, xerr=c_std[idx], fmt='o', color=colors[i], capsize=5, markersize=8, label=f'{idx}')

    ax.set_title(f'control c{cdim} distribution (mean +/- std)', fontsize=14, weight='bold')
    ax.set_xlabel(f'c{cdim}', fontsize=12)
    ax.set_ylabel(f'class', fontsize=12)

    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    return f


def plot_control_trajectory(ck, ak, num_classes, cdim, ex, figsize=(8, 3)):
    ck = ck.detach().cpu().numpy()
    ak = ak.detach().cpu().numpy()
    x = np.arange(len(ck))

    cmap = plt.get_cmap('winter', num_classes)
    # colors = [cmap(i) for i in range(num_classes)]

    f, ax = plt.subplots(1, figsize=figsize)

    ax.plot(ck)
    ax.scatter(x, ck, c=ak, cmap=cmap, s=20, edgecolor='none')

    # Annotate each point with its class label
    for xi, yi, label in zip(x, ck, ak):
        ax.text(xi, yi, str(label), fontsize=6, color='black', ha='center', va='bottom') 

    # Set title and labels
    ax.set_title(f'control trajectory ex{ex}', fontsize=14, weight='bold')
    ax.set_xlabel(f'time frames', fontsize=12)
    ax.set_ylabel(f'c{cdim}', fontsize=12)

    ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()

    return f


def plot_spectral_centroid_with_spectrogram(audio, sr=None, n_fft=2048, hop_length=512, figsize=(10, 5)):
    """
    Plot spectral centroid over a log-amplitude spectrogram.
    """

    # Load audio
    if isinstance(audio, str):
        audio_path = audio
        y, sr = librosa.load(audio_path, sr=sr)
    else:
        y = audio
        assert sr is not None

    # Compute spectrogram (magnitude)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # Compute spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )[0]

    # Time axis
    times = librosa.times_like(
        spectral_centroids,
        sr=sr,
        hop_length=hop_length
    )

    fig, ax = plt.subplots(figsize=figsize)

    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="log",
        ax=ax
    )

    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    ax.plot(
        times,
        spectral_centroids,
        color="white",
        linewidth=2,
        label="Spectral Centroid"
    )

    ax.legend(loc="upper right")
    ax.set_title(f"Spectrogram with Spectral Centroid n_fft={n_fft} hop_length={hop_length}")

    fig.tight_layout()

    return fig


def plot_embeddings_pca(z, y, attr_name=""):
    classes, _ = torch.unique(y, return_counts=True)
    X = z - z.mean(dim=0, keepdim=True)
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    X2d = X @ Vt[:2].T  # (B, 2)

    fig, ax = plt.subplots(figsize=(7, 6))

    for cls in classes.tolist():
        mask = (y == cls)
        ax.scatter(
            X2d[mask, 0].cpu(),
            X2d[mask, 1].cpu(),
            s=20,
            alpha=0.6,
            label=f"class {cls}"
        )

    ax.set_title(f"Embedding distributions per class (PCA 2D) {attr_name}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.set_aspect("equal")

    plt.grid()
    plt.tight_layout()
    return fig


def plot_piano_roll_seconds(pm, fs=100):
    roll = pm.get_piano_roll(fs=fs)
    times = np.arange(roll.shape[1]) / fs

    plt.figure(figsize=(12, 4))
    plt.imshow(
        roll,
        aspect='auto',
        origin='lower',
        extent=[times[0], times[-1], 0, 128]
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("MIDI pitch")
    plt.title("Piano Roll (seconds)")
    plt.colorbar(label="Velocity")
    plt.show()


def plot_class_distribution_per_control_dim(ck, fk, num_classes, cdim, figsize=(8, 3), min_range = -20, max_range = 20):
    ck = ck.detach().cpu().numpy()
    fk = fk.detach().cpu().numpy()

    cmap = plt.get_cmap('winter', num_classes)
    colors = [cmap(i) for i in range(num_classes)]

    f, ax = plt.subplots(1, figsize=figsize)

    for k in range(num_classes):
        values = ck[fk == k]
        if len(values) < 2:
            continue  # KDE needs at least 2 points

        kde = gaussian_kde(values)
        x_grid = np.linspace(min_range, max_range, 500)
        ax.plot(x_grid, kde(x_grid), label=f"Class {k}", color=colors[k])
    
    # Set title and labels
    ax.set_title('control distributions per classes', fontsize=14, weight='bold')
    ax.set_xlabel(f'control dim c{cdim}', fontsize=12)
    ax.set_ylabel('density', fontsize=12)

    ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    return f
