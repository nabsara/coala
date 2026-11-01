import numpy as np
import librosa
import torch
import pretty_midi
import pyloudnorm as pyln
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from scipy.ndimage import uniform_filter1d

from .timbral_models import timbral_booming, timbral_sharpness


def _spectral_descriptor_fns(roll_percent: float):
    return {
        "centroid": librosa.feature.spectral_centroid,
        "bandwidth": librosa.feature.spectral_bandwidth,
        "rolloff": lambda **kw: librosa.feature.spectral_rolloff(
            roll_percent=roll_percent, **kw
        ),
    }


def _filter_descriptor(
        x: np.ndarray,
        energy: np.ndarray | None,
        energy_thresh: float | None,
        median: bool,
        median_kernel: int,
        clip_percentiles: tuple,
        normalize: str | None,
    ):
    if energy is not None and energy_thresh is not None:
        x = np.where(energy >= energy_thresh, x, np.nan)

    if median:
        k = median_kernel if median_kernel % 2 == 1 else median_kernel + 1
        x = medfilt(
            np.nan_to_num(x, nan=np.nanmedian(x)),
            kernel_size=k
        )

    low, high = np.nanpercentile(x, clip_percentiles)
    x = np.clip(x, low, high)

    if normalize == "log":
        x = np.log1p(np.maximum(x, 0.0))
    elif normalize == "zscore":
        mu = np.nanmean(x)
        sigma = np.nanstd(x) + 1e-8
        x = (x - mu) / sigma

    return x


def _resample_to_latent(x, sr, hop_length, ae_ratio, z_length):
    t_latent = (np.arange(z_length) * ae_ratio) / sr
    t_x = librosa.times_like(x, sr=sr, hop_length=hop_length)

    valid = t_x <= t_latent[-1]
    x = x[valid]
    t_x = t_x[valid]

    interp = interp1d(
        t_x,
        x,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    return interp(t_latent)


def compute_latent_loudness(y, sr, z_length, compression=4096):
    loudness_vals = []
    half = compression // 2

    for i in range(z_length):
        # centered window
        center = i * compression + half
        start = max(0, center - half)
        end = min(len(y), center + half)

        window = y[start:end]

        if len(window) == 0 or np.max(np.abs(window)) < 1e-5:
            loudness = -80.0
        else:
            rms = np.sqrt(np.mean(window**2) + 1e-12)
            loudness = 20 * np.log10(rms)

            if not np.isfinite(loudness):
                loudness = -80.0

        loudness = np.clip(loudness, -80.0, 0.0)

        loudness_vals.append(loudness)

    loudness_vals = np.array(loudness_vals, dtype=np.float32)

    # slight smoothing
    loudness_vals = uniform_filter1d(loudness_vals, size=3, mode="nearest")

    return loudness_vals


def compute_melody_from_midi(
        pm: pretty_midi.PrettyMIDI,
        ae_ratio: int,
        sr: int,
        z_length: int,
    ):
    latent_hop = ae_ratio / sr
    min_overlap = latent_hop / 2.0

    melody = np.full(z_length, -1, dtype=np.int32)

    # collect notes
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        notes.extend(inst.notes)

    if not notes:
        return melody

    for i in range(z_length):
        # centered window
        center = i * latent_hop + latent_hop / 2
        t0 = center - latent_hop / 2
        t1 = center + latent_hop / 2

        overlaps = {}

        for note in notes:
            # skip invalid notes
            if note.end <= note.start:
                continue

            overlap = max(0.0, min(note.end, t1) - max(note.start, t0))

            if overlap >= min_overlap:
                # optional: velocity weighting
                weight = overlap * (note.velocity / 127.0)
                overlaps[note.pitch] = overlaps.get(note.pitch, 0.0) + weight

        if overlaps:
            melody[i] = max(overlaps.items(), key=lambda x: x[1])[0]

    return melody


def process_melody(melody, default_midi_note=60):
    last_note = default_midi_note
    new_melody = melody.copy()
    for i in range(melody.shape[-1]):
        current_note = melody[i]
        if current_note > 0:
            last_note = current_note
        new_melody[i] = last_note
    return new_melody


def compute_active_onset(melody: np.ndarray):
    """
    melody: array with MIDI pitch or -1 for silence

    returns:
        active: 0/1
        onset:  0/1
    """
    active = (melody >= 0).astype(np.float32)

    onset = np.zeros_like(active, dtype=np.float32)

    for i in range(len(melody)):
        if active[i] == 1:
            if i == 0:
                onset[i] = 1.0
            else:
                # onset if:
                # - previous was silence OR
                # - pitch changed
                if active[i-1] == 0 or melody[i] != melody[i-1]:
                    onset[i] = 1.0

    return active, onset


def midi_to_pitch_octave(midi):
    """
    Returns 
    - pitch class index [0..11] (all semitones)
    - octave number C1 (MIDI 24) -> octave 1
    or -1 for silence.
    """
    midi = np.asarray(midi)
    pitch = np.full_like(midi, -1, dtype=np.int32)
    octave = np.full_like(midi, -1, dtype=np.int32)
    
    voiced = midi >= 0
    pitch[voiced] = midi[voiced] % 12
    octave[voiced] = (midi[voiced] // 12) - 1

    return pitch, octave


def midi_to_instrument(midi, instrument_value):
    """
    Returns instrument aligned with midi
    with -1 for silence.
    """
    midi = np.asarray(midi)
    instrument = np.full_like(midi, -1, dtype=np.int32)
    
    voiced = midi >= 0
    instrument[voiced] = instrument_value

    return instrument


def compute_timbral(
    y: np.ndarray,
    sr: int,
    ae_ratio: int,
    z_length: int,
    descriptors: list,
    resampler=None,
    mean: bool = False,
) -> dict:
    """
    Compute global timbral descriptors and duplicate over latent frames.

    Returns
    -------
    dict[str, np.ndarray]
        Each feature has shape (z_length,)
    """

    features_dict = {
        "booming": timbral_booming,
        "sharpness": timbral_sharpness,
    }

    active = [d for d in descriptors if d in features_dict]
    if not active:
        return {}

    # --- resample to 44.1kHz (required) ---
    if sr != 44100:
        if resampler is None:
            y = librosa.resample(y, orig_sr=sr, target_sr=44100)
        else:
            y = resampler(torch.tensor(y)).numpy()
        sr = 44100

    features = {}

    for name in active:
        try:
            v = features_dict[name](y, fs=sr, mean=mean).astype(np.float32)
        except Exception:
            if mean:
                v = 0.0
            else:
                # fallback: zero sequence with correct length (~internal frames)
                n_frames = int(np.ceil(len(y) / 4096))
                v = np.zeros(n_frames, dtype=np.float32)

        if mean:
            # --- duplicate over latent frames ---
            features[name] = np.full((z_length,), v, dtype=np.float32)
        else:
            v_latent = _resample_to_latent(
                v,
                sr=sr,
                hop_length=4096,
                ae_ratio=ae_ratio,
                z_length=z_length,
            ).astype(np.float32)
            
            v_latent = medfilt(v_latent, kernel_size=9)
            features[name] = v_latent

    return features


def compute_audio_descriptors(
    y: np.ndarray,
    sr: int,
    ae_ratio: int,
    z_length: int,
    descriptors: list[str],
    roll_percent: float = 0.85,
    energy_percentile: float = 10.0,
    median: bool = True,
    median_kernel: int = 3,
    clip_percentiles: tuple = (5.0, 95.0),
    normalize: str | None = None,
    instrument_val: int | None = None,
    pm: pretty_midi.PrettyMIDI | None = None,
    device: str = "cpu",
) -> dict:
    """
    Computes descriptors including:
    - rms
    - spectral (raw + filtered) (centroid, bandwidth, rolloff)
    - instrument
    """

    n_fft = ae_ratio
    hop_length = n_fft // 2
    latent_hop_sec = ae_ratio / sr

    features = {}

    # Energy 
    energy = librosa.feature.rms(
        y=y,
        frame_length=n_fft,
        hop_length=hop_length,
        center=True,
    )[0]

    energy_thresh = np.percentile(energy, energy_percentile)

    energy_latent = librosa.feature.rms(
        y=y,
        frame_length=ae_ratio,
        hop_length=ae_ratio,
        center=False,
    )[0][:z_length]
    
    # RMS
    if "rms" in descriptors:
        rms_db = librosa.amplitude_to_db(energy_latent, ref=np.max)
        features["rms"] = rms_db.astype(np.float32)
    
    if "loudness" in descriptors:
        features["loudness"] = compute_latent_loudness(y, sr, z_length, compression=ae_ratio)

    # Spectral (centroid, bandwidth, rolloff)
    spectral_fns = _spectral_descriptor_fns(roll_percent)

    for desc in descriptors:
        is_filtered = desc.endswith("-filt")
        base = desc.replace("-filt", "")

        if base not in spectral_fns:
            continue

        x = spectral_fns[base](
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            center=True,
        )[0]

        if is_filtered:
            x = _filter_descriptor(
                x,
                energy,
                energy_thresh,
                median,
                median_kernel,
                clip_percentiles,
                normalize,
            )

        features[desc] = _resample_to_latent(
            x, sr, hop_length, ae_ratio, z_length
        ).astype(np.float32)

    if any(d in ["booming", "sharpness"] for d in descriptors):
        timbral_feats = compute_timbral(
            y=y,
            sr=sr,
            ae_ratio=ae_ratio,
            z_length=z_length,
            descriptors=descriptors,
            mean=False,
        )
        features.update(timbral_feats)

    if "instrument" in descriptors: 
        features["instrument"] = np.full((z_length,), instrument_val).astype(np.int32)
    
    if "melody" in descriptors:
        if pm is None:
            raise ValueError("melody descriptor requires a PrettyMIDI instance")

        melody = compute_melody_from_midi(
            pm=pm,
            ae_ratio=ae_ratio,
            sr=sr,
            z_length=z_length,
        ).astype(np.int32)
        features["melody"] = melody

        active, onset = compute_active_onset(melody)

        if "active" in descriptors:
            features["active"] = active.astype(np.int32)

        if "onset" in descriptors:
            features["onset"] = onset.astype(np.int32)

        melody_processed = process_melody(melody, default_midi_note=60)
        features["melody-processed"] = melody_processed

        pitch, octave = midi_to_pitch_octave(melody_processed)
        if "pitch" in descriptors:
            features["pitch"] = pitch
        if "octave" in descriptors:
            features["octave"] = octave

    return features
