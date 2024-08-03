import numpy as np
from spafe.utils.converters import mel2hz
from spafe.utils.vis import show_fbanks
from spafe.fbanks.mel_fbanks import mel_filter_banks

# init var
fs = 8000
nfilt = 7
nfft = 1024
low_freq = 0
high_freq = fs / 2

# compute freqs for xaxis
mhz_freqs = np.linspace(low_freq, high_freq, nfft //2+1)

for scale, label in [("constant", ""), ("ascendant", "Ascendant "), ("descendant", "Descendant ")]:
    # compute fbank
    mel_fbanks_mat, mel_freqs = mel_filter_banks(nfilts=nfilt,
                                                    nfft=nfft,
                                                    fs=fs,
                                                    low_freq=low_freq,
                                                    high_freq=high_freq,
                                                scale=scale)

    # visualize fbank
    show_fbanks(
        mel_fbanks_mat,
        [mel2hz(freq) for freq in mel_freqs],
        mhz_freqs,
        label + "Mel Filter Bank",
        ylabel="Weight",
        x1label="Frequency / Hz",
        x2label="Frequency / Mel",
        figsize=(14, 5),
        fb_type="mel")