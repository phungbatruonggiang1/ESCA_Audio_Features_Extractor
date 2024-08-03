from scipy.io.wavfile import read
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow
from spafe.utils.vis import show_features
from argparse import ArgumentParser

parser = ArgumentParser(description='program for running other processes')

# arguments that is needed for every type
parser.add_argument('-f', '--filePath', help='source to split', required=True)
args = parser.parse_args()

# # read audio
fpath = args.filePath
fs, sig = read(fpath)


# compute mfccs and mfes
imfccs  = gfcc(sig,
                fs=fs,
                pre_emph=1,
                pre_emph_coeff=0.97,
                window=SlidingWindow(0.03, 0.015, "hamming"),
                nfilts=128,
                nfft=2048,
                low_freq=0,
                high_freq=8000,
                normalize="mvn")


# visualize features
show_features(imfccs, "Gammatone Frequency Cepstral Coefﬁcients", "IMFCC Index","Frame Index")
