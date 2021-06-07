import os
import math
import random
import numpy as np
from numpy.linalg import norm
from scipy.io import wavfile
import pyroomacoustics as pra
import matplotlib.pyplot as plt

audio_list = os.listdir('clean_wav_32k')
# audio_file = audio_list[random.randint(0, len(audio_list) - 1)]

f = open('label.csv', 'w')
f.write('SNR, noise angle, speech angle, reveberation time \n')

for (i, path) in enumerate(audio_list):
    # dimensions of the room
    room_dim = [8, 8, 3]  # meters

    fs, audio = wavfile.read('./clean_wav_32k/' + path)
    fs, noise = wavfile.read('noise_only_32k.wav')

    noise_start = random.randint(0, noise.shape[0] - audio.shape[0] - 1)
    noise = noise[noise_start:(noise_start + audio.shape[0])]

    # random SNR from 0-20dB
    snr = random.random() * 20

    noise = noise / norm(noise) * norm(audio) / (10.0 ** (0.05 * snr))

    # define the locations of the microphones
    mic_locs = np.c_[
        [3.95, 4, 1.5], [4.05, 4, 1.5],  # mic 1  # mic 2
    ]

    # random angle for noise and speech
    noise_angle = random.random() * math.pi
    speech_angle = random.random() * math.pi

    mic_center = [4, 4, 1.5]
    noise_loc = [mic_center[0] + math.cos(noise_angle), mic_center[0] + math.sin(noise_angle), 1.5]
    speech_loc = [mic_center[0] + math.cos(speech_angle), mic_center[0] + math.sin(speech_angle), 1.5]

    # random reverberation time from 0.2s to 0.6s
    rt60_tgt = random.random() * 0.4 + 0.2

    print(i, snr, noise_angle, speech_angle, rt60_tgt)
    f.write(str(snr) + ', ' + str(noise_angle) + ', ' + str(speech_angle) + ', ' + str(rt60_tgt) + '\n')

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

    room = pra.ShoeBox(
                room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
            )

    # place the source in the room
    room.add_source(speech_loc, signal=audio, delay=0)
    room.add_source(noise_loc, signal=noise, delay=0)

    # finally place the array in the room
    room.add_microphone_array(mic_locs)

    # Run the simulation (this will also build the RIR automatically)
    room.simulate()
    room.mic_array.to_wav(
        './noisy_wav_32k/' + path,
        norm=True,
        bitdepth=np.int16,
    )