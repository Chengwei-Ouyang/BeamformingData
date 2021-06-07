import os
import librosa
import soundfile as sf

audio_path = './clean_wav_32k/'
audio_list = os.listdir('./clean_wav_32k')
output_path = './clean_wav_32k/'

for audio in audio_list:
    print(audio)
    wavfile = librosa.load(audio_path+audio, sr=32000)
    sf.write(output_path+audio, wavfile[0], wavfile[1])