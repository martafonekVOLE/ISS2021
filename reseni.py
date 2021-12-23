from ctypes import sizeof
from math import dist, sqrt
from scipy.io import wavfile
from scipy.fft import fft
from scipy.signal import spectrogram
from scipy.signal import find_peaks
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

#========================================================
# Úkol 1
#========================================================

#načtení signálu
samples_freq, samples = wavfile.read("xpechm00.wav")

#vzorkovací frekvence
print ("Vzorkovaci frekvence:", samples_freq, "[Hz]")

#délka trvání signálu
amountOf_samples = len(samples)
timeIn_s = amountOf_samples / samples_freq

#počet vzorků signálu
amountOf_samples = len(samples)

#výpis vzorků
print("Počet vzorků signálu:", amountOf_samples, "[Vzorků]" )
print("Délka trvání signálu:", timeIn_s, "[s]")

#nejvyšší a nejnižší hodnota signálu
maxSample = -32768
minSample = 32767
for sample in samples:
    if sample > maxSample:
        maxSample = sample

for sample in samples:
    if sample < minSample:
        minSample = sample

print("Nejvyšší vzorek:", maxSample)
print("Nejmenší vzorek:", minSample)

#graf k zadanému signálu
textMaxMin = 'Max: %d\nMin: %d\n'%(maxSample, minSample)
x = np.arange(0, timeIn_s, timeIn_s/amountOf_samples)
y = samples
plt.figure(figsize=(10, 5))
plt.title("Vstupní signál")
plt.xlabel("Čas [t]")
plt.ylabel("Amplituda")
plt.grid(True)
plt.figtext(0.91, 0.45, textMaxMin, fontsize=7, color="blue", va="center")
plt.plot(x,y)
#plt.show()  #--UNCOMMENT

#========================================================
# Úkol 2
#========================================================
median = 0
median = np.median(samples) #np.mean -> průměr (0)
print("Střední hodnota: ", int(median))

#ustředění signálu 
consentratedSamples = []
for sample in samples:
    consentratedSamples.append(sample - int(median))

#normalizace signálu
absValue = maxSample - int(median)
if(((minSample-int(median)*(-1)) > maxSample)):
    absValue = minSample*(-1)                       #chyba?

normalizedSamples = []
for consSample in consentratedSamples:
    consSample = consSample/absValue
    normalizedSamples.append(consSample)
#ověření, že normalizovaný signál je v mezích -1 & 1
exitCode = 0
for consSample in normalizedSamples:
    if(consSample > 1):
        print("Chyba", consSample)
        exitCode = exitCode + 1
    elif(consSample < -1):
        print("Chyba", consSample)
        exitCode = exitCode + 1
    else:
        exitCode = exitCode + 0
if exitCode > 0:
    print("Celkový počet chyb:", exitCode)
else:
    print("Signál se podařilo úspěšně znormalizovat!")

#rozdělení na rámce
i = 0
i2 = 0
matrix = []
subMatrix = []
matrixOverflow = []
tmp = 0
x = 0

while True:
    if i == amountOf_samples:
        matrix.append(subMatrix)
        subMatrix = []
        matrixOverflow = []
        break

    if (len(matrix) == 0):
        if i2 < 1024:
            subMatrix.append(normalizedSamples[i])
            if i2 >= 512:
                matrixOverflow.append(normalizedSamples[i])
            i2 = i2 + 1
            i = i + 1
        else:
            matrix.append(subMatrix)
            subMatrix = []
            i2 = 0

    elif (len(matrix)!=0):
        if len(subMatrix) == 0:
            subMatrix.extend(matrixOverflow)
            matrixOverflow = []
            i2 = 512    
        if i2 < 1024:
            subMatrix.append(normalizedSamples[i])
            matrixOverflow.append(normalizedSamples[i])
            i2 = i2 + 1
            i = i + 1
        else:
           
            matrix.append(subMatrix)
            subMatrix = []
            i2 = 0
            x = x + 1
#nebo
#délka rámce = list(sf.blocks(".wav", blocksize, overflow))


#délka jednoho rámce
frameTimeIn_s = 1024/samples_freq
print(frameTimeIn_s)

#generování grafu hezkého rámce
x = np.arange(0, frameTimeIn_s, frameTimeIn_s/1024)
y = matrix[2]
plt.figure(figsize=(10, 5))
plt.title("Znělý signál - rámec 2")
plt.xlabel("Čas [t]")
plt.ylabel("Amplituda")
plt.grid(True)
plt.plot(x,y)

#plt.show()  #--UNCOMMENT

#========================================================
# Úkol 3
#========================================================
#testovací FFT funkce z knihovny scipy
testFFT = []
FFTresult = []
testFFT = fft(matrix[2])

#moje FFT funkce            TODO - je správně? Správný rámec? Co ukazovat má? Co je tam navíc (na začátku) Správné osy
N = 1024
n = np.arange(N)
k = n.reshape((N,1))
M = np.exp(-2j * np.pi * k * n / N)
FFTresult = np.dot(M, matrix[2])


if(np.allclose(FFTresult, testFFT) == True):
    print("Moje FFT se shoduje s scipy.fft!")
else:
    print("Moje FFT se neshoduje s scipy.fft!")

#graf k DFT
FFTresultToPrint = []
i = 1
while True:
    if i == 513:
        break
    FFTresultToPrint.append(FFTresult[i])
    i = i + 1

x = np.arange(0, samples_freq/2, samples_freq/2/512) #od nuly po FS/2 (1024 vzorků)
y = np.abs(FFTresultToPrint)   #má délku 1024 prvků FFTresult[:512]
plt.figure(figsize=(10, 5))
plt.title("Graf DFT - rámec 2")
plt.xlabel("Frekvence [Hz]")
plt.ylabel("Amplituda")
plt.grid(True)
plt.plot(x,y)
#plt.show()
#print(np.max(np.abs(np.real(testFFT))))

#========================================================
# Úkol 4
#========================================================
#spektrogram

normalizedSamplesSpectrum = np.array(normalizedSamples)

freq, time, spectro = spectrogram(normalizedSamplesSpectrum, samples_freq)

plt.figure(figsize=(10, 5))
plt.title("Spectrogram")
plt.grid(False)
plt.pcolormesh(time, freq, 10 * np.log10(spectro), shading='gouraud', cmap='jet')
plt.colorbar()
plt.ylabel('f[Hz]')
plt.xlabel('t[s]')
#plt.show()

#========================================================
# Úkol 5
#========================================================
#detekce rušivých signálů
disturbingOnes = []
disturbingFreq = []
i = 0
peaks, _ = find_peaks(np.abs(FFTresultToPrint), distance=50, height=2)
np.diff(peaks)

#výpis indexů rušivých signálů
print(peaks)

while True:
    if i == 4:
        break
    disturbingOnes.append(FFTresultToPrint[peaks[i]])
    i += 1
#výpis hodnot rušivých signálů
print(disturbingOnes)

i = 0
for p in peaks:
    disturbingFreq.append(peaks[i]*samples_freq/N)
    i += 1
#výpis rušivých frekvencí
print(disturbingFreq)

#vykreslení peaků
plt.figure(figsize=(10,5))
plt.plot(np.abs(FFTresultToPrint))
plt.plot(peaks, np.abs(FFTresultToPrint)[peaks], "x")
plt.show()

#ověření harmonické vztažnosti
i = 1
deviation = samples_freq / N

print("Nejmenší frekvence je:", disturbingFreq[0], "Hz")
while True:
    if i == 4:
        break
    if ((disturbingFreq[i]-(i*deviation))/disturbingFreq[0]):
        print("Frekvence", disturbingFreq[i], "Hz je harmonická!")
    else:
        print("Frekvence", disturbingFreq[i], "Hz je enharmonická!")
    i += 1
print("Při ověřování harmoničnosti je třeba brát v úvahu odchylku po DFT, která činí", deviation, "(Fs/N).")

#========================================================
# Úkol 6
#========================================================
#generování frekvencí

samplesArr = []
i = 0
#tvorba časových vzorků
while True:
    if i == amountOf_samples:
        break

    samplesArr.append(i/samples_freq)
    i += 1

#generování finální cosinusovky
i = range(4)
out_cos = []
finalCos = 0
for each in i:
    frq = disturbingFreq[each]
    out_cos.append(np.cos(2 * np.pi * frq * np.array(samplesArr)))
    finalCos += out_cos[each]

wavfile.write("audio/4cos.wav", samples_freq, finalCos.astype(np.int16))       #potichu, int8 lepší, ale nekvalitní
test,_ = wavfile.read("audio/4cos.wav")

if(test != 0): print("Signál složený z rušivých cosinusovek úspěšně vytvořen do audio/4cos.wav.")
else: print("Došlo k chybě při tvorbě signálu.")

#========================================================
# Úkol 6
#========================================================


