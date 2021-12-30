import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from ctypes import sizeof
from math import dist, sqrt
from typing import final
from scipy.io import wavfile
from scipy.fft import fft
from scipy.signal import spectrogram, lfilter, filtfilt, find_peaks, buttord, butter
from scipy.signal.filter_design import freqz, tf2zpk

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
figureCounter = 1

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
plt.savefig('Figure_' + str(figureCounter) + '.png', dpi=400)
figureCounter += 1
#========================================================
# Úkol 2
#========================================================
median = 0
median = np.median(samples) #np.mean -> průměr (0) TODO headclass tvrdií mean, já myslel median
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

#generování grafu hezkého rámce
x = np.arange(0, frameTimeIn_s, frameTimeIn_s/1024)
y = matrix[35]
plt.figure(figsize=(10, 5))
plt.title("Znělý signál - rámec 2")
plt.xlabel("Čas [t]")
plt.ylabel("Amplituda")
plt.grid(True)
plt.plot(x,y)
plt.savefig('Figure_' + str(figureCounter) + '.png', dpi=400)
figureCounter += 1


#plt.show()  #--UNCOMMENT

#========================================================
# Úkol 3
#========================================================
#testovací FFT funkce z knihovny scipy
testFFT = []
FFTresult = []
testFFT = fft(matrix[35]) #TODO maybe 23?

#moje FFT funkce            TODO - je správně? Správný rámec? Co ukazovat má? Co je tam navíc (na začátku) Správné osy
N = 1024
n = np.arange(N)
k = n.reshape((N,1))
M = np.exp(-2j * np.pi * k * n / N)
FFTresult = np.dot(M, matrix[35])

#srovnání mojí DFT funkce s FFT funkcí
if(np.allclose(FFTresult, testFFT) == True):
    print("Moje FFT se shoduje s scipy.fft!")
else:
    print("Moje FFT se neshoduje s scipy.fft!")

#načtení periodického rámce do FFTresult
FFTresult = np.dot(M, matrix[2])
FFTresultToPrint = []
i = 1
while True:
    if i == 513:
        break
    FFTresultToPrint.append(FFTresult[i])
    i = i + 1

#graf k DFT - použití znělého rámce, tj. rámce číslo 8
x = np.arange(0, samples_freq/2, samples_freq/2/512) #od nuly po FS/2 (1024 vzorků)
y = np.abs(FFTresult[1:513])   #má délku 1024 prvků FFTresult[:512]
plt.figure(figsize=(10, 5))
plt.title("Graf DFT - rámec 2")
plt.xlabel("Frekvence [Hz]")
plt.ylabel("Amplituda")
plt.grid(True)
plt.plot(x,y)
plt.savefig('Figure_' + str(figureCounter) + '.png', dpi=400)
figureCounter += 1
#plt.show()
#print(np.max(np.abs(np.real(testFFT))))

#========================================================
# Úkol 4
#========================================================
#spektrogram

normalizedSamplesSpectrum = np.array(normalizedSamples)

freq, time, spectro = spectrogram(normalizedSamplesSpectrum, samples_freq)#, nperseg=1024, noverlap=512) TODO!!

plt.figure(figsize=(10, 5))
plt.title("Spectrogram")
plt.grid(False)
plt.pcolormesh(time, freq, 10 * np.log10(spectro), shading='gouraud', cmap='jet')
plt.colorbar()
plt.ylabel('f[Hz]')
plt.xlabel('t[s]')
plt.savefig('Figure_' + str(figureCounter) + '.png', dpi=400)
figureCounter += 1
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
print("Indexy jednotlivých rušivých frekvencí", peaks)

while True:
    if i == 4:
        break
    disturbingOnes.append(FFTresultToPrint[peaks[i]])
    i += 1

i = 0
for p in peaks:
    disturbingFreq.append(peaks[i]*samples_freq/N)
    i += 1
#výpis rušivých frekvencí
print("Hodnoty jednotlivých rušivých frekvencí:", disturbingFreq)

#vykreslení peaků
#plt.figure(figsize=(10,5))
#plt.plot(np.abs(FFTresultToPrint))
#plt.plot(peaks, np.abs(FFTresultToPrint)[peaks], "x")
#plt.show()

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


#generování spektrogramu finální cosinusovky            TODO je to tak? TODO upravit DFT -> graf
freq, time, spectro = spectrogram(finalCos, samples_freq)

plt.figure(figsize=(10, 5))
plt.title("Spectrogram výsledné cosinusovky")
plt.grid(False)
plt.pcolormesh(time, freq, 10 * np.log10(spectro), shading='gouraud', cmap='jet')
plt.colorbar()
plt.ylabel('f[Hz]')
plt.xlabel('t[s]')
plt.savefig('Figure_' + str(figureCounter) + '.png', dpi=400)
figureCounter += 1
#plt.show()
#========================================================
# Úkol 7
#========================================================
#pásmová zádrž jednotlivých rušivých frekvencí

#pásmová zádrž nejnižší rušívé frekvence
#jednotkový impuls
amountOfimpulses = 120
impuls = [1, *np.zeros(amountOfimpulses-1)]

#generování jednotlivých impulsních odezev
i = 0
generateNoiseFreeSignal = []
while True:
    if i == 4:
        break
    amountOf_impulses = amountOfimpulses - (i*15)
    impuls = [1, *np.zeros(amountOf_impulses-1)]

    firstOne, secondOne = buttord([(disturbingFreq[i]-50)/(samples_freq/2), (disturbingFreq[i]+50)/(samples_freq/2)], [(disturbingFreq[i]-15)/(samples_freq/2), (disturbingFreq[i]+15)/(samples_freq/2)], 3, 40, False)
    b, a = butter(firstOne, secondOne, 'bandstop', analog=False)

    #aplikace FIR/IIR filtru
    afterFiltering = filtfilt(b, a, impuls)  #TODO filtfilt vs lfilter

    #generování grafu
    plt.figure(figsize=(10,5))
    plt.grid()
    plt.stem(np.arange(amountOf_impulses), afterFiltering, basefmt=' ')
    plt.gca().set_xlabel('Počet vzorků [n]')
    plt.gca().set_title("Impulsní odezva {}. rušivé frekvence".format(i+1))
    plt.tight_layout()
    plt.savefig('Figure_' + str(figureCounter) + '.png', dpi=400)
    figureCounter += 1
    
    i += 1

#plt.show()

#========================================================
# Úkol 8
#========================================================
z = []
p = []
k = []
i = 0
while True:
    if i == 4:
        break

    firstOne, secondOne = buttord([(disturbingFreq[i]-50)/(samples_freq/2), (disturbingFreq[i]+50)/(samples_freq/2)], [(disturbingFreq[i]-15)/(samples_freq/2), (disturbingFreq[i]+15)/(samples_freq/2)], 3, 40, False)
    b, a = butter(firstOne, secondOne, 'bandstop', analog=False)
    helper1, helper2, helper3 = tf2zpk(b,a)

    z.append(helper1)
    p.append(helper2)
    k.append(helper3)
    i+=1

circle = np.linspace(0, 2*np.pi, 100)

_, ax = plt.subplots(2,2,figsize=(8,8))

i = 0
ax[0,0].plot(np.cos(circle), np.sin(circle))
ax[0,0].set_title("Nulové body a póly 1. frekvence")
ax[0,0].scatter(np.real(z[i]), np.imag(z[i]), marker='o', label='nuly')
ax[0,0].scatter(np.real(p[i]), np.imag(p[i]), marker='x', label='póly')
ax[0,0].set_ylabel("Imaginární složka")
ax[0,0].grid()
ax[0,0].legend(loc='upper left')

i += 1
ax[0,1].plot(np.cos(circle), np.sin(circle))
ax[0,1].set_title("Nulové body a póly 2. frekvence")
ax[0,1].scatter(np.real(z[i]), np.imag(z[i]), marker='o', label='nuly')
ax[0,1].scatter(np.real(p[i]), np.imag(p[i]), marker='x', label='póly')
ax[0,1].grid()
ax[0,1].legend(loc='upper left')

i += 1
ax[1,0].plot(np.cos(circle), np.sin(circle))
ax[1,0].set_title("Nulové body a póly 3. frekvence")
ax[1,0].scatter(np.real(z[i]), np.imag(z[i]), marker='o', label='nuly')
ax[1,0].scatter(np.real(p[i]), np.imag(p[i]), marker='x', label='póly')
ax[1,0].set_xlabel("Reálná složka")
ax[1,0].set_ylabel("Imaginární složka")
ax[1,0].grid()
ax[1,0].legend(loc='upper left')

i += 1
ax[1,1].plot(np.cos(circle), np.sin(circle))
ax[1,1].set_title("Nulové body a póly 4. frekvence")
ax[1,1].scatter(np.real(z[i]), np.imag(z[i]), marker='o', label='nuly')
ax[1,1].scatter(np.real(p[i]), np.imag(p[i]), marker='x', label='póly')
ax[1,1].set_xlabel("Reálná složka")
ax[1,1].grid()
ax[1,1].legend(loc='upper left')

plt.savefig('Figure_' + str(figureCounter) + '.png', dpi=400)
figureCounter += 1

#========================================================
# Úkol 9
#========================================================

w = []
h = []
helper1 = 0
helper2 = 0
i = 0

while True:
    if i == 4:
        break

    firstOne, secondOne = buttord([(disturbingFreq[i]-50)/(samples_freq/2), (disturbingFreq[i]+50)/(samples_freq/2)], [(disturbingFreq[i]-15)/(samples_freq/2), (disturbingFreq[i]+15)/(samples_freq/2)], 3, 40, False)
    b, a = butter(firstOne, secondOne, 'bandstop', analog=False)

    helper1, helper2 = freqz(b,a)
    w.append(helper1)
    h.append(helper2)
    i+=1

plt.figure(figsize=(10,5))
plt.title("Frekvenční charakteristika filtru")
plt.plot(w[0] / 2 / np.pi * samples_freq, np.abs(h[0]))
plt.plot(w[1] / 2 / np.pi * samples_freq, np.abs(h[1]))
plt.plot(w[2] / 2 / np.pi * samples_freq, np.abs(h[2]))
plt.plot(w[3] / 2 / np.pi * samples_freq, np.abs(h[3]))
plt.grid()
plt.savefig('Figure_' + str(figureCounter) + '.png', dpi=400)
figureCounter += 1

plt.show()
#========================================================
# Úkol 10
#========================================================

i = 0
generateNoiseFreeSignal = []

while True:
    if i == 4:
        break

    firstOne, secondOne = buttord([(disturbingFreq[i]-50)/(samples_freq/2), (disturbingFreq[i]+50)/(samples_freq/2)], [(disturbingFreq[i]-15)/(samples_freq/2), (disturbingFreq[i]+15)/(samples_freq/2)], 3, 40, False)
    b, a = butter(firstOne, secondOne, 'bandstop', analog=False)
    
    #aplikace FIR/IIR filtru
    afterFiltering = filtfilt(b, a, out_cos[i])  #TODO filtfilt vs lfilter

    generateNoiseFreeSignal.append(afterFiltering)

    i+=1

afterFiltering = filtfilt(b, a, samples)

#generateNoiseFreeSignal:   # cos1 - 43316 vzorků
                            # cos2 - 43316 vzorků
                            # cos3 - 43316 vzorků
                            # cos4 - 43316 vzorků

