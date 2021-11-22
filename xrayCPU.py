from numpy import sqrt, exp, pi
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from numba import jit, prange
import numpy as np

start = timer()

Wavelength = 0.56
K = 2 * pi / Wavelength
NumberOfSlits = 10
SlitLength = 50
SourcesPerSlit = 50
ApertureLength = (2*NumberOfSlits - 1) * SlitLength
SourceSpacing = SlitLength / SourcesPerSlit
NumObservationPoints = 2000000
GratingLength = 1e7
NumSpacing = GratingLength / NumObservationPoints
D2 = 5e7
D3 = 10e7
InitialAmplitude = 1
InitialPhase = 1
Mu = 2.203e-5
Thickness = 1000
N0 = 1
N1 = 1-0.000141727127
y = np.array([-ApertureLength/2 + i*SlitLength + SourceSpacing/2 + j*SourceSpacing for i in range(2*NumberOfSlits-1) for j in range(SourcesPerSlit)])
o = np.array([-GratingLength/2 + NumSpacing/2 + j*NumSpacing for j in range(NumObservationPoints)])
u1 = np.zeros((2*NumberOfSlits-1) * SourcesPerSlit)
i2 = np.zeros(NumObservationPoints)

@jit(nopython=True, parallel=True, fastmath=True)
def calcs(y, o, U1, I2, wavelength, numberOfSlits, slitLength, sourcesPerSlit, numObservationPoints, gratingLength, d2, initialAmplitude, initialPhase, mu, thickness, n0, n1):
    k = 2 * pi / wavelength
    sourceSpacing = slitLength / sourcesPerSlit
    wavelength1 = n0 * wavelength / n1
    
    for i in prange(numObservationPoints):
        amplitudeSum = 0
        for j in prange((2*numberOfSlits-1) * sourcesPerSlit):
            if (j//sourcesPerSlit) % 2 == 0:
                U1[j] = initialAmplitude
                r = sqrt((d2)**2 + (o[i] - y[j])**2)
                phase = exp(1j * k * r) * initialPhase
                U = initialAmplitude * phase / r
                amplitudeSum = amplitudeSum + U * sourceSpacing 
            else:
                initialPhase1 = initialPhase * exp(1j * (thickness / wavelength1) * 2 * pi)
                initialAmplitude1 = initialAmplitude * exp(-mu * thickness)
                U1[j] = initialAmplitude1
                r = sqrt((d2)**2 + (o[i] - y[j])**2)
                phase = exp(1j * k * r) * initialPhase1
                U = initialAmplitude1 * phase / r
                amplitudeSum = amplitudeSum + U * sourceSpacing
        I2[i] = amplitudeSum.real ** 2 + amplitudeSum.imag ** 2 
    return U1,I2

U1,I2 = calcs(y, o, u1, i2, Wavelength, NumberOfSlits, SlitLength, SourcesPerSlit, NumObservationPoints, GratingLength, D2, InitialAmplitude, InitialPhase, Mu, Thickness, N0, N1)

end = timer()
print("Time for program to run =", end - start, "seconds.")

N = 20
label1 = "Number of slits = " + str(NumberOfSlits) + "\n" + "Slit length = " + str(SlitLength) + " nm" + "\n" + "Source points per slit = " + str(SourcesPerSlit) + "\n" + "Grating pitch = " + str(SlitLength*2) + " nm" + "\n" + "Aperture diameter = " + str(ApertureLength) + " nm" + "\n" + "Grating length = " + str(GratingLength) + " nm" + "\n" + "Grating thickness = " + str(Thickness) + " nm" + "\n" + "Wavelength = " + str(Wavelength) + " nm" + "\n" + "Index of refraction of x-rays in si3n4 = " + str(N1) + "\n" + "Attenuation coefficient = " + str(Mu) + "\n" + "Elapsed time = " + str(end-start) + " s"
label2 = "Number of slits = " + str(NumberOfSlits) + "\n" + "Slit length = " + str(SlitLength) + " nm" + "\n" + "Source points per slit = " + str(SourcesPerSlit) + "\n" + "Grating pitch = " + str(SlitLength*2) + " nm" + "\n" + "Aperture diameter = " + str(ApertureLength) + " nm" + "\n" + "Grating length = " + str(GratingLength) + " nm" + "\n" + "Grating thickness = " + str(Thickness) + " nm" + "\n" + "Number of observation points = " + str(NumObservationPoints) + "\n" + "Wavelength = " + str(Wavelength) + " nm" + "\n" + "Distance from grating 1 = " + str(D2) + " nm" + "\n" + "Index of refraction of x-rays in si3n4 = " + str(N1) + "\n" + "Attenuation coefficient = " + str(Mu) + "\n" + "Elapsed time = " + str(end-start) + " s"

fig = plt.figure()
plt.plot(y, np.square(U1), 'k', markersize=N//5, label=label1)
plt.xlabel('y-Position on Aperture (nm)', fontsize=1.5*N)
plt.ylabel('Initial Amplitude', fontsize=1.5*N)
plt.title('Intensity Distribution of X-rays on the back side of the First Grating', fontsize=2*N)
plt.legend(loc="upper left", fontsize=1.5*N)
fig.set_size_inches(2*N,1*N)
plt.xticks(fontsize=1.5*N)
plt.yticks(fontsize=1.5*N)
plt.gca().xaxis.get_offset_text().set_fontsize(1.5*N)
plt.gca().yaxis.get_offset_text().set_fontsize(1.5*N)
plt.savefig('1_Speed2/xrayFirstGrating.png')
plt.close(fig)

fig = plt.figure()
plt.plot(o, I2, 'black', markersize=1, label=label2)
plt.xlabel('y-Position on Second Grating (nm)', fontsize=1.5*N)
plt.ylabel('Intensity (lux)', fontsize=1.5*N)
plt.title('Intensity Distribution of X-rays on the front side of the Second Grating', fontsize=2*N)
plt.legend(loc="upper left", fontsize=1.5*N)
fig.set_size_inches(2*N,1*N)
plt.xticks(fontsize=1.5*N)
plt.yticks(fontsize=1.5*N)
plt.gca().xaxis.get_offset_text().set_fontsize(1.5*N)
plt.gca().yaxis.get_offset_text().set_fontsize(1.5*N)
plt.savefig('1_Speed2/xraySecondGrating.png')
plt.close(fig)