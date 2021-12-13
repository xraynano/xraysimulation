from numpy import sqrt, exp, pi, complex128
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from numba import jit, prange
import numpy as np
import math

start = timer()

# --Customizable Parameters--
NumberOfSlits = 7 ## Number of slits on first grating (Default 100,000, but too slow to run).
SlitLength = 50 # Length of the slit (nm) (Default 50 nm).
SourcesPerSlit = 50 # Sources per slit (Default 50).
NumObservationPoints2 = 10000 # Number of observation points in second grating (Default 10,000,000 but too slow to run).
GratingLength = 1e7 # Length of second grating (nm) (Default 1e7 nm).
D = 5e7 # Distance between gratings (nm) (Default 5e7 nm).
InitialAmplitude = 1 # Initial amplitude of the incoming plane wave (Default 1).
InitialPhase = 1 # Initial phase of the incoming plane wave (Default 1).
Mu = 2.203e-5 # Attenuation coefficient of x-rays in silcion nitride
Thickness = 1500 # Thickness of each grating (nm) (No default yet).
N0 = 1 # Index of refraction of x-rays in a vacuum (Default 1)
N1 = 1-0.000141727127 # Index of refraction of x-rays in silicon nitride.
Wavelength = 0.56 # Wavelength of x-rays in vacuum (nm) (Default 0.56 nm).

# --Results from Parameters--
ApertureLength = (2*NumberOfSlits - 1) * SlitLength # Aperture length of first grating (nm).
SourceSpacing = SlitLength / SourcesPerSlit # Spacing between source points (nm).
NumSpacing2 = GratingLength / NumObservationPoints2 # Spacing between observation points in second grating (nm).
Wavelength1 = N0 * Wavelength / N1 # Wavelength of x-rays in silicon nitride (nm).
K = 2 * pi / Wavelength # Wavenumber.
# Y is the list of y-positions of the source points on the first grating.
Y = np.array([-ApertureLength/2 + i*SlitLength + SourceSpacing/2 + j*SourceSpacing for i in range(2*NumberOfSlits-1) for j in range(SourcesPerSlit)])
# O2 is the list of y-positions of the observation points in the front of the second grating.
O2 = np.array([-GratingLength/2 + NumSpacing2/2 + j*NumSpacing2 for j in range(NumObservationPoints2)])
U1 = np.zeros((2*NumberOfSlits-1) * SourcesPerSlit) # Initializing amplitudes of the source points in the back of first grating.
# P1Slit and P1Closed are the phase changes of the source points of the slits and closed parts as it travels grating thickness from the first grating.
P1Slit = np.zeros((2*NumberOfSlits-1) * SourcesPerSlit, dtype=complex128) # Initializing phases of the source points of slits in the back of first grating.
P1Closed = np.zeros((2*NumberOfSlits-1) * SourcesPerSlit, dtype=complex128) # Initializing phases of the source points of closed part in the back of first grating.
I2 = np.zeros(NumObservationPoints2) # Initializing intensity distribution of the front of second grating.

# --Functions--
@jit(nopython=True, parallel=True, fastmath=True)
def calcs(y, o, u1, P1Slit, P1Closed, I2, wavelength, wavelength1, numberOfSlits, sourceSpacing, slitLength, sourcesPerSlit, numObservationPoints, d, k, initialAmplitude, initialPhase, mu, thickness, n0, n1):
    '''Function returns the intensity distribution of the source points in the back of the first grating and also
    returns the intensity distribution of the observation points in the front of the second grating.'''
    for i in prange(numObservationPoints):
        amplitudeSum = 0 # Sum of intensities of each spherical source point on first grating incident to an observation point on second grating.
        for j in prange((2*numberOfSlits-1) * sourcesPerSlit):
            if (j//sourcesPerSlit) % 2 == 0: # Source point on first grating lies in a slit.
                u1[j] = initialAmplitude # Amplitude of source point on the back side of the first grating
                initialPhase1 = initialPhase * exp(1j * (thickness / wavelength) * 2 * pi)
                P1Slit[j] = initialPhase1
                r = sqrt(d**2 + (o[i] - y[j])**2) # Distance from source point on first grating to observation point on second grating.
                phase = exp(1j * k * r) * initialPhase1
                # phase = exp(1j * k * r) * initialPhase # Complex phase of spherical source point at a certain  observation point.
                U = initialAmplitude * phase / r
                amplitudeSum = amplitudeSum + U * sourceSpacing 
            else: # Source point on first grating lies on a closed part
                initialPhase1 = initialPhase * exp(1j * (thickness / wavelength1) * 2 * pi) # Phase change as x-ray traverses through closed part of first grating
                P1Closed[j] = initialPhase1
                initialAmplitude1 = initialAmplitude * exp(-mu * thickness) # Amplitude reduction as x-ray traverses through closed part of first grating
                u1[j] = initialAmplitude1 # Amplitude of source point on the back side of the first grating
                r = sqrt(d**2 + (o[i] - y[j])**2) # Distance from source point on first grating to observation point on second grating.
                phase = exp(1j * k * r) * initialPhase1 # Complex phase of spherical source point at a certain observation point.
                U = initialAmplitude1 * phase / r # Complex amplitude of spherical source point at a certain observation point.
                amplitudeSum = amplitudeSum + U * sourceSpacing
        I2[i] = amplitudeSum.real ** 2 + amplitudeSum.imag ** 2 # Intensity of observation point on second grating
    return u1, P1Slit, P1Closed, I2
U1, P1Slit, P1Closed, I2 = calcs(Y, O2, U1, P1Slit, P1Closed, I2, Wavelength, Wavelength1, NumberOfSlits, SourceSpacing, SlitLength, SourcesPerSlit, NumObservationPoints2, D, K, InitialAmplitude, InitialPhase, Mu, Thickness, N0, N1)

end = timer()
print("Time for program to run =", end - start, "seconds.")

# --Plots--
N = 20
P1Slit = list(set([x for x in P1Slit if x != 0]))[0]
P1Closed = list(set([x for x in P1Closed if x != 0]))[0]
phaseSlit = math.atan(P1Slit.imag / P1Slit.real)
phaseClosed = math.atan(P1Closed.imag / P1Closed.real)
if P1Slit.real < 0:
    phaseSlit = phaseSlit + pi
if P1Closed.real < 0:
    phaseClosed = phaseClosed + pi
phaseDifference = abs(phaseClosed - phaseSlit) # Phase shift between x-rays through closed part and x-rays through slits at the back of the first grating.

label1 = "Number of slits = " + str(NumberOfSlits) + "\n" + "Slit length = " + str(SlitLength) + " nm" + "\n" + "Source points per slit = " + str(SourcesPerSlit) + "\n" + "Grating pitch = " + str(SlitLength*2) + " nm" + "\n" + "Aperture diameter = " + str(ApertureLength) + " nm" + "\n" + "Length of second grating = " + str(GratingLength) + " nm" + "\n" + "Grating thickness = " + str(Thickness) + " nm" + "\n" + "Amplitude of source point in slit = " + str(list(set(U1))[1]) + "\n" + "Amplitude of source point through closed part = " + str(list(set(U1))[0]) + "\n" + "Index of refraction of x-rays in si3n4 = " + str(N1) + "\n" + "Attenuation coefficient = " + str(Mu) + "\n" + "Wavelength of x-rays in vacuum = " + str(Wavelength) + " nm" + "\n" + "Wavelength of x-rays in si3n4 = " + str(Wavelength1) + " nm" + "\n" + "Phase shift = " + str(phaseDifference) + " rad" +  "\n" + "Elapsed time = " + str(end-start) + " s"
label2 = "Number of slits = " + str(NumberOfSlits) + "\n" + "Slit length = " + str(SlitLength) + " nm" + "\n" + "Source points per slit = " + str(SourcesPerSlit) + "\n" + "Grating pitch = " + str(SlitLength*2) + " nm" + "\n" + "Aperture diameter = " + str(ApertureLength) + " nm" + "\n" + "Length of second grating = " + str(GratingLength) + " nm" + "\n" + "Grating thickness = " + str(Thickness) + " nm" + "\n" + "Number of observation points in front of grating 2 = " + str(NumObservationPoints2) + "\n" + "Distance between gratings = " + str(D) + " nm" + "\n" + "Index of refraction of x-rays in si3n4 = " + str(N1) + "\n" + "Attenuation coefficient = " + str(Mu) + "\n" + "Wavelength of x-rays in vacuum = " + str(Wavelength) + " nm" + "\n" + "Wavelength of x-rays in si3n4 = " + str(Wavelength1) + " nm" + "\n" + "Elapsed time = " + str(end-start) + " s"

# Graph 1: Intensity Distribution of X-rays on the back side of the First Grating
fig = plt.figure()
plt.plot(Y, U1, 'ko', markersize=N//5, label=label1)
plt.xlabel('y-Position on Aperture (nm)', fontsize=1.5*N)
plt.ylabel('Initial Amplitude', fontsize=1.5*N)
plt.ylim(0,2)
plt.title('Amplitude of X-rays on the back side of the First Grating', fontsize=2*N)
plt.legend(loc="upper left", fontsize=1.5*N)
fig.set_size_inches(2*N,1*N)
plt.xticks(fontsize=1.5*N)
plt.yticks(fontsize=1.5*N)
plt.gca().xaxis.get_offset_text().set_fontsize(1.5*N)
plt.gca().yaxis.get_offset_text().set_fontsize(1.5*N)
plt.savefig('xrayFirstGrating.png')
plt.close(fig)

# Graph 2: Intensity Distribution of X-rays on the front side of the Second Grating
fig = plt.figure()
plt.plot(O2, I2, 'black', markersize=1, label=label2)
plt.xlabel('y-Position on Second Grating (nm)', fontsize=1.5*N)
plt.ylabel('Intensity (lux)', fontsize=1.5*N)
plt.title('Intensity Distribution of X-rays on the front side of the Second Grating', fontsize=2*N)
plt.legend(loc="upper left", fontsize=1.5*N)
fig.set_size_inches(2*N,1*N)
plt.xticks(fontsize=1.5*N)
plt.yticks(fontsize=1.5*N)
plt.gca().xaxis.get_offset_text().set_fontsize(1.5*N)
plt.gca().yaxis.get_offset_text().set_fontsize(1.5*N)
plt.savefig('xraySecondGrating.png')
plt.close(fig)