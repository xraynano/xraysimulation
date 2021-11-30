from numpy import sqrt, exp, pi, absolute, complex128
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from numba import jit, prange
import numpy as np
import math

start = timer()

# --Parameters--
NumberOfSlits = 100 # Number of slits in grating 1
NumberOfSlits2 = 1 # Number of slits in grating 2
SlitLength = 50 # Length of the slit (nm)
SourcesPerSlit = 50 # Sources per slit
ApertureLength = (2*NumberOfSlits - 1) * SlitLength # Aperture length of grating 1 (nm)
ApertureLength2 = (2*NumberOfSlits2 - 1) * SlitLength # Aperture length of grating 2 (nm)
SourceSpacing = SlitLength / SourcesPerSlit # Spacing between source points (nm)
NumObservationPoints2 = 2000000 # Number of observation points in grating 2
NumObservationPoints3 = 1 # Number of observation points in grating 3
OSLength2 = 1e7 # Length of observation screen of grating 2 (nm) (Default is 1e7 which is the length of the grating)
OSLength3 = 1e4 # Length of observation screen of grating 3 (nm) (Default is 1e7 which is the length of the grating)
NumSpacing2 = OSLength2 / NumObservationPoints2 # Spacing between observation points in grating 2 (nm)
NumSpacing3 = OSLength3 / NumObservationPoints3 # Spacing between observation points in grating 3 (nm)
d = 5e7 # Distance between gratings (nm)
InitialAmplitude = 1 # Initial amplitude of the incoming plane wave
InitialPhase = 1 # Initial phase of the incoming plane wave
Wavelength = 0.56 # Wavelength (nm)
K = 2 * pi / Wavelength # Wavenumber
# y, y2 are the list of y-positions of the source points in grating 1 and grating 2 respectively
y = np.array([-ApertureLength/2 + i*2*SlitLength + SourceSpacing/2 + j*SourceSpacing for i in range(NumberOfSlits) for j in range(SourcesPerSlit)])
y2 = np.array([-ApertureLength2/2 + i*2*SlitLength + SourceSpacing/2 + j*SourceSpacing for i in range(NumberOfSlits2) for j in range(SourcesPerSlit)])
# o2, o3 are the list of y-positions of the observation points in grating 2 and grating 3 respectively
o2 = np.array([-OSLength2/2 + NumSpacing2/2 + j*NumSpacing2 for j in range(NumObservationPoints2)])
o3 = np.array([-OSLength3/2 + NumSpacing3/2 + j*NumSpacing3 for j in range(NumObservationPoints3)])
I2temp = np.zeros(NumObservationPoints2) # Initialize list that containts the intensity distribution of grating 2
I3temp = np.zeros(NumObservationPoints3) # Initialize list that containts the Intensity distribution of grating 3
G2P = np.zeros(NumberOfSlits2 * SourcesPerSlit, dtype=complex128) # Phases of the source points of grating 2
G2U = np.zeros(NumberOfSlits2 * SourcesPerSlit, dtype=complex128) # Phases of the source points of grating 3

@jit(nopython=True, parallel=True, fastmath=True)
def calcs(y, o2, I2, D, k, numberOfSlits, sourceSpacing, sourcesPerSlit, numObservationPoints2, initialAmplitude, initialPhase):
    '''Function returns the intensity distribution of the observation points on grating 2.'''
    for i in prange(numObservationPoints2):
        amplitudeSum = 0
        for j in prange(numberOfSlits * sourcesPerSlit):
            r = sqrt(D**2 + (o2[i] - y[j])**2)
            phase = exp(1j * k * r) * initialPhase
            U = initialAmplitude * phase / r
            amplitudeSum = amplitudeSum + U * sourceSpacing
        I2[i] = amplitudeSum.real**2 + amplitudeSum.imag**2
    return I2
I2 = calcs(y, o2, I2temp, d, K, NumberOfSlits, SourceSpacing, SourcesPerSlit, NumObservationPoints2, InitialAmplitude, InitialPhase)

@jit(nopython=True, parallel=True, fastmath=True)
def calcs2(y, y2, g2U, g2P, D, k, numberOfSlits, numberOfSlits2, sourceSpacing, sourcesPerSlit, initialAmplitude, initialPhase): 
    '''Function returns the amplitudes and phases of the source points of grating 2.'''
    for i in prange(numberOfSlits2 * sourcesPerSlit):
        amplitudeSum = 0
        phaseSum = 1
        for j in prange(numberOfSlits * sourcesPerSlit):
            r = sqrt(D**2 + (y2[i] - y[j])**2)
            phase = exp(1j * k * r) * initialPhase
            U = initialAmplitude * phase / r
            amplitudeSum = amplitudeSum + U * sourceSpacing
            phaseSum = phaseSum * phase
        g2P[i] = phaseSum
        g2U[i] = absolute(amplitudeSum)
    return g2U,g2P
G2U,G2P = calcs2(y, y2, G2U, G2P, d, K, NumberOfSlits, NumberOfSlits2, SourceSpacing, SourcesPerSlit, InitialAmplitude, InitialPhase)

@jit(nopython=True, parallel=True, fastmath=True)
def calcs3(y2, o3, g2U, g2P, I3, D, k, numberOfSlits2, sourceSpacing, sourcesPerSlit, numObservationPoints3):
    '''Function returns the intensity distribution of the source points from grating 3.'''
    for i in prange(numObservationPoints3):
        amplitudeSum = 0
        for j in prange(numberOfSlits2 * sourcesPerSlit):
            r = sqrt(D**2 + (o3[i] - y2[j])**2)
            phase = exp(1j * k * r) * g2P[j]
            U = g2U[j] * phase / r
            amplitudeSum = amplitudeSum + U * sourceSpacing
        I3[i] = amplitudeSum.real**2 + amplitudeSum.imag**2
    return I3
I3 = calcs3(y2, o3, G2U, G2P, I3temp, d, K, NumberOfSlits2, SourceSpacing, SourcesPerSlit, NumObservationPoints3)

end = timer()
print("Time for program to run =", end - start, "seconds.")

# --Plots--
N = 20
label2 = "Number of slits on Grating 1 = " + str(NumberOfSlits) + "\n" + "Slit length = " + str(SlitLength) + " nm" + "\n" + "Grating pitch = " + str(2*SlitLength) + " nm" + "\n" + "Source points per slit = " + str(SourcesPerSlit) + "\n" + "Distance between slit centers = " + str(SlitLength*2) + " nm" + "\n" + "Aperture diameter = " + str(ApertureLength) + " nm" + "\n" + "Observation screen length grating 2 = " + str(OSLength2) + " nm" + "\n" + "Number of observation points in grating 2 = " + str(NumObservationPoints2) + "\n" + "Wavelength = " + str(Wavelength) + " nm" + "\n" + "Distance between gratings = " + str(d) + " nm" + "\n" + "Elapsed time = " + str(end-start) + " s"
label3 = "Number of slits on Grating 2 = " + str(NumberOfSlits2) + "\n" + "Slit length = " + str(SlitLength) + " nm" + "\n" + "Grating pitch = " + str(2*SlitLength) + " nm" + "\n" + "Source points per slit = " + str(SourcesPerSlit) + "\n" + "Distance between slit centers = " + str(SlitLength*2) + " nm" + "\n" + "Aperture diameter = " + str(ApertureLength2) + " nm" + "\n" + "Observation screen length grating 3 = " + str(OSLength3) + " nm" + "\n" + "Number of observation points in grating 3 = " + str(NumObservationPoints3) + "\n" + "Wavelength = " + str(Wavelength) + " nm" + "\n" + "Distance between gratings = " + str(d) + " nm" + "\n" + "Elapsed time = " + str(end-start) + " s"

# Graph 1: Intensity Distribution on Second Grating for Muonium Simulation
fig = plt.figure()
plt.rcParams['agg.path.chunksize'] = 1000000
plt.plot(o2, I2, 'k', markersize=1, label=label2)
plt.xlabel('y-Position on Second Grating (nm)', fontsize=1.5*N)
plt.ylabel('Intensity (lux)', fontsize=1.5*N)
plt.title('Intensity Distribution on Second Grating for Muonium Simulation', fontsize=2*N)
plt.legend(loc="upper left", fontsize=1.5*N)
fig.set_size_inches(2*N,1*N)
plt.xticks(fontsize=1.5*N)
plt.yticks(fontsize=1.5*N)
plt.gca().xaxis.get_offset_text().set_fontsize(1.5*N)
plt.gca().yaxis.get_offset_text().set_fontsize(1.5*N)
plt.tick_params(axis='both', length=N, width=N//10)
plt.savefig('muoniumSecondGrating.png')
plt.close(fig)

# Graph 2: Intensity Distribution on Third Grating for Muonium Simulation
fig = plt.figure()
plt.rcParams['agg.path.chunksize'] = 1000000
plt.plot(o3, I3, 'k', markersize=1, label=label3)
plt.xlabel('y-Position on Third Grating (nm)', fontsize=1.5*N)
plt.ylabel('Intensity (lux)', fontsize=1.5*N)
plt.title('Intensity Distribution on Third Grating for Muonium Simulation', fontsize=2*N)
plt.legend(loc="upper left", fontsize=1.5*N)
fig.set_size_inches(2*N,1*N)
plt.xticks(fontsize=1.5*N)
plt.yticks(fontsize=1.5*N)
plt.gca().xaxis.get_offset_text().set_fontsize(1.5*N)
plt.gca().yaxis.get_offset_text().set_fontsize(1.5*N)
plt.tick_params(axis='both', length=N, width=N//10)
plt.savefig('muoniumThirdGrating.png')
plt.close(fig)

# Graph 3: Intensity Distribution on Third Grating for Muonium Simulation Zoomed in
fig = plt.figure()
plt.rcParams['agg.path.chunksize'] = 1000000
plt.plot(o3, I3, 'k', markersize=1, label=label3)
plt.xlim(-10*SlitLength,10*SlitLength)
plt.xlabel('y-Position on Third Grating (nm)', fontsize=1.5*N)
plt.ylabel('Intensity (lux)', fontsize=1.5*N)
plt.title('Intensity Distribution on Third Grating for Muonium Simulation Zoomed in', fontsize=2*N)
plt.legend(loc="upper left", fontsize=1.5*N)
fig.set_size_inches(2*N,1*N)
plt.xticks(fontsize=1.5*N)
plt.yticks(fontsize=1.5*N)
plt.gca().xaxis.get_offset_text().set_fontsize(1.5*N)
plt.gca().yaxis.get_offset_text().set_fontsize(1.5*N)
plt.tick_params(axis='both', length=N, width=N//10)
plt.savefig('muoniumThirdGratingZoomedIn.png')
plt.close(fig)

# Graph 4: Theoretical Intensity Distribution on Second Grating'
lambda0 = 0.56
# apertureLength = 1e3
# NumberOfSlits = 1
slitLength = 50
# d = (apertureLength - N*slitLength)/N + slitLength
d = 100 # Grating pitch (nm)
D = 5e7
numObs = 2000000
screenLength = 1e7
sourceSpacing = screenLength/numObs
y = [-screenLength/2 + sourceSpacing/2 + i*sourceSpacing for i in range(0,numObs)]
I1 = [(math.sin(math.pi * slitLength * i/math.sqrt(i**2 + D**2) / lambda0) / (math.pi * slitLength * i/math.sqrt(i**2 + D**2) / lambda0))**2 * (math.sin(NumberOfSlits * math.pi * d * i/math.sqrt(i**2 + D**2) / lambda0) / math.sin(math.pi * d * i/math.sqrt(i**2 + D**2) / lambda0))**2 for i in y]
fig = plt.figure()
plt.plot(y, I1, 'r', markersize=1)
plt.xlabel('y-Position on Second Grating (nm)', fontsize=1.5*N)
plt.ylabel('Intensity (lux)', fontsize=1.5*N)
plt.title('Theoretical Intensity Distribution on Second Grating', fontsize=2*N)
fig.set_size_inches(2*N,1*N)
plt.xticks(fontsize=1.5*N)
plt.yticks(fontsize=1.5*N)
plt.gca().xaxis.get_offset_text().set_fontsize(1.5*N)
plt.gca().yaxis.get_offset_text().set_fontsize(1.5*N)
plt.tick_params(axis='both', length=N, width=N//10)
plt.savefig('muoniumSecondGratingTheoretical.png')
plt.close(fig)





