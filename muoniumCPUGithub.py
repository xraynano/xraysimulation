from numpy import sqrt, exp, pi, absolute, complex128
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from numba import jit, prange
import numpy as np

start = timer()

# --Customizable Parameters--
NumberOfSlits = 5 # Number of slits on first grating (Default 100,000, but too slow to run).
NumberOfSlits2 = 10 # Number of slits on second grating (Default 100,000, but too slow to run).
SlitLength = 50 # Length of the slit (nm) (Default 50 nm).
SourcesPerSlit = 50 # Sources per slit (Default 50).
NumObservationPoints2 = 1000 # Number of observation points in second grating (Default 10,000,000 but too slow to run).
NumObservationPoints3 = 10000 # Number of observation points in third grating (Default 10,000,000 but too slow to run).
GratingLength2 = 1e7 # Length of second grating (nm) (Default 1e7 nm).
GratingLength3 = 1e7 # Length of third grating (nm) (Default 1e7 nm).
D = 5e7 # Distance between gratings (nm).
InitialAmplitude = 1 # Initial amplitude of the incoming plane wave (Default 1).
InitialPhase = 1 # Initial phase of the incoming plane wave (Default 1).
Wavelength = 0.56 # Wavelength (nm) (Default 0.56 nm).

# --Results from Parameters--
ApertureLength = (2*NumberOfSlits - 1) * SlitLength # Aperture length of first grating (nm).
ApertureLength2 = (2*NumberOfSlits2 - 1) * SlitLength # Aperture length of second grating (nm).
SourceSpacing = SlitLength / SourcesPerSlit # Spacing between source points (nm).
NumSpacing2 = GratingLength2 / NumObservationPoints2 # Spacing between observation points in second grating (nm).
NumSpacing3 = GratingLength3 / NumObservationPoints3 # Spacing between observation points in third grating (nm).
K = 2 * pi / Wavelength # Wavenumber.
# Y, Y2 are the list of y-positions of the source points in the first and second gratings respectively.
Y = np.array([-ApertureLength/2 + i*2*SlitLength + SourceSpacing/2 + j*SourceSpacing for i in range(NumberOfSlits) for j in range(SourcesPerSlit)])
Y2 = np.array([-ApertureLength2/2 + i*2*SlitLength + SourceSpacing/2 + j*SourceSpacing for i in range(NumberOfSlits2) for j in range(SourcesPerSlit)])
# O2, O3 are the list of y-positions of the observation points in sceond and third gratings respectively.
O2 = np.array([-GratingLength2/2 + NumSpacing2/2 + j*NumSpacing2 for j in range(NumObservationPoints2)])
O3 = np.array([-GratingLength3/2 + NumSpacing3/2 + j*NumSpacing3 for j in range(NumObservationPoints3)])
I2 = np.zeros(NumObservationPoints2) # Initialize list that will contain the intensity distribution of the second grating.
I3 = np.zeros(NumObservationPoints3) # Initialize list that will contain the intensity distribution of the third grating.
G2P = np.zeros(NumberOfSlits2 * SourcesPerSlit, dtype=complex128) # Phases of the source points on the second grating.
G2U = np.zeros(NumberOfSlits2 * SourcesPerSlit, dtype=complex128) # Amplitudes of the source points on the second grating.

# --Functions---------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True, fastmath=True)
def calcs(y, o2, I2, d, k, numberOfSlits, sourceSpacing, sourcesPerSlit, numObservationPoints2, initialAmplitude, initialPhase):
    '''Function returns the intensity distribution of the observation points on the second grating.'''
    for i in prange(numObservationPoints2):
        amplitudeSum = 0 # Sum of intensities of each spherical source point on first grating incident to an observation point on second grating.
        for j in prange(numberOfSlits * sourcesPerSlit):
            r = sqrt(d**2 + (o2[i] - y[j])**2) # Distance from source point on first grating to observation point on second grating.
            phase = exp(1j * k * r) * initialPhase # Complex phase of spherical source point at a certain  observation point.
            U = initialAmplitude * phase / r # Complex amplitude of spherical source point at a certain observation point.
            amplitudeSum = amplitudeSum + U * sourceSpacing
        I2[i] = amplitudeSum.real**2 + amplitudeSum.imag**2 # Intensity of observation point on second grating
    return I2
I2 = calcs(Y, O2, I2, D, K, NumberOfSlits, SourceSpacing, SourcesPerSlit, NumObservationPoints2, InitialAmplitude, InitialPhase)

@jit(nopython=True, parallel=True, fastmath=True)
def calcs2(y, y2, g2U, g2P, d, k, numberOfSlits, numberOfSlits2, sourceSpacing, sourcesPerSlit, initialAmplitude, initialPhase): 
    '''Function returns the amplitudes and phases of the source points on the second grating.'''
    for i in prange(numberOfSlits2 * sourcesPerSlit):
        amplitudeSum = 0 # Sum of intensities of each source point on the first grating incident to a source point on the second grating.
        phaseSum = 1 # Sum of phases of each spherical source point on the first grating incident to a source point on the second grating.
        for j in prange(numberOfSlits * sourcesPerSlit):
            r = sqrt(d**2 + (y2[i] - y[j])**2) # Distance from source point on first grating to a certain source point on the second grating.
            phase = exp(1j * k * r) * initialPhase # Complex phase of spherical source point at a certain  observation point.
            U = initialAmplitude * phase / r # Complex amplitude of spherical source point at a certain observation point.
            amplitudeSum = amplitudeSum + U * sourceSpacing
            phaseSum = phaseSum * phase
        g2P[i] = phaseSum # Phase of a source point on second grating.
        g2U[i] = absolute(amplitudeSum) # Amplitude of a source point on second grating.
    return g2U,g2P
G2U,G2P = calcs2(Y, Y2, G2U, G2P, D, K, NumberOfSlits, NumberOfSlits2, SourceSpacing, SourcesPerSlit, InitialAmplitude, InitialPhase)

@jit(nopython=True, parallel=True, fastmath=True)
def calcs3(y2, o3, g2U, g2P, I3, d, k, numberOfSlits2, sourceSpacing, sourcesPerSlit, numObservationPoints3):
    '''Function returns the intensity distribution of the observation points on the third grating.'''
    for i in prange(numObservationPoints3):
        amplitudeSum = 0 # Sum of intensities of each source point on the second grating incident to an observation point on the third grating.
        for j in prange(numberOfSlits2 * sourcesPerSlit):
            r = sqrt(d**2 + (o3[i] - y2[j])**2) # Distance from source point on second grating to a certain observation point on the third grating.
            phase = exp(1j * k * r) * g2P[j] # Complex phase of a source point on the second grating at a certain observation point on the third grating.
            U = g2U[j] * phase / r # Complex amplitude of a source point on second grating at a certain observation point on third grating.
            amplitudeSum = amplitudeSum + U * sourceSpacing
        I3[i] = amplitudeSum.real**2 + amplitudeSum.imag**2 # Intensity of an observation point on third grating.
    return I3
I3 = calcs3(Y2, O3, G2U, G2P, I3, D, K, NumberOfSlits2, SourceSpacing, SourcesPerSlit, NumObservationPoints3)

end = timer()
print("Time for program to run =", end - start, "seconds.")

# --Plots---------------------------------------------------------------------------------------------------------------
N = 20
label2 = "Number of slits on Grating 1 = " + str(NumberOfSlits) + "\n" + "Slit length = " + str(SlitLength) + " nm" + "\n" + "Grating pitch = " + str(2*SlitLength) + " nm" + "\n" + "Source points per slit = " + str(SourcesPerSlit) + "\n" + "Distance between slit centers = " + str(SlitLength*2) + " nm" + "\n" + "Aperture diameter = " + str(ApertureLength) + " nm" + "\n" + "Length of second grating = " + str(GratingLength2) + " nm" + "\n" + "Number of observation points in grating 2 = " + str(NumObservationPoints2) + "\n" + "Wavelength = " + str(Wavelength) + " nm" + "\n" + "Distance between gratings = " + str(D) + " nm" + "\n" + "Elapsed time = " + str(end-start) + " s"
label3 = "Number of slits on Grating 2 = " + str(NumberOfSlits2) + "\n" + "Slit length = " + str(SlitLength) + " nm" + "\n" + "Grating pitch = " + str(2*SlitLength) + " nm" + "\n" + "Source points per slit = " + str(SourcesPerSlit) + "\n" + "Distance between slit centers = " + str(SlitLength*2) + " nm" + "\n" + "Aperture diameter = " + str(ApertureLength2) + " nm" + "\n" + "Length of third grating = " + str(GratingLength3) + " nm" + "\n" + "Number of observation points in grating 3 = " + str(NumObservationPoints3) + "\n" + "Wavelength = " + str(Wavelength) + " nm" + "\n" + "Distance between gratings = " + str(D) + " nm" + "\n" + "Elapsed time = " + str(end-start) + " s"

# Graph 1: Intensity Distribution on Second Grating for Muonium Simulation
fig = plt.figure()
plt.rcParams['agg.path.chunksize'] = 1000000
plt.plot(O2, I2, 'k', markersize=1, label=label2)
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
plt.plot(O3, I3, 'k', markersize=1, label=label3)
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
plt.plot(O3, I3, 'k', markersize=1, label=label3)
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

# # Graph 4: Theoretical Intensity Distribution on Second Grating (Equation can be found in Fall 2021 x-ray final report)
# p = 2 * SlitLength # Grating pitch (nm)
# NumObservationPoints2_theoretical = 2000000
# sourceSpacing = GratingLength2/NumObservationPoints2_theoretical
# y = [-GratingLength2/2 + sourceSpacing/2 + i*sourceSpacing for i in range(0, NumObservationPoints2_theoretical)]
# I1 = [(np.sin(np.pi * SlitLength * i/np.sqrt(i**2 + D**2) / Wavelength) / (np.pi * SlitLength * i/np.sqrt(i**2 + D**2) / Wavelength))**2 * (np.sin(NumberOfSlits * np.pi * p * i/np.sqrt(i**2 + D**2) / Wavelength) / np.sin(np.pi * p * i/np.sqrt(i**2 + D**2) / Wavelength))**2 for i in y]
# fig = plt.figure()
# plt.plot(y, I1, 'r', markersize=1)
# plt.xlabel('y-Position on Second Grating (nm)', fontsize=1.5*N)
# plt.ylabel('Intensity (lux)', fontsize=1.5*N)
# plt.title('Theoretical Intensity Distribution on Second Grating', fontsize=2*N)
# fig.set_size_inches(2*N,1*N)
# plt.xticks(fontsize=1.5*N)
# plt.yticks(fontsize=1.5*N)
# plt.gca().xaxis.get_offset_text().set_fontsize(1.5*N)
# plt.gca().yaxis.get_offset_text().set_fontsize(1.5*N)
# plt.tick_params(axis='both', length=N, width=N//10)
# plt.savefig('muoniumSecondGratingTheoretical.png')
# plt.close(fig)





