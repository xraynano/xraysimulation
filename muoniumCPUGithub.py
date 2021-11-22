from numpy import sqrt, exp, pi, absolute, complex128
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from numba import jit, prange
import numpy as np
# import pandas as pd

start = timer()

NumberOfSlits = 100000
NumberOfSlits2 = 10000
SlitLength = 50
SourcesPerSlit = 50
ApertureLength = (2*NumberOfSlits - 1) * SlitLength
ApertureLength2 = (2*NumberOfSlits2 - 1) * SlitLength
DS = SlitLength / SourcesPerSlit
NumObservationPoints = 2000000
GratingLength = 1e7
NumSpacing = GratingLength / NumObservationPoints
d = 5e7
# U0 = 1
InitialAmplitude = 1
InitialPhase = 1
Wavelength = 0.56
K = 2 * pi / Wavelength
y = np.array([-ApertureLength/2 + i*2*SlitLength + DS/2 + j*DS for i in range(NumberOfSlits) for j in range(SourcesPerSlit)])
y2 = np.array([-ApertureLength2/2 + i*2*SlitLength + DS/2 + j*DS for i in range(NumberOfSlits2) for j in range(SourcesPerSlit)])
o = np.array([-GratingLength/2 + NumSpacing/2 + j*NumSpacing for j in range(NumObservationPoints)])
I2temp = np.zeros(NumObservationPoints)
I3temp = np.zeros(NumObservationPoints)
G2P = np.zeros(NumberOfSlits2 * SourcesPerSlit, dtype=complex128)
G2U = np.zeros(NumberOfSlits2 * SourcesPerSlit, dtype=complex128)

@jit(nopython=True, parallel=True, fastmath=True)
def calcs(y, o, I2, D, k, numberOfSlits, dS, sourcesPerSlit, numObservationPoints, initialAmplitude, initialPhase):
    for i in prange(numObservationPoints):
        amplitudeSum = 0
        for j in prange(numberOfSlits * sourcesPerSlit):
            r = sqrt(D**2 + (o[i] - y[j])**2)
            phase = exp(1j * k * r) * initialPhase
            U = initialAmplitude * phase / r
            amplitudeSum = amplitudeSum + U * dS
        I2[i] = amplitudeSum.real**2 + amplitudeSum.imag**2
    return I2

I2 = calcs(y, o, I2temp, d, K, NumberOfSlits, DS, SourcesPerSlit, NumObservationPoints, InitialAmplitude, InitialPhase)

# @jit(nopython=True, parallel=True, fastmath=True)
# def calcs2(y, y2, g2U, g2P, D, k, numberOfSlits, numberOfSlits2, dS, sourcesPerSlit, initialAmplitude, initialPhase): 
#     for i in prange(numberOfSlits2 * sourcesPerSlit):
#         amplitudeSum = 0
#         phaseSum = 1
#         for j in prange(numberOfSlits * sourcesPerSlit):
#             r = sqrt(D**2 + (y2[i] - y[j])**2)
#             phase = exp(1j * k * r) * initialPhase
#             U = initialAmplitude * phase / r
#             amplitudeSum = amplitudeSum + U * dS
#             phaseSum = phaseSum * phase
#         g2P[i] = phaseSum
#         g2U[i] = absolute(amplitudeSum)
        
#     return g2U,g2P

# G2U,G2P = calcs2(y, y2, G2U, G2P, d, K, NumberOfSlits, NumberOfSlits2, DS, SourcesPerSlit, InitialAmplitude, InitialPhase)

# @jit(nopython=True, parallel=True, fastmath=True)
# def calcs3(y2, o, g2U, g2P, I3, D, k, numberOfSlits2, dS, sourcesPerSlit, numObservationPoints):
#     for i in prange(numObservationPoints):
#         amplitudeSum = 0
#         for j in prange(numberOfSlits2 * sourcesPerSlit):
#             r = sqrt(D**2 + (o[i] - y2[j])**2)
#             phase = exp(1j * k * r) * g2P[j]
#             U = g2U[j] * phase / r
#             amplitudeSum = amplitudeSum + U * dS
#         I3[i] = amplitudeSum.real**2 + amplitudeSum.imag**2
#     return I3

# I3 = calcs3(y2, o, G2U, G2P, I3temp, d, K, NumberOfSlits2, DS, SourcesPerSlit, NumObservationPoints)

# df = pd.DataFrame(data={"col1": o, "col2": I2, "col3": I3})
# df.to_csv("intensity.csv", sep=',',index=False)

end = timer()
print("Time for program to run =", end - start, "seconds.")

N = 20
label2 = "Number of slits on Grating 1 = " + str(NumberOfSlits) + "\n" + "Slit length = " + str(SlitLength) + " nm" + "\n" + "Grating pitch = " + str(2*SlitLength) + " nm" + "\n" + "Source points per slit = " + str(SourcesPerSlit) + "\n" + "Distance between slit centers = " + str(SlitLength*2) + " nm" + "\n" + "Aperture diameter = " + str(ApertureLength) + " nm" + "\n" + "Grating length = " + str(GratingLength) + " nm" + "\n" + "Number of observation points = " + str(NumObservationPoints) + "\n" + "Wavelength = " + str(Wavelength) + " nm" + "\n" + "Distance between gratings = " + str(d) + " nm" + "\n" + "Elapsed time = " + str(end-start) + " s"
label3 = "Number of slits on Grating 2 = " + str(NumberOfSlits2) + "\n" + "Slit length = " + str(SlitLength) + " nm" + "\n" + "Grating pitch = " + str(2*SlitLength) + " nm" + "\n" + "Source points per slit = " + str(SourcesPerSlit) + "\n" + "Distance between slit centers = " + str(SlitLength*2) + " nm" + "\n" + "Aperture diameter = " + str(ApertureLength2) + " nm" + "\n" + "Grating length = " + str(GratingLength) + " nm" + "\n" + "Number of observation points = " + str(NumObservationPoints) + "\n" + "Wavelength = " + str(Wavelength) + " nm" + "\n" + "Distance between gratings = " + str(d) + " nm" + "\n" + "Elapsed time = " + str(end-start) + " s"

fig = plt.figure()
plt.rcParams['agg.path.chunksize'] = 1000000
plt.plot(o, I2, 'k', markersize=1, label=label2)
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

# fig = plt.figure()
# plt.rcParams['agg.path.chunksize'] = 1000000
# plt.plot(o, I3, 'k', markersize=1, label=label3)
# plt.xlabel('y-Position on Third Grating (nm)', fontsize=1.5*N)
# plt.ylabel('Intensity (lux)', fontsize=1.5*N)
# plt.title('Intensity Distribution on Third Grating for Muonium Simulation', fontsize=2*N)
# plt.legend(loc="upper left", fontsize=1.5*N)
# fig.set_size_inches(2*N,1*N)
# plt.xticks(fontsize=1.5*N)
# plt.yticks(fontsize=1.5*N)
# plt.gca().xaxis.get_offset_text().set_fontsize(1.5*N)
# plt.gca().yaxis.get_offset_text().set_fontsize(1.5*N)
# plt.tick_params(axis='both', length=N, width=N//10)
# plt.savefig('muoniumThirdGrating.png')
# plt.close(fig)

# fig = plt.figure()
# plt.rcParams['agg.path.chunksize'] = 1000000
# plt.plot(o, I3, 'k', markersize=1, label=label3)
# plt.xlim(-10*SlitLength,10*SlitLength)
# plt.xlabel('y-Position on Third Grating (nm)', fontsize=1.5*N)
# plt.ylabel('Intensity (lux)', fontsize=1.5*N)
# plt.title('Intensity Distribution on Third Grating for Muonium Simulation Zoomed in', fontsize=2*N)
# plt.legend(loc="upper left", fontsize=1.5*N)
# fig.set_size_inches(2*N,1*N)
# plt.xticks(fontsize=1.5*N)
# plt.yticks(fontsize=1.5*N)
# plt.gca().xaxis.get_offset_text().set_fontsize(1.5*N)
# plt.gca().yaxis.get_offset_text().set_fontsize(1.5*N)
# plt.tick_params(axis='both', length=N, width=N//10)
# plt.savefig('muoniumThirdGratingZoomedIn.png')
# plt.close(fig)



