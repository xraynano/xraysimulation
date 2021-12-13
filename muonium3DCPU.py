from numpy import sqrt, exp, pi
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit, prange
import numpy as np

start = timer()

# --Customizable Parameters--
NumberOfSlitsX = 3 # Number of slit columns on first grating (Default 100,000, but too slow to run).
NumberOfSlitsY = 15 # Number of slit rows on first grating (Default 100,000, but too slow to run).
SlitLengthX = 500 # Slit length in x-dimension (nm) (Default 500 nm).
SlitLengthY = 50 # Slit length in y-dimension (nm) (Default 50 nm).
SourcesPerSlitX = 250 # Number of source point columns in each slit (Default 500).
SourcesPerSlitY = 25 # Number of source point rows in each slit (Default 50).
NumObservationPointsX = 500 # Number of observation points in second grating (Default 10,000,000 but too slow to run).
NumObservationPointsY = 500 # Number of observation points in second grating (Default 10,000,000 but too slow to run).
GratingLengthX = 0.75e6 # Length of grating in x-dimension (nm) (Default 1e7 nm).
GratingLengthY = 0.75e6 # Length of grating in y-dimension (nm) (Default 1e7 nm).
D = 5e7 # Distance between gratings (nm)
InitialAmplitude = 1 # Initial amplitude of the incoming plane wave
InitialPhase = 1 # Initial phase of the incoming plane wave
Wavelength = 0.56 # Wavelength of muonium (nm)

# --Results from Parameters--
ApertureLengthX = NumberOfSlitsX * SlitLengthX + (NumberOfSlitsX - 1)*50 # Aperture length in x-dimension (nm).
ApertureLengthY = (2*NumberOfSlitsY - 1) * SlitLengthY # Aperture length in y-dimension (nm).
SourceSpacingX = SlitLengthX / SourcesPerSlitX # Spacing between source points in x-dimension (nm).
SourceSpacingY = SlitLengthY / SourcesPerSlitY # Spacing between source points in y-dimension (nm).
NumSpacingX = GratingLengthX / NumObservationPointsX # Spacing between observation points in x-dimension in second grating (nm).
NumSpacingY = GratingLengthY / NumObservationPointsY # Spacing between observation points in y-dimension in second grating (nm).
K = 2 * pi / Wavelength # Wavenumber
# Y is the list of y-positions of the source points on the first grating.
Y = np.array([(-ApertureLengthX/2 + i*(SlitLengthX + 50) + SourceSpacingX/2 + j*SourceSpacingX, -ApertureLengthY/2 + k*2*SlitLengthY + SourceSpacingY/2 + l*SourceSpacingY) for k in range(NumberOfSlitsY) for l in range(SourcesPerSlitY) for i in range(NumberOfSlitsX) for j in range(SourcesPerSlitX)])
# O is the list of y-positions of the observation points on the second grating.
O = np.array([(-GratingLengthX/2 + NumSpacingX/2 + i*NumSpacingX, -GratingLengthY/2 + NumSpacingY/2 + j*NumSpacingY) for j in range(NumObservationPointsY) for i in range(NumObservationPointsX)])
I2 = np.zeros(NumObservationPointsX * NumObservationPointsY) # Initialize list that containts the intensity distribution of grating 2

# --Functions--
@jit(nopython=True, parallel=True, fastmath=True)
def calcs(y, o, I2, k, numberOfSlitsX, numberOfSlitsY, slitLengthX, slitLengthY, sourcesPerSlitX, sourcesPerSlitY, sourceSpacingX, sourceSpacingY, numObservationPointsX, numObservationPointsY, osLengthX, osLengthY, d, initialAmplitude, initialPhase):
    '''Function returns the intensity distribution of the observation points on the second grating.'''
    for i in prange(numObservationPointsX * numObservationPointsY):
        amplitudeSum = 0 # Sum of intensities of each spherical source point on first grating incident to an observation point on second grating.
        for j in prange(numberOfSlitsX * sourcesPerSlitX * numberOfSlitsY * sourcesPerSlitY):
            r = sqrt(d**2 + (o[i][0] - y[j][0])**2 + (o[i][1] - y[j][1])**2) # Distance from source point on first grating to observation point on second grating.
            phase = exp(1j * k * r) * initialPhase # Complex phase of spherical source point at a certain  observation point.
            U = initialAmplitude * phase / r # Complex amplitude of spherical source point at a certain observation point.
            amplitudeSum = amplitudeSum + U * sourceSpacingX * sourceSpacingY
        I2[i] = amplitudeSum.real ** 2 + amplitudeSum.imag ** 2  # Intensity of observation point on second grating
    return I2
I2 = calcs(Y, O, I2, K, NumberOfSlitsX, NumberOfSlitsY, SlitLengthX, SlitLengthY, SourcesPerSlitX, SourcesPerSlitY, SourceSpacingX, SourceSpacingY, NumObservationPointsX, NumObservationPointsY, GratingLengthX, GratingLengthY, D, InitialAmplitude, InitialPhase)

end = timer()
print("Time for program to run =", end - start, "seconds.")

# --Plots--
N = 50
label1 = "Number of slit columns on grating 1 = " + str(NumberOfSlitsX) + "\n" + "Number of slit rows on grating 1 = " + str(NumberOfSlitsY) + "\n" + "Slit length x-dim = " + str(SlitLengthX) + " nm" + "\n" + "Slit length y-dim = " + str(SlitLengthY) + " nm" + "\n" + "Sources per slit x-dim = " + str(SourcesPerSlitX) + "\n" + "Sources per slit y-dim = " + str(SourcesPerSlitY) + "\n" + "Aperture length x-dim = " + str(ApertureLengthX) + " nm" + "\n" + "Aperture length y-dim = " + str(ApertureLengthY) + " nm" + "\n" + "Wavelength = " + str(Wavelength) + "\n" + "Elapsed time = " + str(end-start) + " s"             
label2 = "Number of slit columns on grating 1 = " + str(NumberOfSlitsX) + "\n" + "Number of slit rows on grating 1 = " + str(NumberOfSlitsY) + "\n" + "Slit length x-dim = " + str(SlitLengthX) + " nm" + "\n" + "Slit length y-dim = " + str(SlitLengthY) + " nm" + "\n" + "Sources per slit x-dim = " + str(SourcesPerSlitX) + "\n" + "Sources per slit y-dim = " + str(SourcesPerSlitY) + "\n" + "Number of observation points x-dim = " + str(NumObservationPointsX) + "\n" + "Number of observation points y-dim = " + str(NumObservationPointsY) + "\n" + "Grating length x-dim = " + str(GratingLengthX) + " nm" + "\n" + "Grating length y-dim = " + str(GratingLengthY) + " nm" + "\n" + "Distance between gratings = " + str(D) + "\n" + "Wavelength = " + str(Wavelength) + "\n" + "Elapsed time = " + str(end-start) + " s"             
o1 = np.array([i[0] for i in O])
o2 = np.array([i[1] for i in O])
O1 = np.reshape(o1, (NumObservationPointsX, NumObservationPointsY))
O2 = np.reshape(o2, (NumObservationPointsX, NumObservationPointsY))
i2 = np.reshape(I2, (NumObservationPointsX, NumObservationPointsY))

# Graph 1: Point Sources Positions in Slits of the First Grating For 3D Muonium Simulation
fig = plt.figure()
plt.rcParams['agg.path.chunksize'] = 1000000
y1 = np.array([i[0] for i in Y])
y2 = np.array([i[1] for i in Y])
A = max(ApertureLengthX, ApertureLengthY)
plt.plot(y1, y2, 'bo', markersize=1, label=label1)
plt.legend(loc="upper left", fontsize=1.5*N)
plt.xlabel('X-Position on Aperture (nm)', fontsize=1.5*N)
plt.ylabel('Y-Position on Aperture (nm)', fontsize=1.5*N)
plt.title('Point Sources Positions in Slits of the First Grating \n with Columnar Structure for 3D Muonium Simulation', fontsize=2*N)
plt.xlim(-A/2, A/2)
plt.ylim(-A/2, A/2)
fig.set_size_inches(N,N)
plt.xticks(fontsize=1.5*N)
plt.yticks(fontsize=1.5*N)
plt.gca().xaxis.get_offset_text().set_fontsize(1.5*N)
plt.gca().yaxis.get_offset_text().set_fontsize(1.5*N)
plt.savefig('muonium3DFirstGrating.png')
plt.close(fig)

# Graph 2: Intensity Distribution on Second Grating for 3D Munoium Simulation
fig = plt.figure()
plt.rcParams['agg.path.chunksize'] = 1000000
ax = fig.add_subplot(projection='3d')
plt.plot(o1, o2, I2, 'bo', markersize=1, label=label2)
plt.legend(loc="upper left", fontsize=1.5*N)
plt.title('Intensity Distribution on Second Grating \n with Columnar Structure for 3D Munoium Simulation', fontsize=2*N)
fig.set_size_inches(N,N)
ax.tick_params(labelsize=1.5*N)
ax.set_xlabel('X-Position on \n Second Grating (nm)', fontsize=1.5*N, labelpad=2*N)
ax.set_ylabel('Y-Position on \n Second Grating (nm)', fontsize=1.5*N, labelpad=2*N)
ax.set_zlabel('Intensity', fontsize=1.5*N, labelpad=2*N)
plt.gca().xaxis.get_offset_text().set_fontsize(1.5*N)
plt.gca().yaxis.get_offset_text().set_fontsize(1.5*N)
plt.gca().zaxis.get_offset_text().set_fontsize(1.5*N)
plt.savefig('muonium3DSecondGrating.png')
plt.close(fig)

# Graph 3: Intensity Colormap of Second Grating for 3D Munoium Simulation
fig = plt.figure()
plt.rcParams['agg.path.chunksize'] = 1000000
plt.imshow(i2, cmap='viridis', extent=[-GratingLengthX/2, GratingLengthX/2, -GratingLengthY/2, GratingLengthY/2])
cbar = plt.colorbar()
plt.xlabel('X-Position on Second Grating (nm)', fontsize=1.5*N)
plt.ylabel('Y-Position on Second Grating (nm)', fontsize=1.5*N)
plt.title('Intensity Colormap of Second Grating with \n Columnar Structure for 3D Munoium Simulation', fontsize=2*N)
fig.set_size_inches(N,N)
plt.xticks(fontsize=1.5*N)
plt.yticks(fontsize=1.5*N)
plt.gca().xaxis.get_offset_text().set_fontsize(1.5*N)
plt.gca().yaxis.get_offset_text().set_fontsize(1.5*N)
cbar.ax.tick_params(labelsize=1.5*N)
cbar.ax.yaxis.get_offset_text().set_fontsize(1.5*N)
plt.savefig('muonium3DSecondGratingColormap.png')
plt.close(fig)



