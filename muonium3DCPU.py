from numpy import sqrt, exp, pi
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from numba import jit, prange
import numpy as np

start = timer()

# --Parameters--
NumberOfSlitsX = 2 # Number of slit columns of grating 1
NumberOfSlitsY = 10 # Number of slit rows of grating 1
SlitLengthX = 500 # Slit length in x-dimension
SlitLengthY = 50 # Slit length in y-dimension
SourcesPerSlitX = 500 # Number of source point columns in each slit
SourcesPerSlitY = 50 # Number of source point rows in each slit
ApertureLengthX = NumberOfSlitsX * SlitLengthX + (NumberOfSlitsX - 1)*50 # Aperture length in x-dimension
ApertureLengthY = (2*NumberOfSlitsY - 1) * SlitLengthY # Aperture length in y-dimension
SourceSpacingX = SlitLengthX / SourcesPerSlitX # Spacing between source points in x-dimension
SourceSpacingY = SlitLengthY / SourcesPerSlitY # Spacing between source points in y-dimension
NumObservationPointsX = 500 # Number of observation points in x-dimension
NumObservationPointsY = 500 # Number of observation points in y-dimension
OSLengthX = 1e7 # Length of observation screen in x-dimension (nm) (1e7 is the default which is the length of the grating in the x-dimension)
OSLengthY = 1e7 # Length of observation screen in y-dimension (nm) (1e7 is the default which is the length of the grating in the y-dimension)
NumSpacingX = OSLengthX / NumObservationPointsX # Spacing between observation points in x-dimension in grating 2 (nm)
NumSpacingY = OSLengthY / NumObservationPointsY # Spacing between observation points in y-dimension in grating 2 (nm)
D = 5e7 # Distance between gratings (nm)
InitialAmplitude = 1 # Initial amplitude of the incoming plane wave
InitialPhase = 1 # Initial phase of the incoming plane wave
Wavelength = 0.56 # Wavelength of muonium (nm)
K = 2 * pi / Wavelength # Wavenumber
# y is the list of y-positions of the source points in grating 1
y = np.array([(-ApertureLengthX/2 + i*(SlitLengthX + 50) + SourceSpacingX/2 + j*SourceSpacingX, -ApertureLengthY/2 + k*2*SlitLengthY + SourceSpacingY/2 + l*SourceSpacingY) for k in range(NumberOfSlitsY) for l in range(SourcesPerSlitY) for i in range(NumberOfSlitsX) for j in range(SourcesPerSlitX)])
# o is the list of y-positions of the observation points in grating 2
o = np.array([(-OSLengthX/2 + NumSpacingX/2 + i*NumSpacingX, -OSLengthY/2 + NumSpacingY/2 + j*NumSpacingY) for j in range(NumObservationPointsY) for i in range(NumObservationPointsX)])
i2 = np.zeros(NumObservationPointsX * NumObservationPointsY) # Initialize list that containts the intensity distribution of grating 2

# --Functions--
@jit(nopython=True, parallel=True, fastmath=True)
def calcs(y, o, I2, k, numberOfSlitsX, numberOfSlitsY, slitLengthX, slitLengthY, sourcesPerSlitX, sourcesPerSlitY, sourceSpacingX, sourceSpacingY, numObservationPointsX, numObservationPointsY, osLengthX, osLengthY, d2, initialAmplitude, initialPhase):
    '''Function returns the intensity distribution of the observation points on grating 2.'''
    for i in prange(numObservationPointsX * numObservationPointsY):
        amplitudeSum = 0 # Sum of intensities of each spherical source point on first grating incident to an observation point on second grating.
        for j in prange(numberOfSlitsX * sourcesPerSlitX * numberOfSlitsY * sourcesPerSlitY):
            r = sqrt((d2)**2 + (o[i][0] - y[j][0])**2 + (o[i][1] - y[j][1])**2) # Distance from source point on first grating to observation point on second grating.
            phase = exp(1j * k * r) * initialPhase # Complex phase of spherical source point at a certain  observation point.
            U = initialAmplitude * phase / r # Complex amplitude of spherical source point at a certain observation point.
            amplitudeSum = amplitudeSum + U * sourceSpacingX * sourceSpacingY
        I2[i] = amplitudeSum.real ** 2 + amplitudeSum.imag ** 2  # Intensity of observation point on second grating
    return I2
I2 = calcs(y, o, i2, K, NumberOfSlitsX, NumberOfSlitsY, SlitLengthX, SlitLengthY, SourcesPerSlitX, SourcesPerSlitY, SourceSpacingX, SourceSpacingY, NumObservationPointsX, NumObservationPointsY, OSLengthX, OSLengthY, D, InitialAmplitude, InitialPhase)

end = timer()
print("Time for program to run =", end - start, "seconds.")

# --Plots--
N = 20
label1 = "Number of slit columns on grating 1 = " + str(NumberOfSlitsX) + "\n" + "Number of slit rows on grating 1 = " + str(NumberOfSlitsY) + "\n" + "Slit length x-dim = " + str(SlitLengthX) + " nm" + "\n" + "Slit length y-dim = " + str(SlitLengthY) + " nm" + "\n" + "Sources per slit x-dim = " + str(SourcesPerSlitX) + "\n" + "Sources per slit y-dim = " + str(SourcesPerSlitY) + "\n" + "Aperture length x-dim = " + str(ApertureLengthX) + " nm" + "\n" + "Aperture length y-dim = " + str(ApertureLengthY) + " nm" + "\n" + "Wavelength = " + str(Wavelength) + "\n" + "Elapsed time = " + str(end-start) + " s"             
label2 = "Number of slit columns on grating 1 = " + str(NumberOfSlitsX) + "\n" + "Number of slit rows on grating 1 = " + str(NumberOfSlitsY) + "\n" + "Slit length x-dim = " + str(SlitLengthX) + " nm" + "\n" + "Slit length y-dim = " + str(SlitLengthY) + " nm" + "\n" + "Sources per slit x-dim = " + str(SourcesPerSlitX) + "\n" + "Sources per slit y-dim = " + str(SourcesPerSlitY) + "\n" + "Number of observation points x-dim = " + str(NumObservationPointsX) + "\n" + "Number of observation points y-dim = " + str(NumObservationPointsY) + "\n" + "Observation screen length x-dim = " + str(OSLengthX) + " nm" + "\n" + "Observation screen length y-dim = " + str(OSLengthY) + " nm" + "\n" + "Distance between gratings = " + str(D) + "\n" + "Wavelength = " + str(Wavelength) + "\n" + "Elapsed time = " + str(end-start) + " s"             
o1 = np.array([i[0] for i in o])
o2 = np.array([i[1] for i in o])
O1 = np.reshape(o1, (NumObservationPointsX, NumObservationPointsY))
O2 = np.reshape(o2, (NumObservationPointsX, NumObservationPointsY))
i2 = np.reshape(I2, (NumObservationPointsX, NumObservationPointsY))

# Graph 1: Point Sources Positions in Slits of the First Grating For 3D Muonium Simulation
fig = plt.figure()
y1 = np.array([i[0] for i in y])
y2 = np.array([i[1] for i in y])
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
plt.savefig('1_Speed2/muonium3DFirstGrating.png')
plt.close(fig)

# Graph 2: Intensity Distribution on Second Grating for 3D Munoium Simulation
fig = plt.figure()
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
plt.savefig('1_Speed2/muonium3DSecondGrating.png')
plt.close(fig)

# Graph 3: Intensity Colormap of Second Grating for 3D Munoium Simulation
fig = plt.figure()
plt.imshow(i2, cmap='viridis', extent=[-OSLengthX/2, OSLengthX/2, -OSLengthY/2, OSLengthY/2])
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
plt.savefig('1_Speed2/muonium3DSecondGratingColormap.png')
plt.close(fig)



