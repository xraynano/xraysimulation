from numpy import sqrt, exp, pi
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from numba import jit, prange
import numpy as np
# import matplotlib.colors as colors
# from matplotlib import cm

start = timer()

Wavelength = 0.56
K = 2 * pi / Wavelength
NumberOfSlitsX = 1
NumberOfSlitsY = 1
SlitLengthX = 50
SlitLengthY = 50
SourcesPerSlitX = 500
SourcesPerSlitY = 50
ApertureLengthX = NumberOfSlitsX * SlitLengthX + (NumberOfSlitsX - 1)*50
ApertureLengthY = (2*NumberOfSlitsY - 1) * SlitLengthY
SourceSpacingX = SlitLengthX / SourcesPerSlitX
SourceSpacingY = SlitLengthY / SourcesPerSlitY
NumObservationPointsX = 10000
NumObservationPointsY = 10000
GratingLengthX = 1e7
GratingLengthY = 1e7
NumSpacingX = GratingLengthX / NumObservationPointsX
NumSpacingY = GratingLengthY / NumObservationPointsY
D2 = 5e7
D3 = 10e7
InitialAmplitude = 1
InitialPhase = 1
y = np.array([(-ApertureLengthX/2 + i*(SlitLengthX + 50) + SourceSpacingX/2 + j*SourceSpacingX, -ApertureLengthY/2 + k*2*SlitLengthY + SourceSpacingY/2 + l*SourceSpacingY) for k in range(NumberOfSlitsY) for l in range(SourcesPerSlitY) for i in range(NumberOfSlitsX) for j in range(SourcesPerSlitX)])
o = np.array([(-GratingLengthX/2 + NumSpacingX/2 + i*NumSpacingX, -GratingLengthY/2 + NumSpacingY/2 + j*NumSpacingY) for j in range(NumObservationPointsY) for i in range(NumObservationPointsX)])
i2 = np.zeros(NumObservationPointsX * NumObservationPointsY)

@jit(nopython=True, parallel=True, fastmath=True)
def calcs(y, o, I2, k, numberOfSlitsX, numberOfSlitsY, slitLengthX, slitLengthY, sourcesPerSlitX, sourcesPerSlitY, numObservationPointsX, numObservationPointsY, gratingLengthX, gratingLengthY, d2, initialAmplitude, initialPhase):
    sourceSpacingX = slitLengthX / sourcesPerSlitX
    sourceSpacingY = slitLengthY / sourcesPerSlitY
    for i in prange(numObservationPointsX * numObservationPointsY):
        amplitudeSum = 0
        for j in prange(numberOfSlitsX * sourcesPerSlitX * numberOfSlitsY * sourcesPerSlitY):
            r = sqrt((d2)**2 + (o[i][0] - y[j][0])**2 + (o[i][1] - y[j][1])**2)
            phase = exp(1j * k * r) * initialPhase
            U = initialAmplitude * phase / r
            amplitudeSum = amplitudeSum + U * sourceSpacingX * sourceSpacingY
        I2[i] = amplitudeSum.real ** 2 + amplitudeSum.imag ** 2 
    return I2

I2 = calcs(y, o, i2, K, NumberOfSlitsX, NumberOfSlitsY, SlitLengthX, SlitLengthY, SourcesPerSlitX, SourcesPerSlitY, NumObservationPointsX, NumObservationPointsY, GratingLengthX, GratingLengthY, D2, InitialAmplitude, InitialPhase)

o1 = np.array([i[0] for i in o])
o2 = np.array([i[1] for i in o])
O1 = np.reshape(o1, (NumObservationPointsX, NumObservationPointsY))
O2 = np.reshape(o2, (NumObservationPointsX, NumObservationPointsY))
i2 = np.reshape(I2, (NumObservationPointsX, NumObservationPointsY))

N = 20
fig = plt.figure()
y1 = np.array([i[0] for i in y])
y2 = np.array([i[1] for i in y])
A = max(ApertureLengthX, ApertureLengthY)
plt.plot(y1,y2, 'bo')
plt.xlabel('X-Position on Aperture (nm)', fontsize=1.5*N)
plt.ylabel('Y-Position on Aperture (nm)', fontsize=1.5*N)
plt.title('Point Sources Positions in Slits of the First Grating \n For Muonium Simulation', fontsize=2*N)
plt.xlim(-A/2, A/2)
plt.ylim(-A/2, A/2)
fig.set_size_inches(N,N)
plt.xticks(fontsize=1.5*N)
plt.yticks(fontsize=1.5*N)
# plt.set_aspect(1)
# plt.zticks(fontsize=1.5*N)
plt.gca().xaxis.get_offset_text().set_fontsize(1.5*N)
plt.gca().yaxis.get_offset_text().set_fontsize(1.5*N)
# plt.gca().zaxis.get_offset_text().set_fontsize(1.5*N)
plt.savefig('1_Speed2/muonium3DFirstGrating.png')
plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plt.plot(o1, o2, I2, 'bo')
# plt.xlabel('X-Position on \n Second Grating (nm)', fontsize=1.5*N) 
# plt.ylabel('Y-Position on \n Second Grating (nm)', fontsize=1.5*N)
plt.title('Intensity Distribution on Second Grating \n for 3D Munoium Simulation', fontsize=2*N)
fig.set_size_inches(N,N)
# plt.xticks(fontsize=1.5*N)
# plt.yticks(fontsize=1.5*N)
# ax.set_zticks(1.5*N)
# plt.zticks(fontsize=1.5*N)
# ax.set_zticks()
ax.tick_params(labelsize=1.5*N)
ax.set_xlabel('X-Position on \n Second Grating (nm)', fontsize=1.5*N, labelpad=2*N)
ax.set_ylabel('Y-Position on \n Second Grating (nm)', fontsize=1.5*N, labelpad=2*N)
ax.set_zlabel('Intensity', fontsize=1.5*N, labelpad=2*N)
# ax.xaxis._axinfo['label']['space_factor'] = 20
# ax.set_fontsize()
plt.gca().xaxis.get_offset_text().set_fontsize(1.5*N)
plt.gca().yaxis.get_offset_text().set_fontsize(1.5*N)
plt.gca().zaxis.get_offset_text().set_fontsize(1.5*N)
plt.savefig('1_Speed2/muonium3DSecondGrating.png')
plt.close(fig)

fig = plt.figure()
plt.imshow(i2, cmap='viridis', extent=[-GratingLengthX/2, GratingLengthX/2, -GratingLengthY/2, GratingLengthY/2])
cbar = plt.colorbar()
plt.xlabel('X-Position on Second Grating (nm)', fontsize=1.5*N)
plt.ylabel('Y-Position on Second Grating (nm)', fontsize=1.5*N)
plt.title('Intensity Colormap of Second Grating \n for 3D Munoium Simulation', fontsize=2*N)
fig.set_size_inches(N,N)
plt.xticks(fontsize=1.5*N)
plt.yticks(fontsize=1.5*N)
plt.gca().xaxis.get_offset_text().set_fontsize(1.5*N)
plt.gca().yaxis.get_offset_text().set_fontsize(1.5*N)
cbar.ax.tick_params(labelsize=1.5*N)
# cb.get_offset_text().set_fontsize(1.5*N)
cbar.ax.yaxis.get_offset_text().set_fontsize(1.5*N)
plt.savefig('1_Speed2/muonium3DSecondGratingColormap.png')
plt.close(fig)

end = timer()
print("Time for program to run =", end - start, "seconds.")