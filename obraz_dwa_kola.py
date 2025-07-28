"""
Modeluję obraz fantomu 2d złożonego z dwóch kól wypełnionych wodą. Chcę uzyskać maksymalne podobieństwo do tego, 
co mam w rzeczywistych pomiarach. Również rozdzielczość
"""
import matplotlib.pyplot as plt
import numpy as np
import csv
from class_voxel import Voxel, gradient_echo_sequence, generate_circle, show_fantom, show_rho_2D, show_S_image, parallel_GE_sequence, gradient_echo_x_axis, parallel_Grad_echo_x, create_single_voxel_fantom

TR = 0.8
TE = 200 * 1e-3 #(pierwsza liczba to ms)
bandwidth = 32
T_RF = 10*1e-3
flip_angle = 35*np.pi/180 
slice_z = 0.0  # Położenie wybranej płaszczyzny w osi Z
Gzz = 0.1
RF_amplitude = 3.67 * 1e-6  # Amplituda RF w Teslach
Nx = 32  # Liczba punktów pomiarowych w osi X
Ny = 32  # Liczba punktów pomiarowych w osi Y
FOVx = 120*1e-3 
FOVy = 120 *1e-3
sampling_frequency = 7000
nazwa_pomiaru = "Pomiar_XY_32x32_2"
file_name = f"{nazwa_pomiaru}.txt"

# Open the file in write mode ('w')
with open(file_name, 'w') as file:
    file.write(f"dt = {Voxel.dt}\n")
    file.write(f"TR = {TR}\n")
    file.write(f"TE = {TE}\n")
    file.write(f"bandwidth = {bandwidth}\n")
    file.write(f"T_RF = {T_RF}\n")
    file.write(f"flip_angle = {flip_angle}\n")
    file.write(f"slice_z = {slice_z}\n")
    file.write(f"Gzz = {Gzz}\n")
    file.write(f"RF_amplitude = {RF_amplitude}\n")
    file.write(f"Nx = {Nx}\n")
    file.write(f"Ny = {Ny}\n")
    file.write(f"FOVx = {FOVx}\n")
    file.write(f"FOVy = {FOVy}\n")
    file.write(f"sampling_frequency = {sampling_frequency}\n")

kolo1 = generate_circle(1.2*1e-2, -1.7*1e-2, 0*1e-2, Nx, Ny)
kolo2 = generate_circle(1.2*1e-2, 1.7*1e-2, 0*1e-2, Nx, Ny)
fantom = kolo1.copy() # Tworzymy kopię pierwszego fantomu, aby nie modyfikować oryginału
fantom.update(kolo2)

# fantom = {}
# create_single_voxel_fantom(fantom, 0.035, 0.020, 0.0)
# create_single_voxel_fantom(fantom, -0.035, -0.020, 0.0)
#show_rho_2D(fantom, 0.0, Nx, Ny)

if __name__ == "__main__":
    # S = parallel_GE_sequence(fantom, 8,TR, TE, T_RF, flip_angle, slice_z, Gzz,Nx, Ny, sampling_frequency, bandwidth, FOVx, FOVy)
    # np.save(f"{nazwa_pomiaru}.npy", S)
#     S = gradient_echo_x_axis(fantom, T_RF, Nx, flip_angle, slice_z, Gzz, sampling_frequency,bandwidth, FOVx)
#     np.save("S_pomiar_x.npy", S)
    # S = parallel_Grad_echo_x(fantom, 8, T_RF, Nx, flip_angle, slice_z, Gzz, sampling_frequency,bandwidth, FOVx)
    # np.save("S_parallel_FOV120_x.npy", S)
    #show_S_image(S, TR, TE, T_RF, Nx, Ny, sampling_frequency)
    # print("Częstotliwość f0 = ", Voxel.B0 * Voxel.gammaHz)
    # # k_space = np.load("S_pomiar_x.npy")
    # # k_space = np.load("S_parallel_x.npy")
    # k_space = np.load("S_parallel_FOV120_x.npy")
    # x_vals = np.arange(-FOVx/2, FOVx/2, FOVx/len(k_space))
    # # print(k_space)
    # obraz1D = np.fft.fft(k_space)
    # #freqs = np.fft.fftshift(np.fft.fftfreq(len(obraz1D), d=1/sampling_frequency))
    # plt.plot(x_vals,np.abs(np.fft.fftshift(obraz1D)))
    # # t = np.arange(0.0, 0.76, 0.76/len(k_space))
    # # plt.plot(t,np.abs(k_space))
    # plt.show()
    
    k_space = np.load(f"{nazwa_pomiaru}.npy")
    image = np.fft.fftshift(np.fft.fft2(k_space))
    absolute_image = np.abs(image)
    #print(matrix1)
    
    plt.figure(figsize=(6, 6))
    extent = [0.0, FOVx*100, 0.0, FOVy*100]
    plt.imshow(np.abs(image).T, cmap='jet', interpolation='nearest', origin = 'lower', extent=extent, aspect='equal')
    plt.colorbar(label='Wartość')
    plt.title('Heatmapa danych')
    plt.xlabel('Oś X [cm]')
    plt.ylabel('Oś Y [cm]')
    plt.tight_layout()
    plt.show()