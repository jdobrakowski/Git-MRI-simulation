import sys
import os

sciezka_do_katalogu = '/Users/jandobrakowski/Documents/Studia/' #tam jest class_voxel.py
if sciezka_do_katalogu not in sys.path:
    sys.path.append(sciezka_do_katalogu)

import matplotlib.pyplot as plt
import numpy as np
import class_voxel
from class_voxel import Voxel, import_real_B0_field, load_k_space_to_matrix

def przesun_sygnal(sygnal, B1_freq, B_prim, DT):
    #Przesuwa sygnał z założeniem, że pole magnetyczne powinno wynosić B0, a wynosiło B_prim
    dw = B1_freq - Voxel.gammaHz*B_prim*1e-6
    N = len(sygnal)
    if N == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    # Stworzenie wektora czasu odpowiadającego próbkom
    t = np.arange(0, N) * DT 
    return sygnal * np.exp(1j * dw * t)

def filtruj_pasmo(sygnal, f_min, f_max, DT):
    #z sygnału wycina częstotliwości poza pasmem (f_min, f_max)
    N = len(sygnal)
    widmo = np.fft.fft(sygnal)
    os_czestotliwosci = np.fft.fftfreq(N, DT) 
    maska_filtru = (np.abs(os_czestotliwosci) >= f_min) & (np.abs(os_czestotliwosci) <= f_max)
    widmo_przefiltrowane = widmo * maska_filtru

    return np.real(np.fft.ifft(widmo_przefiltrowane))

def popraw_wiersz_k_space(wiersz, j, bandwidth, N, B0_real, B1_freq):
    #bede skalowal kazda wartosc z osobna
    #jesli pole zmienia sie z b0 na b', to czestosc skaluje sie razy b'/b0.
    #powinienem narysowac wykres p(omega) = ile procent sygnalu jest emitowane z dana czestotlioscia zamiast b0
    B = []

    p = np.zeros(N, float)
    for i in range(N):
        time = j*N + i
        B.append(B0_real[time])
        #time = polarization_ends[j] + time_points[i]
        #B.append((Voxel.B0_real[time] + Voxel.B0_real[time+1] + Voxel.B0_real[time+2])/3)
    B = np.array(B)
    B_min = np.min(B)
    B_max = np.max(B)
    dB = (B_max - B_min)/N
    
    for B_i in B:
        p[int((B_i*0.9999 - B_min)//dB)] += 1/N
    
    #Teraz robie n = N kopii k space i każdą transformuję wg wagi pola B0 w ustalonym zakresie, mnoze przez wagi p[] i dodaję z powrotem
    sygnal_poprawiony = np.zeros(N, complex)
    DT = 1/bandwidth
    for i in range(N):
        sygnal_poprawiony += p[i]*przesun_sygnal(wiersz, B1_freq, i*dB + B_min + dB/2, DT)
    
    return sygnal_poprawiony

def popraw_wiersz_do_sredniej(wiersz, j, bandwidth, B0_real):
    B = []
    for i in range(bandwidth):
        time = j*bandwidth + i
        B.append(B0_real[time])

    DT = 1/bandwidth

    return przesun_sygnal(wiersz, np.mean(B0_real), np.mean(B), DT)

#B0_data_path = '/Users/jandobrakowski/Documents/Studia/Pomiary 30 czerwca/Grad Echo 30 czerwca Z 10-15-kopia.csv'
#B0_data_path = '/Users/jandobrakowski/Documents/Studia/Pomiary 30 czerwca/Grad Echo 30 czerwca Y 10-16 -kopia.csv'
B0_data_path = '/Users/jandobrakowski/Documents/Studia/Pomiary 30 czerwca/Grad Echo YZ 1-3-kopia.csv'
import_real_B0_field(B0_data_path)


#B0_dt = (9.948490583E0 - 1.130729169E-2)/1000#pomiar XY nr 5
#B0_dt = (9.956118042E0-1.823466690E-2)/1000 #Pomiar Y nr 16
#B0_dt = (9.952965625E0-1.555450028E-2)/1000 #Pomiar Z nr 14
B0_dt = (9.949946000E0 -1.278149989E-2)/1000 #Pomiar YZ nr 3 (30 czerwca)

kolejnosc_wierszy = [16, 15, 17, 14, 18, 13, 19, 12, 20, 11, 21, 10, 22, 9,23, 8,24,7,25,6,26,5,27,4,28,3,29,2,30,1,31,0] #tablica przechowująca indeksy wierszy w k_space w kolejnosci ich zapisywania przez MRI
czas_wiersza = np.copy(kolejnosc_wierszy)
for i in range(32):
    czas_wiersza[kolejnosc_wierszy[i]] = i

polarization_ends = [] #tablica z indeksami momentów końca czasu polaryzacji
#polarization_ends.append(3972) #pomiar5
#polarization_ends.append(1111) #pomiar4

#Poniższe dane należy przepisać z pliku acqu.par
Nx = 32 #Nread w acqu.par
Ny = 32 #Nphase1 w acqu.par
TE = 200*1e-3
B1_freq = 1878 #częstotliwość pobudzania
t_grad = 50*1e-3#pomiar Z nr 14 i Y nr 16 i YZ nr 3
FOVx = 120 #[mm]
FOVy = 240 #[mm]
#bandwidth = 32 #Pomiar Z nr 14
#bandwidth = 16 #Pomiar Y nr 16
bandwidth = 16 #Pomiar YZ nr 3
t_read = Nx/bandwidth
t_read_probki = t_read/B0_dt
t_delay = t_grad + 70*1e-3
t_delay_probki = t_delay//B0_dt
print("B0_dt = ", B0_dt)
print("t_read = ",t_read)
print("t_delay_probki = ",t_delay_probki)
print("Skok_probki = ", int(t_read_probki//Nx))
print("Skok polarization = ", int((43803-18647)//Ny))

# plt.plot(Voxel.B0_real)

#Poniższą liczbę należy odczytać z powyższego wykresu pola Voxel.B0_real
polarization_ends.append(18647)#pomiar YZ nr 3
#polarization_ends.append(16055)#pomiar Y nr 16
#polarization_ends.append(9808)#pomiar Z nr 14
# plt.axvline(x=18647, color='red')
for i in range(polarization_ends[0]+2, len(Voxel.B0_real)-2, 1):
    if np.abs(Voxel.B0_real[i+1] - Voxel.B0_real[i]) <5 and Voxel.B0_real[i-1] - Voxel.B0_real[i]>5 and Voxel.B0_real[i-2] - Voxel.B0_real[i-1]>5:
        polarization_ends.append(i+1)
        # plt.axvline(x=i+1, color='red')
    
# print("Liczba znalezionych końców: ", len(polarization_ends))
# plt.show()
# plt.clf()


#print(polarization_ends[-1])
time_points = []#interwały 
time_points.append(t_delay_probki)
for i in range(Nx-1):
    time_points.append(time_points[0]+ int((i+1)*t_read_probki/Nx))

B0_real = []
#plt.figure(figsize=(10, 6)) # Możesz dostosować rozmiar wykresu

for i in polarization_ends:
    #plt.axvline(x=num*96, color='red', linestyle='--', linewidth=1.5)
    for j in time_points:
        time = i + j

        B0_real.append(np.mean(Voxel.B0_real[int(time):int(time+t_read_probki//Nx)]))
        """
        Test: B0_rand losowe z tą samą średnią i odchyleniem standardowym co B0_real
        """
        #plt.axvline(x=time, color='r', linestyle='--')

odchyl_stand = np.std(B0_real)
print("Odchylenie standardowe: ", odchyl_stand)
B0_rand = np.random.randn(len(B0_real))
B0_rand = B0_rand*odchyl_stand + np.mean(B0_real)


# plt.plot(B0_real)
# plt.show()
# plt.clf()

"""
sred = []
for i in range(32):
    sred.append(np.mean(B0_real[i*32:(i+1)*32]))
    #print("Średnia pola wiersza nr ", kolejnosc_wierszy[i], " to: ", sred[-1])

plt.plot(sred)
plt.show()
"""

"""
plt.plot(B0_real, label = "prawdziwe")
#plt.plot(B0_rand, label = "losowe")
plt.legend()
plt.show()
"""

#plt.show()
B0 = np.mean(B0_real)
#print(B0)
#plt.plot(B0)
#plt.show()


#k_space = load_k_space_to_matrix("/Users/jandobrakowski/Documents/Studia/Pomiary 30 czerwca/PomiarY numer 16.txt") #pomiar Y nr 16
#k_space = load_k_space_to_matrix("/Users/jandobrakowski/Documents/Studia/Pomiary 30 czerwca/Pomiar Z numer 14.txt")#pomiar Z nr 14
# k_space_popr = popraw_wiersz_k_space(k_space, 0, bandwidth,N, B0_real)
# window_1D = np.hamming(N)
# obraz1D_popr = np.fft.fft(k_space_popr)
# obraz1D_popr_shifted = np.fft.fftshift(obraz1D_popr)
# obraz1D = np.fft.fft(k_space)
# obraz1D_shifted = np.fft.fftshift(obraz1D)

# freqs = np.fft.fftfreq(len(k_space), d=1/bandwidth)
# freqs_shifted = np.fft.fftshift(freqs)
# plt.plot(np.abs(obraz))
# plt.show()
"""
k_space5 = load_k_space_to_matrix("/Users/jandobrakowski/Documents/Studia/PomiaryMRI z 20 czerwca i poprzednie/Pomiar YZ 20 czerwca nr 5/gradecho_ztelefonem5_kspace-kopia.txt")
k_space5 = np.array(k_space5)
"""

# fig, axes = plt.subplots(1, 2, figsize=(11, 5))
# im1 = axes[0].plot(freqs_shifted,np.abs(obraz1D_shifted))
# im1 = axes[1].plot(freqs_shifted,np.abs(obraz1D_popr_shifted))
# axes[0].set_title('surowy obraz pomiary Grad Echo Y nr 16')
# axes[1].set_title('poprawiony wg zmierzonego B0')
# plt.show()

k_space = np.array(load_k_space_to_matrix("/Users/jandobrakowski/Documents/Studia/Pomiary 30 czerwca/Pomiar YZ numer 3.txt"))#pomiar YZ nr 3
k_space_popr = np.zeros((Nx,Ny), complex)

for i in range(Ny):
    print(i)
    #k_space_popr[i,:] = popraw_wiersz_do_sredniej(np.copy(k_space[i,:]), czas_wiersza[i], 32, np.copy(B0_real))
    k_space_popr[kolejnosc_wierszy[i],:] = popraw_wiersz_k_space(k_space[kolejnosc_wierszy[i],:], i,bandwidth, Nx, B0_real, B1_freq)

# window_1d_x = np.hamming(Nx)
# window_1d_y = np.hamming(Ny)
# window_2d = window_1d_y[:, np.newaxis] * window_1d_x[np.newaxis, :]
# obraz2D = np.fft.fft2(k_space)
# obraz2D_popr = np.fft.fft2(k_space_popr)
# obraz2D_shifted = np.fft.fftshift(obraz2D)
# obraz2D_popr_shifted = np.fft.fftshift(obraz2D_popr)


#ucinam k_space zostawiając 3 środkowe wiersze, reszta zera
# Tworzymy kopię macierzy, aby nie modyfikować oryginalnej
mod_matrix_popr = k_space_popr.copy()
mod_matrix = k_space.copy()

# Wyliczamy indeksy środkowych wierszy
# Dla macierzy 32x32 środkowe wiersze to 14, 15, 16
middle_row_start = (k_space.shape[0] // 2) - 5
middle_row_end = (k_space.shape[0] // 2) + 5# +2, bo wycinek jest wyłączny na końcu

# Tworzymy macierz zer o tych samych wymiarach
zeros_matrix_popr = np.zeros_like(k_space_popr)
zeros_matrix = np.zeros_like(k_space)

# Kopiujemy trzy środkowe wiersze z oryginalnej macierzy do macierzy zer
zeros_matrix[middle_row_start:middle_row_end, :] = mod_matrix[middle_row_start:middle_row_end, :]
zeros_matrix_popr[middle_row_start:middle_row_end, :] = mod_matrix_popr[middle_row_start:middle_row_end, :]

obraz_modyfikowany = np.fft.fft2(zeros_matrix)
obraz_modyfikowany_popr = np.fft.fft2(zeros_matrix_popr)

obraz2D_shifted = np.fft.fftshift(obraz_modyfikowany)
obraz2D_popr_shifted = np.fft.fftshift(obraz_modyfikowany_popr)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

extent = (FOVx, 0, 0, FOVy)

im1 = axes[0].imshow(np.abs(obraz2D_shifted), cmap='jet', extent=extent, aspect='equal')
axes[0].set_title('Surowy obraz Grad Echo YZ nr 3')
axes[0].invert_xaxis()

im2 = axes[1].imshow(np.abs(obraz2D_popr_shifted), cmap='jet', extent=extent, aspect='equal')
axes[1].set_title('Poprawiony obraz')
axes[1].invert_xaxis()

fig.colorbar(im1, ax=axes[0])
fig.colorbar(im2, ax=axes[1])

plt.show()
