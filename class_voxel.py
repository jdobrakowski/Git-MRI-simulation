#Klasa voxel
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from numba import njit
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import os
import webbrowser
from matplotlib import rcParams
rcParams['animation.embed_limit'] = 100  # Limit w MB (np. 100 MB)

"""
Idea multoprocesingu: podczas obliczeń w symulacji często korystamy z wielu wokseli, które mogą być obliczane równolegle.
Czyli kazdą pętlę po wokselach mogę dzzielić na rdzenia procesora.
"""

class Voxel:
    #Parametry ogólne
    FOVx = 5.6*1e-2 #maksymalny rozmiar fantomu w osi x w metrach (Field of view)
    FOVy = 5.6*1e-2 #maksymalny rozmiar fantomu w osi y w metrach (Field of view)

    #Stała żyromagnetyczna
    gammaHz = 42.58 * 1e6  # w jednostkach Hz/T
    gammaRad = 2.675 * 1e8  # w jednostkach rad/(s*T)

    #Wartość stałego pola magnetycznego
    B0 = 60*1e-6 # Średnia wartość B_z w Teslach. 

    #Gęstość voxeli
    dx = 0.5*1e-2 #Wartość w metrach.
    dy = 0.5*1e-2 #Wartość w metrach.
    dz = 0.5*1e-2 #Wartość w metrach.

    #Pasmo odbiornika
    BW = 64 #Pasmo odbiornika w Hz

    #Promień cewki odbiorczej
    coil_radius = 3*1e-2 #3 cm

    #Krok czasowy symulacji
    dt = 1e-6 #Wartość w sekundach
    actual_time = 0.
    stage = []
    B0_real = []
    B0_data_dt = 0.009938000003
   
    def __init__(self, x, y, z, proton_density, t1, t2, t2_star, m0=0.015):
        self.x = x
        self.y = y
        self.z = z
        self.proton_density = proton_density
        self.T1 = t1
        self.T2 = t2
        self.T2_star = t2_star
        self.M0 = m0 * proton_density
        self.magnetization = np.array([0.0, 0.0, self.M0])# Przechowywanie aktualnego stanu magnetyzacji (Mx, My, Mz)
        self.B = np.array([0.0, 0.0, 0.0]) # Na początku zawsze pole magnetyczne w miejscu woksela wynosi 0
        self.previous_B = np.array([0.0, 0.0, 0.0]) # pole magnetyczne B w poprzedniej iteracji
        self.magnetization_snapshots = []
        self.B_snapshots = []
    
    def set_magnetization_in_direction(self, theta, phi):
        mx = self.M0 * np.sin(theta) * np.cos(phi)
        my = self.M0 * np.sin(theta) * np.sin(phi)
        mz = self.M0 * np.cos(theta)
        self.magnetization = np.array([mx, my, mz])

    def change_magnetization_to(self, M):
        self.magnetization = M

def precession(voxels_set, global_time): #Wykonuje precesję magnetyzacji woksela w zadanym polu magnetycznym o czas dt
    def f(M, B, M0, T1, T2, gammaHz):
        M_cross_B = np.cross(M, B)  # Moment magnetyczny z precesji
        result = gammaHz * M_cross_B  # Jeszcze bez relaksacji
        result[0] -= M[0]/T2  # Zanikanie Mx
        result[1] -= M[1]/T2  # Zanikanie My
        result[2] -= (M[2]-M0)/T1
        return result

    for voxel in voxels_set.values():  
        M = voxel.magnetization
        B = voxel.B + np.array([0., 0., dB_noise_function(global_time)]) #dodaję szum do pola B
        previous_B = voxel.previous_B
        T1 = voxel.T1
        T2 = voxel.T2
        #Korzystam z metody Rungego-Kutty 4 rzędu
        k1 = Voxel.dt * f(M, previous_B, voxel.M0, T1, T2, Voxel.gammaRad)
        k2 = Voxel.dt * f(M + 0.5 * k1, (B+previous_B)/2, voxel.M0, T1, T2, Voxel.gammaRad)
        k3 = Voxel.dt * f(M + 0.5 * k2, (B+previous_B)/2, voxel.M0, T1, T2, Voxel.gammaRad)
        k4 = Voxel.dt * f(M + k3, B, voxel.M0, T1, T2, Voxel.gammaRad)

        M_new = M + (k1 + 2*k2 + 2*k3 + k4)/6
       
        # Aktualizacja magnetyzacji dla voxela
        voxel.change_magnetization_to(M_new)

def non_relaxation_precession(voxels_set): #Wykonuje precesję magnetyzacji woksela bez relaksacji
    for voxel in voxels_set.values():  
        M = voxel.magnetization
       
        M_cross_B = np.cross(M, voxel.B)  # Moment magnetyczny z precesji
        M_new = M + Voxel.gammaRad * M_cross_B * Voxel.dt 
                      
        # Aktualizacja magnetyzacji dla voxela
        voxel.change_magnetization_to(M_new)

@njit
def f(M, B, M0, T1, T2, gammaHz, dt):
    M_cross_B = np.cross(M, B)  # Moment magnetyczny z precesji
    result = gammaHz * M_cross_B  # Jeszcze bez relaksacji
    result[0] -= M[0]/T2  # Zanikanie Mx
    result[1] -= M[1]/T2  # Zanikanie My
    result[2] -= (M[2]-M0)/T1
    return result * dt

@njit
def precession_numba_time(magnetization, B, previous_B, M0, T1, T2, gammaRad, dt, time):
    """
    Funkcja wykonująca precesję magnetyzacji dla wszystkich wokseli sekwencyjnie.
    """
    num_voxels = magnetization.shape[0]  # Liczba wokseli
    for t in time:
        for i in range(num_voxels):  # Przetwarzanie sekwencyjne
            M = magnetization[i]
            B_now = B[i]
            B_prev = previous_B[i]

            k1 = f(M, B_prev, M0[i], T1[i], T2[i], gammaRad, dt)
            k2 = f(M + 0.5 * k1, (B_now + B_prev) / 2, M0[i], T1[i], T2[i], gammaRad, dt)
            k3 = f(M + 0.5 * k2, (B_now + B_prev) / 2, M0[i], T1[i], T2[i], gammaRad, dt)
            k4 = f(M + k3, B_now, M0[i], T1[i], T2[i], gammaRad, dt)

            # Nowa wartość magnetyzacji
            magnetization[i] += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return magnetization

def fast_long_precession(voxels_set, T):
    """
    Funkcja wywołująca przyspieszoną wersję `precession_numba` na wokselach.

    Przyspiesza symulację około 50 razy!
    """

    # Tworzymy macierze NumPy dla magnetyzacji, pola B i parametrów
    magnetization = np.array([voxel.magnetization for voxel in voxels_set.values()])
    B = np.array([voxel.B for voxel in voxels_set.values()])
    previous_B = np.array([voxel.previous_B for voxel in voxels_set.values()])
    M0 = np.array([voxel.M0 for voxel in voxels_set.values()])
    T1 = np.array([voxel.T1 for voxel in voxels_set.values()])
    T2 = np.array([voxel.T2 for voxel in voxels_set.values()])
    
    # Pobranie stałych wartości
    gammaRad = Voxel.gammaRad
    dt = Voxel.dt
    time = np.arange(0., T, dt)
    # Uruchomienie przyspieszonej wersji z Numbą (sekwencyjnie)
    magnetization = precession_numba_time(magnetization, B, previous_B, M0, T1, T2, gammaRad, dt, time)
    
    # Przypisanie nowych wartości magnetyzacji do obiektów Voxel
    for i, voxel in enumerate(voxels_set.values()):
        voxel.magnetization = magnetization[i]


#Widok na ustalona płaszczyzne stałego z
def show_slice_in_z(voxels_set, z_fixed):
    x = []
    y = []
    
    # Tworzenie podmapy dla ustalonego z_fixed
    voxels_xy = {(x, y): value for (x, y, z), value in voxels_set.items() if z == z_fixed}

    return voxels_xy

def show_fantom_time_3d(voxels_set,n, xlim, ylim, zlim, speed): 
    """
    xlim - tuple (a,b)
    """
    xlim = list(xlim)
    ylim = list(ylim)
    zlim = list(zlim)
    xlim[0] *= 1e2#zamieniam na cm
    ylim[0] *= 1e2
    zlim[0] *= 1e2
    xlim[1] *= 1e2#zamieniam na cm
    ylim[1] *= 1e2
    zlim[1] *= 1e2

    voxels = voxels_set  #tylko zeby nie zmieniać dalej w kodzie
    dt = Voxel.dt

    # Tworzenie figury z dwoma podwykresami
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    titles = ["XY Plane", "XZ Plane"]
    planes = ['xy', 'xz']
    vectors = {'xy': [], 'xz': []}

    for ax, plane, title in zip(axes, planes, titles):
        if plane == 'xy':
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        elif plane == 'xz':
            ax.set_xlim(xlim)
            ax.set_ylim(zlim)

        ax.set_title(title)
        ax.set_xlabel(plane[0].upper())
        ax.set_ylabel(plane[1].upper())
        ax.grid()
        ax.set_aspect('equal', adjustable='box')

    # Inicjalizacja wektorów dla każdej płaszczyzny
    for voxel in voxels.values():
        position = np.array((voxel.x*1e2, voxel.y*1e2, voxel.z*1e2))
        history = voxel.magnetization_snapshots

        for plane, ax in zip(planes, axes):
            if plane == 'xy':
                vector = ax.quiver(position[0], position[1], 1, 0, angles='xy', scale_units='xy', scale=1)
            elif plane == 'xz':
                vector = ax.quiver(position[0], position[2], 1, 0, angles='xy', scale_units='xy', scale=1)

            vectors[plane].append({
                'vector': vector,
                'position': position,
                'history': history,
            })

    # Dodanie tekstu wyświetlającego numer klatki
    frame_text = axes[0].text(0.05, 0.95, '', transform=axes[0].transAxes, fontsize=12, verticalalignment='top')
    stage_text = axes[0].text(0.05, 0.85, '', transform=axes[0].transAxes, fontsize=10, verticalalignment='top')

    # Funkcja animująca
    def animate(frame_index):
        i = frame_index
        for plane in planes:
            for vec in vectors[plane]:
                if i + 1 < len(vec['history']):
                    end_position = vec['history'][i]*1e2

                    if plane == 'xy':
                        dx, dy = end_position[0], end_position[1]
                        vec['vector'].set_UVC(dx, dy)
                    elif plane == 'xz':
                        dx, dz = end_position[0], end_position[2]
                        vec['vector'].set_UVC(dx, dz)

        frame_text.set_text(f"Time: {i * n * dt * 1000:.2f} ms")
        stage_text.set_text(f"Faza: {Voxel.stage[i]}")
        return [vec['vector'] for vec in vectors['xy']] + \
            [vec['vector'] for vec in vectors['xz']] + [frame_text]



    # Liczba klatek
    max_frames = len(list(vectors['xy'][0]['history'])) - 1
    ani = FuncAnimation(fig, animate, frames=max_frames, interval=5/speed, blit=True)

    # Generowanie HTML
    html_content = ani.to_jshtml()

    # Dodanie stylu dla lepszej czytelności
    resized_html = f"""
    <div style="width: 100%; height: auto; transform: scale(1); transform-origin: top left; margin: 0 auto;">
        {html_content}
    </div>
    """

    # Zapisanie do pliku HTML
    with open("fantom_animation_two_planes.html", "w") as f:
        f.write(resized_html)

    # Otwieranie animacji w przeglądarce
    output_file = "fantom_animation_two_planes.html"
    webbrowser.open(f"file://{os.path.abspath(output_file)}")
    print("Animacja zapisana jako HTML: fantom_animation_two_planes.html")
    plt.clf()

def generate_RF_signal(voxels_set, frequency, time_length, amplitude, n_snapshots):
    """
    Sygnał RF w przeciwieństwie do gradientów, działa przez pewien ustalony czas (wykonując precesję w czasie)

    Działa takze na pewnym ustalonym tle, do którego tylko "dodaje" sygnał w osi x
    """
    dt = Voxel.dt
    iterator = 0
    for t in np.arange(0.,time_length, dt):
        Bx = 2 * amplitude * np.cos(-2*np.pi*frequency * t)
        By = 0.
        Bz = 0.
        B = np.array([Bx, By, Bz])
       
        for voxel in voxels_set.values():
            voxel.previous_B = voxel.B
            voxel.B += B

        precession(voxels_set, t)

        if iterator %n_snapshots == 0:
            stage = "Sygnał RF " + str(np.round(frequency, 2)) + "hz"
            Voxel.stage.append(stage)
            for voxel in voxels_set.values():    
                voxel.magnetization_snapshots.append(voxel.magnetization)
                #print(10*voxel.magnetization)

        for voxel in voxels_set.values():
            voxel.B -= B #wracam do tła bez RF
        
        iterator += 1
        
def gradient(voxels_set, Gx, Gy, Gz):#ustawia aktualny stan pola z gradientami w x, y, z
    """
    Poniewaz gradienty nie zmieniają się tak dynamicznie, jak syngał RF, oraz są wykorzystywane w czasie pomiarów, to
    przyjmuję zasadę, ze raz ustalone, są dopóki nie zostaną zmienione, lub wyzerowane

    Ta funkcja jest niezalezna od czasu. Niejako ustawia tło do innych działań (podobnie zresztą do shimmingu)
    """
    for voxel in voxels_set.values():
        B = np.array([0.0, 0.0, Gz*voxel.z + Voxel.B0 + Gy*voxel.y + Gx*voxel.x])   
        voxel.B = B
        voxel.previous_B = B

def show_S_image(S, TR, TE, T_RF,  Nx, Ny, sampling_frequency):
    Gx = 30/(Voxel.FOVx*Voxel.gammaRad*0.002)
    freq0 = Voxel.B0*Voxel.gammaHz
    freq_max = freq0 + Gx*Voxel.FOVx*Voxel.gammaHz
    DT = 1/sampling_frequency

    S_fft = np.fft.fft2(S)  # FFT 2D
    S_fft_amplitude = np.abs(S_fft).T  # Pobieramy amplitudę obrazu
    # Uzyskanie częstotliwości dla każdej osi

    freqs_x = np.fft.fftfreq(Nx, DT)
    freqs_y = np.arange(Ny)

    print(freqs_x)
    print("freq_x_max: ", freq_max)
    print("freq_x_min: ", freq0)

    T_grad_y = TE - Nx/sampling_frequency - T_RF/2
    base_Gy = 10*np.pi/(Voxel.FOVy*Voxel.gammaHz*T_grad_y)
    freq_y_max = Voxel.FOVy*Voxel.gammaHz*base_Gy*T_grad_y
    #print(freq_y_max)   
    #print(freqs_y)
    a_y = base_Gy*Voxel.gammaHz*T_grad_y
    b_y = 0
    # Przykład użycia częstotliwości do filtrowania lub innych operacji
    a_x = Voxel.FOVx/(freq_max-freq0)
    b_x = -freq0*a_x
    x = []
    index_freq0 = 0
    y = []
    for i in range(len(freqs_x)):
        if freqs_x[i] < freq0 and freqs_x[i] > 0:
            index_freq0 = i
        if freqs_x[i] >= freq0 and freqs_x[i] <= sampling_frequency/2:
            x.append(freqs_x[i]*a_x+b_x)

    for j in range(len(freqs_y)):
        if freqs_y[j] >= 0 and freqs_y[j] <= freq_y_max:
            y.append(freqs_y[j]/a_y + b_y)

    print("len(y): ",len(y))
    print("len(x): ",len(x))
    image = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            image[i, j] = S_fft_amplitude[i + index_freq0, j]
            
    # Wizualizacja obrazu w dziedzinie przestrzennej
    plt.figure(figsize=(6, 6))
    plt.title("Obraz w przestrzeni rzeczywistej (metry)")
    #plt.imshow(S_fft_amplitude, origin='lower',cmap='inferno')#, extent = [freqs_x[0], freqs_x[len(freqs_x)-1], freqs_y[0], 4*freqs_y[len(freqs_y)-1]])
    plt.imshow(image, origin='lower',cmap='inferno', extent = [x[0], x[len(x)-1], y[0], y[len(y)-1]])
    plt.colorbar(label="Amplituda")
    plt.xlabel("Pozycja x (m)")
    plt.ylabel("Pozycja y (m)")
    plt.show()
    plt.clf()

def show_S_image_experimental(S, TR, TE, T_RF,  Nx, Ny, sampling_frequency): #Nie ma sensu
    Gx = 30/(Voxel.FOVx*Voxel.gammaRad*0.002)
    freq0 = Voxel.B0*Voxel.gammaHz
    freq_max = freq0 + Gx*Voxel.FOVx*Voxel.gammaHz
    DT = 1/sampling_frequency

    Sx = S[:, 10]
    Sy = S[5, :]
    Sy_fft = np.fft.fft(Sy)  # FFT 2D
    Sy_fft_amplitude = np.abs(Sy_fft)  # Pobieramy amplitudę obrazu

    Sx_fft = np.fft.fft(Sx)  # FFT 2D
    Sx_fft_amplitude = np.abs(Sx_fft)  # Pobieramy amplitudę obrazu
    # Uzyskanie częstotliwości dla każdej osi

    freqs_x = np.fft.fftfreq(Nx, DT)
    freqs_y = np.arange(Ny)

    #print(freqs_x)
    print("freq_x_max: ", freq_max)
    print("freq_x_min: ", freq0)

    T_grad_y = TE - Nx/sampling_frequency - T_RF/2
    base_Gy = 10*np.pi/(Voxel.FOVy*Voxel.gammaHz*T_grad_y)
    freq_y_max = 1*Voxel.FOVy*Voxel.gammaHz*base_Gy*T_grad_y
    #print(freq_y_max)   
    #print(freqs_y)
    a_y = base_Gy*Voxel.gammaHz*T_grad_y
    # Przykład użycia częstotliwości do filtrowania lub innych operacji
    a_x = Voxel.FOVx/(freq_max-freq0)
    b_x = -freq0*a_x
    x = []
    index_freq0 = 0
    y = []
    image_y = []
    for i in range(len(freqs_y)):
        if freqs_y[i] >= 0 and freqs_y[i] <= freq_y_max:
            image_y.append(Sy_fft_amplitude[i])
            #print(freqs[i])
            y.append(freqs_y[i]/a_y)
    image_x = []
    print(freqs_x)
    for i in range(len(freqs_x)):
        if freqs_x[i] >= freq0 and freqs_x[i] <= freq_max+500:
            x.append(freqs_x[i]*a_x+b_x)
            image_x.append(Sx_fft_amplitude[i])
    plt.plot(x, image_x)
    plt.show()

    plt.plot(y, image_y)
    plt.show()
    image = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            image[i, j] = image_x[i]*image_y[j]
    
    # Wizualizacja obrazu w dziedzinie przestrzennej
    plt.figure(figsize=(6, 6))
    plt.title("Obraz w przestrzeni rzeczywistej (metry)")
    #plt.imshow(S_fft_amplitude, origin='lower',cmap='inferno')#, extent = [freqs_x[0], freqs_x[len(freqs_x)-1], freqs_y[0], 4*freqs_y[len(freqs_y)-1]])
    plt.imshow(image, origin='lower',cmap='inferno', extent = [x[0], x[len(x)-1], y[0], y[len(y)-1]])
    plt.colorbar(label="Amplituda")
    plt.xlabel("Pozycja x (m)")
    plt.ylabel("Pozycja y (m)")
    plt.show()

def gradient_echo_sequence(voxels_set, TR, TE, T_RF, flip_angle, slice_z, Gzz, Nx, Ny, sampling_frequency): #aby dzialalo trzeba ustawic dB = 0
    """
    flip_angle - kąt obrotu wokół osi x w radianach
    slice_z - koordynata warstwy w osi z

    Pętla po punktach kx ky przestrzeni k-space
        #1. Impuls RF o kąt flip_angle przy jednoczesnym gradiencie w z - wybór warstwy
        #2. Gradienty defazujące
        #3. Precesja w obecności gradientów defazujących (przez jakiś czas nwm jaki)
        #4. Precesja z gradientami refazującymi i jednoczesny pomiar

        #Gy jest liniową funkcją ky, natomiast Gx skacze tylko z wartosci ujemnej na dodatnią i odwrotnie

        Zwraca k-space
    """
    T_read = Nx/sampling_frequency
    S = np.zeros((Nx, Ny), dtype=complex)
    frequency_RF = Voxel.gammaHz*(Voxel.B0 + Gzz*slice_z)
    Gx = 30/(Voxel.FOVx*Voxel.gammaRad*0.002)
    print(Gx)
    amplitude_RF = flip_angle/(2*np.pi*Voxel.gammaHz * T_RF)#Dobrane tak, aby obrót był o flip_angle
    T_dephase = T_read/2
    coil_area = np.pi * (Voxel.coil_radius**2)  # Powierzchnia cewki
    T_wait = TR - (T_RF + T_dephase + T_read)
    coil_area = np.pi * (Voxel.coil_radius**2)  # Powierzchnia cewki
    T_grad_y = TE - T_read - T_RF/2
    base_Gy = 10*np.pi/(Voxel.FOVy*Voxel.gammaHz*T_grad_y)

    """
    print("frequency_RF: ",frequency_RF)
    print("amplitude_RF: ",amplitude_RF)
    """
    print("T_grad_y: ",T_read)
    """
    print("T_read: ",T_read)
    print("Gx: ",Gx)
    print("base_Gy: ",base_Gy)
    print("T_RF: ",T_RF)
    print("T_dephase: ",T_dephase)
    print("T_wait: ",T_wait)
    print("TE: ",TE)
    print("TR: ",TR)
    """

    for j in range(Ny):#główna pętla powtarzająca sygnał RF co czas TR
        for voxel in voxels_set.values():#szybsza alternatywa dla T_wait
            voxel.set_magnetization_in_direction(0.0, 0.0)
        
        Gy = base_Gy * j/Ny
        n = 4000000
        gradient(voxels_set, 0., 0.,Gzz)#gradient w z do wyboru warstwy
        generate_RF_signal(voxels_set, frequency_RF, T_RF, amplitude_RF, n)
        
        #show_snapshot_3d(voxels_set, z_fixed=slice_z, y_fixed=0.004)
        
        Voxel.dt *= 10 #zmniejszam precyzję w mało kluczowych momentach
        gradient(voxels_set, 0., Gy, 0.)#Gradient fazujący (w osi y) 
        fast_long_precession(voxels_set, T_grad_y)
        
        gradient(voxels_set, -Gx, 0., 0.)#gradient defazujący w osi x
        fast_long_precession(voxels_set, T_dephase)

        gradient(voxels_set, Gx, 0., 0.)#Gradienty refazujące (w osi x) 
        Voxel.dt /= 10 #zwiększam precyzję w waznych momentach

        ### Zaczynamy pomiar!
        total_flux = np.zeros(int(T_read/Voxel.dt)+1)

        licznik = 0
        for t in np.arange(0., T_read, Voxel.dt):
            precession(voxels_set, t)
            for voxel in voxels_set.values():
                total_flux[licznik] += voxel.magnetization[0]*coil_area#sumuje składowe x magnetyzacji
            licznik += 1

        V = -np.gradient(total_flux, Voxel.dt)  # Użycie kroku czasowego jako odstępu
        V[len(V)-1] = 0 #ostatni element ustawiam sztucznie, bo nie ma następnego momentu czasu
        V[len(V)-2] = 0 #same
        #plt.plot(np.arange(len(V)), V)
        #plt.show()

        #próbkowanie V
        DT = T_read/Nx
        print("Progress: ", int(j/Ny*100), "%")
        for i in range(Nx): #teraz mając FID idziemy po kolejnych t_x i aktualizujemy je
            if int(i*DT/Voxel.dt) == len(V):
                break
            else:
                S[i, j] += V[int(i*DT/Voxel.dt)] 
        
        """
        Voxel.dt *= 10 #zmniejszam precyzję w mało kluczowych momentach
        fast_long_precession(voxels_set, T_wait)
        Voxel.dt /= 10 #zwiększam precyzję w waznych momentach
        """
    
    return S

def import_real_B0_field():
    """
    importuje plik .csv z rzeczywistym polem B0
    """
    import pandas as pd
    # Importowanie danych z pliku CSV
    data = pd.read_csv('B0_field.csv')
    B0_real = data["Absolute field (µT)"]
    Voxel.B0_real = B0_real[100:]#usuwam pierwsze 100 elementów, poniewaz roznia sie znaczaco or reszty
    Voxel.B0_real = Voxel.B0_real.reset_index(drop=True)
    #Voxel.B0_real = B0_real[:100]#pozostawiam jedynie 100 elementów, bo do reszty i tak nie dojdzie

#@njit
def dB_noise_function(t):
    """
    Funkcja generująca szum zewnętrznego pola B wzdłóż osi Z.
    
    Args:
        t (float): Czas w sekundach.
    
    Returns:
        dB: szum pola B w jednostkach Tesla.
    """

    """
    #szum sinusoidalny
    # Parametry szumu
    B0 = 30*1e-6
    noise_amplitude = 2*B0  # Amplituda szumu
    noise_frequency = 50.0  # Częstotliwość szumu

    # Generowanie szumu
    dB = noise_amplitude * np.sin(2 * np.pi * noise_frequency * t)

    return dB
    """
    """
    B0_dt = Voxel.B0_data_dt
    #print(Voxel.B0_real[0])
    B0 = 30*1e-6
    t1 = int(t/B0_dt)*B0_dt 
    t2 = (int(t/B0_dt)+1)*B0_dt
    B1 = Voxel.B0_real[int(t1/B0_dt)]
    B2 = Voxel.B0_real[int(t2/B0_dt)]
    return 1e-6*(np.abs(t-t1)*B2 + np.abs(t-t2)*B1)/np.abs(t1-t2) - B0
    """
    return 1e-6*Voxel.B0_real[int(t/Voxel.B0_data_dt)] - Voxel.B0
    #return 0.0

def gradient_echo_y_axis(voxels_set, T_RF, Ny, flip_angle, slice_z, Gzz, sampling_frequency):
    DT = 1/sampling_frequency #tyle będzie trwało zbieranie V(t)
    S = np.zeros(Ny, dtype=complex)
    frequency_RF = Voxel.gammaHz*(Voxel.B0 + Gzz*slice_z)
    amplitude_RF = flip_angle/(2*np.pi*Voxel.gammaHz * T_RF)#Dobrane tak, aby obrót był o flip_angle
    coil_area = np.pi * (Voxel.coil_radius**2)  # Powierzchnia cewki
    T_grad_y = 0.004
    #print("freq_max: ", freq_max)
    #print("T_read: ", T_read)
    n = 4000000
    base_Gy = 20*np.pi/(Voxel.FOVy*Voxel.gammaHz*T_grad_y)
    #print("base_Gy: ", base_Gy)

    #show_snapshot_3d(voxels_set, z_fixed=slice_z, y_fixed=0.004)Co o
    for j in range(Ny):
        print("Progress: ", int(j/Ny*100), "%")
        for voxel in voxels_set.values():#szybsza alternatywa dla T_wait
            voxel.set_magnetization_in_direction(0.0, 0.0)

        gradient(voxels_set, 0., 0.,Gzz)#gradient w z do wyboru warstwy
        generate_RF_signal(voxels_set, frequency_RF, T_RF, amplitude_RF, n)

        Gy = base_Gy*j/Ny
        gradient(voxels_set, 0., Gy, 0.)#Gradient fazujący (w osi y) 

        for t in np.arange(0., T_grad_y, Voxel.dt):
            precession(voxels_set, t+T_RF)
        
        ### Zaczynamy pomiar!
        gradient(voxels_set, 0., 0., 0.) #zeruje gradienty
        total_flux = np.zeros(int(DT/Voxel.dt)+1)

        licznik = 0
        for t in np.arange(0., DT, Voxel.dt):
            precession(voxels_set, t+T_grad_y + T_RF)
            for voxel in voxels_set.values():
                total_flux[licznik] += voxel.magnetization[0]*coil_area#sumuje składowe x magnetyzacji
            licznik += 1

        V = -np.gradient(total_flux, Voxel.dt)  # Użycie kroku czasowego jako odstępu
        V[len(V)-1] = 0 #ostatni element ustawiam sztucznie, bo nie ma następnego momentu czasu
        #plt.plot(np.arange(len(V)), V)
        #plt.show()
        
        S[j] += np.mean(V) #średnia wartość sygnału
    print("Czas trwania sekwencji: ", T_grad_y + T_RF + DT)
    return S

def gradient_echo_x_axis(voxels_set, T_RF, Nx, flip_angle, slice_z, Gzz, sampling_frequency):
    T_read = Nx/sampling_frequency
    S = np.zeros(Nx, dtype=complex)
    frequency_RF = Voxel.gammaHz*(Voxel.B0 + Gzz*slice_z)
    Gx = (sampling_frequency/2 - Voxel.B0*Voxel.gammaHz)/(Voxel.FOVx*Voxel.gammaHz*1.1)
    amplitude_RF = flip_angle/(2*np.pi*Voxel.gammaHz * T_RF)#Dobrane tak, aby obrót był o flip_angle
    T_dephase = T_read/2
    coil_area = np.pi * (Voxel.coil_radius**2)  # Powierzchnia cewki
    #print("freq_max: ", freq_max)
    print("Czas trwania sekwencji: ", T_read+ T_RF + T_dephase)
    n = 4000000
    gradient(voxels_set, 0., 0.,Gzz)#gradient w z do wyboru warstwy
    generate_RF_signal(voxels_set, frequency_RF, T_RF, amplitude_RF, n)
    
    #show_snapshot_3d(voxels_set, z_fixed=slice_z, y_fixed=0.004)Co o
    
    gradient(voxels_set, -Gx, 0., 0.)#gradient defazujący w osi x
    
    for t in np.arange(0., T_dephase, Voxel.dt):
        precession(voxels_set, t + T_RF)

    gradient(voxels_set, Gx, 0., 0.)#Gradienty refazujące (w osi x) 
    

    ### Zaczynamy pomiar!
    total_flux = np.zeros(int(T_read/Voxel.dt)+1)
    
    licznik = 0
    for t in np.arange(0., T_read, Voxel.dt):
        precession(voxels_set, t + T_dephase + T_RF)
        for voxel in voxels_set.values():
            total_flux[licznik] += voxel.magnetization[0]*coil_area#sumuje składowe x magnetyzacji
        licznik += 1
    
    V = -np.gradient(total_flux, Voxel.dt)  # Użycie kroku czasowego jako odstępu
    V[len(V)-1] = 0 #ostatni element ustawiam sztucznie, bo nie ma następnego momentu czasu
    V[len(V)-2] = 0 #same
    #plt.plot(np.arange(len(V)), V)
    #plt.show()

    #próbkowanie V
    DT = T_read/Nx
    print("DT: ", DT)
    for i in range(Nx): #teraz mając FID idziemy po kolejnych t_x i aktualizujemy je
        # Zaktualizuj wartość w przestrzeni k-space
        if int(i*DT/Voxel.dt) == len(V):
            break
        else:
            S[i] += V[int(i*DT/Voxel.dt)] 

    return S

def gaussian(x, a, x0, sigma):
    """Funkcja Gaussa."""
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def find_peaks(image, x,threshold=0.5, distance=10):
    from scipy.signal import find_peaks
    """Znajdź indeksy pików w tablicy."""
    peaks, _ = find_peaks(image, height=threshold, distance=distance)
    peaks_x = [x[i] for i in peaks]
    return peaks, peaks_x

def fit_gaussian_to_peaks(image,x, peaks):
    """Dopasuj funkcję Gaussa do każdego piku."""
    from scipy.optimize import curve_fit
    fit_params = []
    for peak in peaks:
        # Wybierz dane wokół piku
        window =  int(len(x)/15) # Rozmiar okna do dopasowania
        start = max(0, peak - window)
        end = min(len(image), peak + window)
        x_data = [x[i] for i in range(start, end)]
        y_data = image[start:end]

        # Początkowe zgadywane parametry: amplituda, środek, sigma
        initial_guess = [image[peak], x[peak], 0.01]

        try:
            params, _ = curve_fit(gaussian, x_data, y_data, p0=initial_guess)
            fit_params.append(params)
        except RuntimeError:
            print(f"Nie udało się dopasować Gaussa do piku w pozycji {peak}")
            fit_params.append(None)
    return fit_params

#funkcje generujące różne rodzaje fantomów
def create_single_voxel_fantom(voxels_set, x, y, z):
    """
    Dodaje pojedynczy voxel z domyślnymi wartościami do istniejącego zbioru voxelów.

    Args:
        voxels_set (dict): Istniejący zbiór voxelów.
        x (float): Pozycja x voxela.
        y (float): Pozycja y voxela.
        z (float): Pozycja z voxela.

    Returns:
        dict: Zaktualizowany zbiór voxelów zawierający nowy voxel.
    """

    # Parametry voxela
    proton_density = 1.0  # Gęstość protonów
    t1 = 200 * 1e-3  # Czas relaksacji T1
    t2 = 80 * 1e-3  # Czas relaksacji T2
    t2_star = 70 * 1e-3  # Czas relaksacji T2*

    # Tworzenie voxela
    voxel = Voxel(
        x=x, y=y, z=z,
        proton_density=proton_density,
        t1=t1,
        t2=t2,
        t2_star=t2_star
    )

    # Dodanie voxela do zbioru voxelów
    voxels_set[(x, y, z)] = voxel

    return voxels_set

def create_line(N):
    """
    Tworzy linię voxelów wzdłuż osi Z o długości N.

    Args:
        N (int): Liczba voxelów w linii.

    Returns:
        dict: Słownik zawierający voxel dla każdej pozycji w linii.
    """
    # Słownik voxelów dla linii
    line_voxels = {}
    z_positions = np.arange(1, 2*N+1, 2)
    # Generowanie linii wzdłuż osi Z
    for z in z_positions:
        # Dodajemy każdy voxel jako element linii
        line_voxels[(0, 0, z)] = Voxel(
            x=0,                # Stała pozycja w X
            y=0,                # Stała pozycja w Y
            z=z * Voxel.dz,     # Pozycja w Z uwzględniająca rozdzielczość dz
            proton_density=1.0, # Stała gęstość protonów dla linii
            t1=500,             # Przykładowe czasy relaksacji
            t2=100,
            t2_star=70
        )

    return line_voxels

def create_cube(N):
    """
    Tworzy sześcian voxelów o rozmiarze N x N x N.

    Args:
        N (int): Rozmiar siatki fantomu.

    Returns:
        dict: Słownik zawierający voxel dla każdej pozycji w sześcianie.
    """
    # Słownik voxelów dla sześcianu
    cube_voxels = {}

    # Generowanie sześcianu o wymiarach N x N x N
    for i in range(N):
        for j in range(N):
            for k in range(N):
                # Dodajemy każdy voxel jako element sześcianu
                x=i * Voxel.dx + Voxel.FOVx/2 - N*Voxel.dx/2  # Uwzględniamy rozdzielczość (dx, dy, dz)
                y=j * Voxel.dy+ Voxel.FOVy/2 - N*Voxel.dy/2
                z=k * Voxel.dz+ Voxel.FOVy/2 - N*Voxel.dz/2
                #print(x, y, z)
                cube_voxels[(x, y, z)] = Voxel(
                    x=x,  
                    y=y,
                    z=z,
                    proton_density=1, # Stała gęstość protonów dla sześcianu
                    t1=400*1e-3,  # Przykładowe czasy relaksacji (możesz dostosować)
                    t2=100*1e-3,
                    t2_star=80*1e-3
                )
    return cube_voxels

def create_apple(diameter_apple, diameter_seed):

    """
    Tworzy fantom jabłka z miąższem i pestkami jako zbiór voxelów.

    Args:
        diameter_apple (float): Średnica jabłka w jednostkach rzeczywistych (np. m).
        diameter_seed (float): Średnica pestek w jednostkach rzeczywistych (np. m).

    Returns:
        dict: Słownik voxelów, gdzie kluczami są współrzędne (x, y, z), 
              a wartościami są obiekty klasy Voxel.
    """
    dx = Voxel.dx  # Rozdzielczość z klasy Voxel w metrach
    N = int(diameter_apple / dx)  # Liczba voxelów wzdłuż jednej osi
    center_x, center_y, center_z = N // 2, N // 2, N // 2  # Środek jabłka

    radius_apple = diameter_apple / 2  # Promień jabłka w jednostkach rzeczywistych (m)
    radius_seed = diameter_seed / 2  # Promień pestek w jednostkach rzeczywistych (m)

    # Słownik voxelów, gdzie kluczami są współrzędne (x, y, z)
    apple_voxels = {}

    # Dodajemy voxele dla jabłka (miąższu)
    for x in range(N):
        for y in range(N):
            for z in range(N):
                real_x = x * dx
                real_y = y * dx
                real_z = z * dx
                distance_to_center = np.sqrt((real_x - center_x * dx)**2 + (real_y - center_y * dx)**2 + (real_z - center_z * dx)**2)
                if distance_to_center <= radius_apple:
                    # Dodajemy voxel miąższu
                    apple_voxels[(real_x, real_y, real_z)] = Voxel(
                        real_x, real_y, real_z, 
                        proton_density=0.9,  # Gęstość protonów w miąższu
                        t1=1000*1e-3,  # Przykładowe czasy relaksacji dla miąższu
                        t2=80*1e-3,
                        t2_star=50*1e-3
                    )

    # Dodajemy voxele dla pestek
    seeds_centers = [
        (center_x * dx + 0.005, center_y * dx, center_z * dx),
        (center_x * dx - 0.005, center_y * dx, center_z * dx),
        (center_x * dx, center_y * dx + 0.005, center_z * dx),
        (center_x * dx, center_y * dx - 0.005, center_z * dx)
    ]

    for seed_center in seeds_centers:
        seed_x, seed_y, seed_z = seed_center
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    real_x = x * dx
                    real_y = y * dx
                    real_z = z * dx
                    distance_to_seed = np.sqrt((real_x - seed_x)**2 + (real_y - seed_y)**2 + (real_z - seed_z)**2)
                    if distance_to_seed <= radius_seed:
                        # Dodajemy voxel pestki
                        apple_voxels[(real_x, real_y, real_z)] = Voxel(
                            real_x, real_y, real_z, 
                            proton_density=0.5,  # Gęstość protonów w pestkach
                            t1=800*1e-3,  # Przykładowe czasy relaksacji dla pestek
                            t2=60*1e-3,
                            t2_star=40*1e-3
                        )

    return apple_voxels

def create_XZ_fantom(a, b):#Tworzy fantom jako plaster a x b wycentrowany w płaszczyźnie XZ
    # Słownik voxelów 
    XZ_voxels = {}

    for i in range(a):
        for j in range(b):
            x=i * Voxel.dx + Voxel.FOVx/2 - a*Voxel.dx/2  # Uwzględniamy rozdzielczość (dx, dy, dz)
            y=0
            z=j * Voxel.dz+ Voxel.FOVy/2 - b*Voxel.dz/2
            #print(x, y, z)
            XZ_voxels[(x, y, z)] = Voxel(
                x=x,  
                y=y,
                z=z,
                proton_density=1, # Stała gęstość protonów dla sześcianu
                t1=400*1e-3,  # Przykładowe czasy relaksacji (możesz dostosować)
                t2=100*1e-3,
                t2_star=80*1e-3
            )
    return XZ_voxels

def generate_ball(diameter, Px, Py, Pz):
    """
    Generuje fantom 3D składający się z kuli
    
    Zwraca:
        fantom (dict): Mapa 3D voxelów { (x,y,z): Voxel }
    """
    # Ustalanie rozdzielczości voxelów na podstawie rozmiaru
    Voxel.dx = Voxel.FOVx / Px
    Voxel.dy = Voxel.FOVy / Py
    Voxel.dz = Voxel.FOVy / Pz  # Przyjmujemy FOVz = FOVy

    # Tworzenie siatki 3D
    x_vals = np.arange(0.0, Voxel.FOVx, Voxel.dx)
    y_vals = np.arange(0.0, Voxel.FOVy, Voxel.dy)
    z_vals = np.arange(0.0, Voxel.FOVy, Voxel.dz)

    # Definiowanie kul
    center = (x_vals[Px//2], y_vals[Py//2], z_vals[Pz//2])  # Środek kuli
    radius = diameter/2# Promien kuli

    # Wartości T1, T2, rho dla obu kul
    T1, T2, rho = 300, 100, 1.0

    # Inicjalizacja mapy voxelowej
    fantom = {}

    # Wypełnianie fantomu voxelami
    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                # Obliczanie odległości od środków kul
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

                if dist <= radius:
                    voxel = Voxel(x, y, z, rho, T1, T2, T2)  # T2* = T2 jako uproszczenie
                    fantom[(x, y, z)] = voxel
            
    return fantom

def generate_two_balls(size):
    """
    Generuje fantom 3D składający się z dwóch kul o różnych właściwościach T1, T2 i gęstości protonowej (rho).
    
    Zwraca:
        fantom (dict): Mapa 3D voxelów { (x,y,z): Voxel }
    """
    # Ustalanie rozdzielczości voxelów na podstawie rozmiaru
    Voxel.dx = Voxel.FOVx / size
    Voxel.dy = Voxel.FOVy / size
    Voxel.dz = Voxel.FOVy / size  # Przyjmujemy FOVz = FOVy

    # Tworzenie siatki 3D
    x_vals = np.arange(-Voxel.FOVx / 2, Voxel.FOVx / 2, Voxel.dx)
    y_vals = np.arange(-Voxel.FOVy / 2, Voxel.FOVy / 2, Voxel.dy)
    z_vals = np.arange(-Voxel.FOVy / 2, Voxel.FOVy / 2, Voxel.dz)

    # Definiowanie kul
    center1 = (0.01, 0.01, 0.0)  # Środek pierwszej kuli
    center2 = (-0.01, -0.01, 0.0)  # Środek drugiej kuli
    radius1, radius2 = 0.01, 0.015  # Promienie kul

    # Wartości T1, T2, rho dla obu kul
    T1_1, T2_1, rho_1 = 300, 100, 1.0
    T1_2, T2_2, rho_2 = 500, 200, 0.6

    # Inicjalizacja mapy voxelowej
    fantom = {}

    # Wypełnianie fantomu voxelami
    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                # Obliczanie odległości od środków kul
                dist1 = np.sqrt((x - center1[0])**2 + (y - center1[1])**2 + (z - center1[2])**2)
                dist2 = np.sqrt((x - center2[0])**2 + (y - center2[1])**2 + (z - center2[2])**2)

                if dist1 <= radius1:
                    voxel = Voxel(x, y, z, rho_1, T1_1, T2_1, T2_1)  # T2* = T2 jako uproszczenie
                    fantom[(x, y, z)] = voxel
                elif dist2 <= radius2:
                    voxel = Voxel(x, y, z, rho_2, T1_2, T2_2, T2_2)
                    fantom[(x, y, z)] = voxel

    return fantom

def show_snapshot_3d(voxels_set, z_fixed, y_fixed):
    """
    Pokazuje migawkę aktualnej magnetyzacji fantomu w płaszczyźnie XZ oraz XY.
    
    Args:
        voxels_set (dict): Słownik voxelów, gdzie kluczami są współrzędne (x, y, z),
                           a wartościami są obiekty klasy Voxel.
        z_fixed (float): Wartość z dla wybranej warstwy XY.
        y_fixed (float): Wartość y dla wybranej warstwy XZ.
    """
    # Ekstrakcja współrzędnych i magnetyzacji dla płaszczyzny XY (z_fixed)
    xy_voxels = {(x, y): voxel for (x, y, z), voxel in voxels_set.items() if np.isclose(z, z_fixed)}
    x_coords_xy = [voxel.x * 100 for voxel in xy_voxels.values()]  # Konwersja z metrów na cm
    y_coords_xy = [voxel.y * 100 for voxel in xy_voxels.values()]  # Konwersja z metrów na cm
    magnetizations_xy = [voxel.magnetization for voxel in xy_voxels.values()]

    # Ekstrakcja współrzędnych i magnetyzacji dla płaszczyzny XZ (y_fixed)
    xz_voxels = {(x, z): voxel for (x, y, z), voxel in voxels_set.items() if np.isclose(y, y_fixed)}
    x_coords_xz = [voxel.x * 100 for voxel in xz_voxels.values()]  # Konwersja z metrów na cm
    z_coords_xz = [voxel.z * 100 for voxel in xz_voxels.values()]  # Konwersja z metrów na cm
    magnetizations_xz = [voxel.magnetization for voxel in xz_voxels.values()]

    # Obliczanie maksymalnej długości wektora magnetyzacji
    max_magnetization_length = max(np.linalg.norm(m) for m in magnetizations_xz)

    # Tworzenie siatek dla płaszczyzn XY i XZ
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # Płaszczyzna XY
    axs[0].quiver(x_coords_xy, y_coords_xy, [m[0] for m in magnetizations_xy], [m[1] for m in magnetizations_xy], angles='xy', scale_units='xy', scale=1)
    axs[0].set_title(f'Magnetyzacja w płaszczyźnie XY (z = {z_fixed * 100:.2f} cm)')
    axs[0].set_xlabel('X (cm)')
    axs[0].set_ylabel('Y (cm)')
    axs[0].set_aspect('equal')

    # Płaszczyzna XZ
    axs[1].quiver(x_coords_xz, z_coords_xz, [m[0] for m in magnetizations_xz], [m[2] for m in magnetizations_xz], angles='xy', scale_units='xy', scale=1)
    axs[1].set_title(f'Magnetyzacja w płaszczyźnie XZ (y = {y_fixed * 100:.2f} cm)')
    axs[1].set_xlabel('X (cm)')
    axs[1].set_ylabel('Z (cm)')
    axs[1].set_aspect('equal')

    # Ustawienie limitów osi z marginesem
    x_min, x_max = min(x_coords_xz), max(x_coords_xz)
    z_min, z_max = min(z_coords_xz), max(z_coords_xz)
    margin = max_magnetization_length  # Dodanie marginesu na podstawie maksymalnej długości wektora magnetyzacji
    axs[1].set_xlim(x_min - margin, x_max + margin)
    axs[1].set_ylim(z_min - margin, z_max + margin)

    plt.tight_layout()
    plt.show()

def show_T2_rho_fantom_2d(voxels_set, z_fixed, atol=1e-1):
    """
    Wyświetla obraz jednej warstwy fantomu w przestrzeni XY.
    
    Kolorami zaznaczone są różne wartości T2_star.

    Args:
        voxels_set (dict): Słownik voxelów, gdzie kluczami są współrzędne (x, y, z),
                           a wartościami są obiekty klasy Voxel.
        z_fixed (float): Wartość z dla wybranej warstwy.
        atol (float): Absolutna tolerancja dla porównania wartości zmiennoprzecinkowych.
    """
    # Tworzenie podmapy dla ustalonego z_fixed
    slice_voxels = {(x, y): voxel for (x, y, z), voxel in voxels_set.items() if np.isclose(z, z_fixed, atol=atol)}

    if not slice_voxels:
        print(f"No voxels found at z = {z_fixed}")
        return

    # Ekstrakcja wartości T2_star i gęstości protonowej dla wybranej warstwy
    x_coords = [x for x, y in slice_voxels.keys()]
    y_coords = [y for x, y in slice_voxels.keys()]
    t2_star_values = [voxel.T2_star * 1000 for voxel in slice_voxels.values()]  # Konwersja z sekund na ms
    rho_values = [voxel.proton_density for voxel in slice_voxels.values()]

    # Tworzenie siatki wartości T2_star i gęstości protonowej
    x_unique = np.unique(x_coords)
    y_unique = np.unique(y_coords)
    t2_star_grid = np.zeros((len(y_unique), len(x_unique)))
    rho_grid = np.zeros((len(y_unique), len(x_unique)))

    for (x, y), t2_star, rho in zip(slice_voxels.keys(), t2_star_values, rho_values):
        x_idx = np.where(x_unique == x)[0][0]
        y_idx = np.where(y_unique == y)[0][0]
        t2_star_grid[y_idx, x_idx] = t2_star
        rho_grid[y_idx, x_idx] = rho

    # Tworzenie wizualizacji T2* i ρ obok siebie
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Wizualizacja T2*
    im1 = axs[0].imshow(t2_star_grid, cmap='viridis', origin='lower',
                        extent=[0, Voxel.FOVx * 100, 0, Voxel.FOVy * 100], vmin=0.0, vmax=np.max(t2_star_grid))
    axs[0].set_title(f'T2* (ms) at z = {z_fixed * 100:.2f} cm')
    axs[0].set_xlabel('X (cm)')
    axs[0].set_ylabel('Y (cm)')
    plt.colorbar(im1, ax=axs[0], label='T2* (ms)')

    # Wizualizacja ρ (gęstości protonowej)
    im2 = axs[1].imshow(rho_grid, cmap='plasma', origin='lower',
                        extent=[0, Voxel.FOVx * 100, 0, Voxel.FOVy * 100], vmin=0.0, vmax=1.0)
    axs[1].set_title(f'ρ (Proton Density) at z = {z_fixed * 100:.2f} cm')
    axs[1].set_xlabel('X (cm)')
    axs[1].set_ylabel('Y (cm)')
    plt.colorbar(im2, ax=axs[1], label='ρ')

    plt.tight_layout()
    plt.show()
    """
    Wyświetla obraz jednej warstwy fantomu w przestrzeni XY.
    
    Kolorami zaznaczone są różne wartości T2_star.

    Args:
        voxels_set (dict): Słownik voxelów, gdzie kluczami są współrzędne (x, y, z),
                           a wartościami są obiekty klasy Voxel.
        z_fixed (float): Wartość z dla wybranej warstwy.
        atol (float): Absolutna tolerancja dla porównania wartości zmiennoprzecinkowych.
    """
    # Tworzenie podmapy dla ustalonego z_fixed
    slice_voxels = {(x, y): voxel for (x, y, z), voxel in voxels_set.items() if np.isclose(z, z_fixed, atol=atol)}

    if not slice_voxels:
        print(f"No voxels found at z = {z_fixed}")
        return

    # Ekstrakcja wartości T2_star i gęstości protonowej dla wybranej warstwy
    x_coords = [x for x, y in slice_voxels.keys()]
    y_coords = [y for x, y in slice_voxels.keys()]
    t2_star_values = [voxel.T2_star * 1000 for voxel in slice_voxels.values()]  # Konwersja z sekund na ms
    rho_values = [voxel.proton_density for voxel in slice_voxels.values()]

    # Tworzenie siatki wartości T2_star i gęstości protonowej
    x_unique = np.unique(x_coords)
    y_unique = np.unique(y_coords)
    t2_star_grid = np.zeros((len(y_unique), len(x_unique)))
    rho_grid = np.zeros((len(y_unique), len(x_unique)))

    for (x, y), t2_star, rho in zip(slice_voxels.keys(), t2_star_values, rho_values):
        x_idx = np.where(x_unique == x)[0][0]
        y_idx = np.where(y_unique == y)[0][0]
        t2_star_grid[y_idx, x_idx] = t2_star
        rho_grid[y_idx, x_idx] = rho

    # Tworzenie wizualizacji T2* i ρ obok siebie
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Wizualizacja T2*
    im1 = axs[0].imshow(t2_star_grid, cmap='viridis', origin='lower',
                        extent=[0, Voxel.FOVx * 100, 0, Voxel.FOVy * 100])
    axs[0].set_title(f'T2* (ms) at z = {z_fixed * 100:.2f} cm')
    axs[0].set_xlabel('X (cm)')
    axs[0].set_ylabel('Y (cm)')
    plt.colorbar(im1, ax=axs[0], label='T2* (ms)')

    # Wizualizacja ρ (gęstości protonowej)
    im2 = axs[1].imshow(rho_grid, cmap='plasma', origin='lower',
                        extent=[0, Voxel.FOVx * 100, 0, Voxel.FOVy * 100], vmin=0.0, vmax=1.0)
    axs[1].set_title(f'ρ (Proton Density) at z = {z_fixed * 100:.2f} cm')
    axs[1].set_xlabel('X (cm)')
    axs[1].set_ylabel('Y (cm)')
    plt.colorbar(im2, ax=axs[1], label='ρ')

    plt.tight_layout()
    plt.show()

def show_rho_2D(fantom, z_fixed,Nx, Ny, atol=1e-1):
    # Tworzenie siatki wartości gęstości protonowej
    x_min, x_max = 0.0, Voxel.FOVx
    y_min, y_max = 0.0, Voxel.FOVy
    x_range = np.arange(x_min, x_max + x_max/Nx, x_max/Nx)
    y_range = np.arange(y_min, y_max + y_max/Ny, y_max/Ny)
    rho_grid = np.zeros((len(y_range), len(x_range)))

    for voxel in fantom.values():
        x_idx = int((voxel.x - x_min) / (x_max/Nx))
        y_idx = int((voxel.y - y_min) / (y_max/Ny))
        rho_grid[y_idx, x_idx] = voxel.proton_density

    # Tworzenie wizualizacji ρ
    plt.imshow(rho_grid, cmap='plasma', origin='lower',
            extent=[x_min, x_max, y_min, y_max], vmin=0.0, vmax=1.0)
    plt.title(f'ρ (Proton Density) at z = {z_fixed:.3f} m')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.colorbar(label='ρ')
    plt.show()

def parallel_GE_sequence(voxels_set, num_workers=8, *args):
    """
    Wykonuje funkcję gradient_echo_sequence równolegle na podzielonych podzbiorach voxelów.

    Parametry:
        voxels_set (list): Pełny zbiór voxelów.
        num_workers (int): Liczba procesów równoległych (domyślnie 8).
        args: Pozostałe argumenty funkcji gradient_echo_sequence.

    Zwraca:
        Macierz S będąca sumą wyników z wszystkich podzbiorów.
    """
    # Podział danych na 8 podzbiorów
    subfantoms = []
    for i in range(num_workers):
        subfantoms.append({})

    iterator = 0
    for voxel in voxels_set.values():
        subfantoms[iterator % num_workers][voxel.x, voxel.y, voxel.z] = voxel
        iterator += 1
    
    # Przygotowanie argumentów dla starmap (każdy podzbiór dostaje te same dodatkowe argumenty)
    tasks = [(subfantom, *args) for subfantom in subfantoms]

    # Uruchomienie obliczeń równoległych z paskiem postępu
    with mp.Pool(processes=num_workers) as pool:
        results = []
        with tqdm(total=num_workers + 1, desc="Processing") as pbar:
            for task in tasks:
                result = pool.apply_async(gradient_echo_sequence, args=task, callback=lambda _: pbar.update(1))
                results.append(result)
            pool.close()
            pool.join()
            S = sum(result.get() for result in results)
            pbar.update(1)  # Update for the final step of saving the matrix
    
    return S

def show_fantom(voxels_set):
    fantom = voxels_set
    import plotly.graph_objects as go
    # Przygotowanie danych do wizualizacji
    x_vox = [pos[0] for pos in fantom]
    y_vox = [pos[1] for pos in fantom]
    z_vox = [pos[2] for pos in fantom]
    
    # Składowe magnetyzacji dla każdego wokselu
    u = [voxel.magnetization[0] for voxel in fantom.values()]
    v = [voxel.magnetization[1] for voxel in fantom.values()]
    w = [voxel.magnetization[2] for voxel in fantom.values()]
    
    # Tworzenie wykresu 3D z wektorami magnetyzacji
    fig = go.Figure(
        data=[
            go.Cone(
                x=x_vox,
                y=y_vox,
                z=z_vox,
                u=u,
                v=v,
                w=w,
                colorscale='Viridis',
                sizemode="absolute",
                sizeref=0.01,
                showscale=True,
                colorbar=dict(title="Magnetization")
            )
        ]
    )
    fig.show()