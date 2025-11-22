import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.animation as animation
from collections import deque
import time

class SistemaControlMotor:
    def __init__(self):
        # Parámetros del sistema
        self.dt = 0.1  # intervalo de tiempo (scan rate)
        
        # Valores deseados
        self.velocidad_nominal = 80  # km/h
        self.rpm_nominal = 5500  # RPM a 80 km/h
        
        # Límites
        self.velocidad_min = 50
        self.velocidad_max = 100
        
        # Variables del sistema
        self.velocidad_actual = 70  # velocidad inicial
        self.rpm_ctrl = 4812  # RPM que salen del controlador (antes de perturbación)
        self.rpm_reales = 4812   # RPM reales (RPM_ctrl + perturbación)
        self.error = 0
        self.senal_control = 0
        self.perturbacion = 0
        self.retroalimentacion = 0
        
        # Parámetros del controlador PID
        self.Kp = 2.0
        self.Ki = 0.3
        self.Kd = 0.4
        self.integral = 0
        self.error_anterior = 0
        
        # Historial para gráficos
        max_points = 150
        self.historial_tiempo = deque(maxlen=max_points)
        self.historial_entrada = deque(maxlen=max_points)
        self.historial_salida = deque(maxlen=max_points)
        self.historial_error = deque(maxlen=max_points)
        self.historial_control = deque(maxlen=max_points)
        self.historial_perturbacion = deque(maxlen=max_points)
        self.historial_retro = deque(maxlen=max_points)
        self.historial_proporcional = deque(maxlen=max_points)
        self.historial_integral = deque(maxlen=max_points)
        self.historial_derivativo = deque(maxlen=max_points)
        self.historial_rpm_ctrl = deque(maxlen=max_points)
        self.historial_rpm_reales = deque(maxlen=max_points)
        
        # Inicializar historiales
        for i in range(5):
            t = i * self.dt
            self.historial_tiempo.append(t)
            self.historial_entrada.append(self.velocidad_nominal)
            self.historial_salida.append(self.velocidad_actual)
            self.historial_error.append(0)
            self.historial_control.append(0)
            self.historial_perturbacion.append(0)
            self.historial_retro.append(self.velocidad_actual)
            self.historial_proporcional.append(0)
            self.historial_integral.append(0)
            self.historial_derivativo.append(0)
            self.historial_rpm_ctrl.append(self.rpm_ctrl)
            self.historial_rpm_reales.append(self.rpm_reales)
    
    def calcular_control_pid(self, error):
        """Calcula la señal de control usando PID"""
        # Integral con limitación
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -15, 15)
        
        # Derivativo
        derivativo = (error - self.error_anterior) / self.dt if self.dt > 0 else 0
        
        P = self.Kp * error
        I = self.Ki * self.integral
        D = self.Kd * derivativo
        
        control = P + I + D
        control = np.clip(control, -20, 20)
        
        self.error_anterior = error
        return control, P, I, D
    
    def actualizar_sistema(self, perturbacion):
        """Actualiza el estado del sistema - CORRECCIÓN FINAL"""
        # Calcular error (basado en velocidad actual)
        self.error = self.velocidad_nominal - self.velocidad_actual
        
        # Calcular señal de control y aportes PID
        self.senal_control, P, I, D = self.calcular_control_pid(self.error)
        
        # CORRECCIÓN: Dinámica correcta de RPM
        # 1. El controlador genera RPM_ctrl
        rpm_cambio = self.senal_control * 25  # Ganancia del controlador
        self.rpm_ctrl += rpm_cambio * self.dt
        
        # Limitar RPM_ctrl
        self.rpm_ctrl = np.clip(self.rpm_ctrl, 4000, 7000)
        
        # 2. Los RPM REALES son RPM_ctrl + PERTURBACIÓN
        self.rpm_reales = self.rpm_ctrl + perturbacion
        
        # Limitar RPM reales
        self.rpm_reales = np.clip(self.rpm_reales, 4000, 7000)
        
        # 3. La VELOCIDAD se calcula desde los RPM REALES
        self.velocidad_actual = (self.rpm_reales / self.rpm_nominal) * self.velocidad_nominal
        
        # Limitar velocidad
        self.velocidad_actual = np.clip(self.velocidad_actual, 
                                       self.velocidad_min, 
                                       self.velocidad_max)
        
        # Retroalimentación igual a salida
        self.retroalimentacion = self.velocidad_actual
        
        # Actualizar historiales
        tiempo_actual = self.historial_tiempo[-1] + self.dt if self.historial_tiempo else 0
        
        self.historial_tiempo.append(tiempo_actual)
        self.historial_entrada.append(self.velocidad_nominal)
        self.historial_salida.append(self.velocidad_actual)
        self.historial_error.append(self.error)
        self.historial_control.append(self.senal_control)
        self.historial_perturbacion.append(perturbacion)
        self.historial_retro.append(self.retroalimentacion)
        self.historial_proporcional.append(P)
        self.historial_integral.append(I)
        self.historial_derivativo.append(D)
        self.historial_rpm_ctrl.append(self.rpm_ctrl)
        self.historial_rpm_reales.append(self.rpm_reales)
        
        # Log de valores (cada 5 scans = 0.5 segundos = 2 logs/segundo)
        if len(self.historial_tiempo) % 5 == 0:
            self.log_valores()
    
    def log_valores(self):
        """Muestra los valores actuales en consola"""
        print(f"T: {self.historial_tiempo[-1]:5.1f}s | "
              f"θi: {self.velocidad_nominal:4.0f} | "
              f"θ₀: {self.velocidad_actual:5.1f} | "
              f"e: {self.error:6.2f} | "
              f"θ₀c: {self.senal_control:6.2f} | "
              f"f: {self.retroalimentacion:5.1f} | "
              f"p: {self.historial_perturbacion[-1]:4.0f}RPM | "
              f"RPM_ctrl: {self.rpm_ctrl:4.0f} | "
              f"RPM_real: {self.rpm_reales:4.0f}")

def animar(i, sistema, slider_perturbacion, lines, axs, text_time):
    """Función de animación"""
    # Obtener valor actual del slider de perturbación
    perturbacion = slider_perturbacion.val
    
    # Actualizar sistema
    sistema.actualizar_sistema(perturbacion)
    
    # Actualizar líneas de los gráficos
    tiempos = sistema.historial_tiempo
    
    if len(tiempos) > 1:
        lines[0].set_data(tiempos, sistema.historial_entrada)
        lines[1].set_data(tiempos, sistema.historial_salida)
        lines[2].set_data(tiempos, sistema.historial_error)
        lines[3].set_data(tiempos, sistema.historial_control)
        lines[4].set_data(tiempos, sistema.historial_perturbacion)
        lines[5].set_data(tiempos, sistema.historial_retro)
        lines[6].set_data(tiempos, sistema.historial_proporcional)
        lines[7].set_data(tiempos, sistema.historial_integral)
        lines[8].set_data(tiempos, sistema.historial_derivativo)
        
        # Actualizar texto del tiempo
        text_time.set_text(f'Tiempo: {tiempos[-1]:.1f}s')
        
        # Ajustar límites del eje x
        x_min = max(0, tiempos[-1] - 15)
        x_max = tiempos[-1] + 1
        
        for ax in axs:
            ax.set_xlim(x_min, x_max)
    
    return lines + [text_time]

def main():
    # Configuración
    plt.rcParams['figure.figsize'] = [14, 10]
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Crear sistema de control
    sistema = SistemaControlMotor()
    
    # Configurar figura
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Sistema de Control de Velocidad - DINÁMICA RPM CORREGIDA', 
                 fontsize=14, fontweight='bold')
    
    # Definir posiciones de los subplots
    ax1 = plt.subplot2grid((3, 2), (0, 0))  # Entrada vs Salida
    ax2 = plt.subplot2grid((3, 2), (1, 0))  # Error
    ax3 = plt.subplot2grid((3, 2), (2, 0))  # Señal de Control
    ax4 = plt.subplot2grid((3, 2), (0, 1))  # Perturbaciones
    ax5 = plt.subplot2grid((3, 2), (1, 1))  # Retroalimentación
    ax6 = plt.subplot2grid((3, 2), (2, 1))  # Aporte PID
    
    axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    
    # Ajustar espaciado
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.12, top=0.90, hspace=0.4, wspace=0.3)
    
    # 1. Entrada nominal y salida del sistema
    line1, = ax1.plot([], [], 'r-', linewidth=2.5, label='Entrada nominal (θi)')
    line2, = ax1.plot([], [], 'b-', linewidth=2, label='Salida del sistema (θ₀)')
    ax1.set_title('Entrada vs Salida', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Velocidad (km/h)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(65, 85)
    ax1.set_xlim(0, 15)
    ax1.axhline(y=80, color='r', linestyle='--', alpha=0.5)
    
    # 2. Error del Sistema
    line3, = ax2.plot([], [], 'g-', linewidth=2, label='Error (e)')
    ax2.set_title('Error del Sistema', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Error (km/h)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(-5, 15)
    ax2.set_xlim(0, 15)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 3. Señal de Control
    line4, = ax3.plot([], [], 'm-', linewidth=2, label='Salida controlador (θ₀c)')
    ax3.set_title('Señal de Control', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Señal de control')
    ax3.set_xlabel('Tiempo (s)')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_ylim(-5, 15)
    ax3.set_xlim(0, 15)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 4. Perturbaciones
    line5, = ax4.plot([], [], 'c-', linewidth=2, label='Perturbación (p)')
    ax4.set_title('Perturbaciones', fontweight='bold', fontsize=11)
    ax4.set_ylabel('RPM')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.set_ylim(-350, 350)
    ax4.set_xlim(0, 15)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 5. Retroalimentación
    line6, = ax5.plot([], [], 'y-', linewidth=2, label='Retroalimentación (f)')
    ax5.set_title('Retroalimentación (f = θ₀)', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Retroalimentación')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.set_ylim(65, 85)
    ax5.set_xlim(0, 15)
    
    # 6. Aporte de cada controlador PID
    line7, = ax6.plot([], [], 'r-', linewidth=1.5, label='Proporcional (P)')
    line8, = ax6.plot([], [], 'g-', linewidth=1.5, label='Integral (I)') 
    line9, = ax6.plot([], [], 'b-', linewidth=1.5, label='Derivativo (D)')
    ax6.set_title('Aporte de Controladores PID', fontweight='bold', fontsize=11)
    ax6.set_ylabel('Aporte')
    ax6.set_xlabel('Tiempo (s)')
    ax6.legend(loc='upper right', fontsize=9)
    ax6.set_ylim(-5, 10)
    ax6.set_xlim(0, 15)
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    lines = [line1, line2, line3, line4, line5, line6, line7, line8, line9]
    
    # Texto para mostrar tiempo actual
    text_time = ax1.text(0.02, 0.98, 'Tiempo: 0.0s', transform=ax1.transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Crear slider para perturbación
    ax_slider = plt.axes([0.25, 0.02, 0.5, 0.03])
    slider_perturbacion = Slider(
        ax_slider, 'Perturbación (RPM)', -300, 300, 
        valinit=0, valstep=10
    )
    
    # Configurar animación
    ani = animation.FuncAnimation(
        fig, animar, fargs=(sistema, slider_perturbacion, lines, axs, text_time),
        interval=100, blit=True, cache_frame_data=False
    )
    
    print("=== SISTEMA CON DINÁMICA RPM CORREGIDA ===")
    print("CORRECCIÓN FINAL:")
    print("1. RPM_ctrl: Salen del controlador (antes de perturbación)")
    print("2. RPM_real: RPM_ctrl + Perturbación (después de perturbación)") 
    print("3. Velocidad: Se calcula desde RPM_real")
    print("")
    print("COMPORTAMIENTO ESPERADO:")
    print("Sin perturbación: RPM_ctrl = RPM_real = 5500, θ₀ = 80")
    print("Con +200 RPM: RPM_real = RPM_ctrl + 200, θ₀ aumenta")
    print("Controlador reduce RPM_ctrl para compensar")
    print("")
    print("=" * 95)
    print("T (s) | θi | θ₀  | Error | θ₀c   | f    | p (RPM) | RPM_ctrl | RPM_real")
    print("-" * 95)
    
    plt.show()

if __name__ == "__main__":
    main()