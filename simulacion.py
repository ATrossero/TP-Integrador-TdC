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
        
        # Límites de velocidad
        self.velocidad_min = 50
        self.velocidad_max = 100
        
        # Parámetros VSS (Vehicle Speed Sensor)
        self.vss_vel_min = 50  # km/h para 0V
        self.vss_vel_max = 100  # km/h para 5V
        self.vss_volt_min = 0  # V
        self.vss_volt_max = 5  # V
        
        # Punto de "No Modificar" para la señal de control en V
        self.senal_ctrl_offset = 2.5 # V
        
        # Rango objetivo RELATIVO a la velocidad nominal (-2, +1)
        self.rango_offset_min = -2
        self.rango_offset_max = +1
        
        # Variables del sistema
        self.velocidad_actual = 70  # velocidad inicial
        self.rpm_ctrl = 4812
        self.rpm_reales = 4812
        self.error_kmh = 0
        # self.error_volts ahora es la ENTRADA al controlador
        self.error_volts = 0  
        self.senal_control_interna = 0 # Valor de control sin conversión a 0-5V
        self.senal_control_volts = 0 # Nuevo: Señal de control en 0-5V
        self.perturbacion = 0
        self.retroalimentacion = 0
        self.tiempo_transcurrido = 0
        
        # Parámetros del controlador - PROPORCIONAL DOMINANTE
        self.Kp = 4.0    # Proporcional alto para respuesta rápida
        self.Ki = 0.2   # Integral mínima
        self.Kd = 0.8    # Derivativo para suavizar
        self.integral = 0
        
        # El error anterior debe ser en km/h para el término derivativo
        self.error_anterior_kmh = 0 
        
        # Control predictivo
        self.velocidad_anterior = 70
        self.tendencia = 0
        
        # Factor de atenuación de perturbación
        self.factor_atenuacion_perturbacion = 0.3  # La perturbación afecta 70% menos
        
        # Historial para gráficos
        max_points = 150
        self.historial_tiempo = deque(maxlen=max_points)
        self.historial_entrada = deque(maxlen=max_points)
        self.historial_salida = deque(maxlen=max_points)
        self.historial_error_kmh = deque(maxlen=max_points)  # Error en km/h (para log)
        self.historial_error_volts = deque(maxlen=max_points)  # Error en volts (ENTRADA)
        # Historial para la nueva señal de control en Volts
        self.historial_control_volts = deque(maxlen=max_points) 
        self.historial_control_interna = deque(maxlen=max_points) # Señal de control antes de 0-5V
        self.historial_perturbacion = deque(maxlen=max_points)
        self.historial_retro = deque(maxlen=max_points)
        self.historial_proporcional = deque(maxlen=max_points)
        self.historial_integral = deque(maxlen=max_points)
        self.historial_derivativo = deque(maxlen=max_points)
        self.historial_rpm_ctrl = deque(maxlen=max_points)
        self.historial_rpm_reales = deque(maxlen=max_points)
        self.historial_en_rango = deque(maxlen=max_points)
        
        # Inicializar historiales
        for i in range(5):
            t = i * self.dt
            self.historial_tiempo.append(t)
            self.historial_entrada.append(self.velocidad_nominal)
            self.historial_salida.append(self.velocidad_actual)
            self.historial_error_kmh.append(0)
            self.historial_error_volts.append(0)
            self.historial_control_interna.append(0)
            self.historial_control_volts.append(self.senal_ctrl_offset) # Inicializar en 2.5V
            self.historial_perturbacion.append(0)
            self.historial_retro.append(self.velocidad_actual)
            self.historial_proporcional.append(0)
            self.historial_integral.append(0)
            self.historial_derivativo.append(0)
            self.historial_rpm_ctrl.append(self.rpm_ctrl)
            self.historial_rpm_reales.append(self.rpm_reales)
            self.historial_en_rango.append(self.esta_en_rango_objetivo())
    
    def velocidad_a_voltaje(self, velocidad):
        """Convierte velocidad (km/h) a voltaje VSS (0-5V)"""
        if self.vss_vel_max == self.vss_vel_min:
            return 0
        return np.clip(
            ((velocidad - self.vss_vel_min) / 
             (self.vss_vel_max - self.vss_vel_min)) * (self.vss_volt_max - self.vss_volt_min) + self.vss_volt_min,
            self.vss_volt_min, self.vss_volt_max
        )
    
    def voltaje_a_velocidad(self, voltaje):
        """Convierte voltaje VSS (0-5V) a velocidad (km/h)"""
        if self.vss_volt_max == self.vss_volt_min:
            return self.vss_vel_min
        return ((voltaje - self.vss_volt_min) / 
                (self.vss_volt_max - self.vss_volt_min)) * (self.vss_vel_max - self.vss_vel_min) + self.vss_vel_min
    
    def error_volts_a_kmh(self, error_volts):
        """Convierte el error en Volts (diferencia) a error en km/h (diferencia)"""
        # Relación de escala: (km/h_rango) / (V_rango)
        escala = (self.vss_vel_max - self.vss_vel_min) / (self.vss_volt_max - self.vss_volt_min)
        return error_volts * escala

    def senal_control_a_voltaje(self, senal_control_interna):
        """
        Adapta la señal de control (-X a +Y) a un rango de 0-5V.
        Señal de control centrada en 2.5V:
        0 (no modificar) -> 2.5V
        Negativo (disminuir) -> < 2.5V
        Positivo (aumentar) -> > 2.5V
        
        Asumimos que el rango operativo interno de la señal de control es de aproximadamente [-25, 25]
        como se limita al final de calcular_control_proporcional_inteligente.
        
        Escalaremos la señal_control_interna a un rango de -2.5V a +2.5V
        para luego sumarle 2.5V.
        """
        # Valor máximo absoluto de la señal de control interna para escalado
        MAX_CTRL_INTERNA = 25.0 
        
        # Escalar la señal_control_interna a un rango de -2.5V a +2.5V
        # Escala: (max_ctrl_volts) / (max_ctrl_interna) = 2.5 / 25.0 = 0.1
        senal_escalada = np.clip(senal_control_interna, -MAX_CTRL_INTERNA, MAX_CTRL_INTERNA) * (self.senal_ctrl_offset / MAX_CTRL_INTERNA)
        
        # Desplazar a 2.5V para obtener el voltaje final (0V a 5V)
        voltaje_final = senal_escalada + self.senal_ctrl_offset
        
        # Asegurar límites de 0V a 5V
        return np.clip(voltaje_final, self.vss_volt_min, self.vss_volt_max)

    def get_rango_objetivo(self):
        """Calcula el rango objetivo basado en la velocidad nominal actual"""
        return (self.velocidad_nominal + self.rango_offset_min, 
                self.velocidad_nominal + self.rango_offset_max)
    
    # MODIFICACIÓN: Ahora recibe error_volts como entrada
    def calcular_control_proporcional_inteligente(self, error_volts): 
        """Calcula control con PID que usa el error en km/h (convertido de error_volts)"""
        
        # CONVERSIÓN CRUCIAL: Convertir el error de volts a km/h para la lógica PID
        error_kmh = self.error_volts_a_kmh(error_volts)
        
        rango_min, rango_max = self.get_rango_objetivo()
        
        # 1. CALCULAR TENDENCIA para predecir sobrepasos
        if len(self.historial_salida) >= 2:
            self.tendencia = (self.velocidad_actual - self.velocidad_anterior) / self.dt
        self.velocidad_anterior = self.velocidad_actual
        
        # 2. PROPORCIONAL ADAPTATIVO - diferente según la situación
        if not self.esta_en_rango_objetivo():
            # FUERA DEL RANGO - Proporcional FUERTE para regresar rápido
            if self.velocidad_actual > rango_max:
                # Por encima del rango - acción negativa fuerte
                error_efectivo = self.velocidad_nominal - self.velocidad_actual # error_kmh
                Kp_efectivo = self.Kp * 1.5  # 50% más fuerte
            else:
                # Por debajo del rango - acción positiva fuerte
                error_efectivo = self.velocidad_nominal - self.velocidad_actual # error_kmh
                Kp_efectivo = self.Kp * 1.3  # 30% más fuerte
        else:
            # DENTRO DEL RANGO - Proporcional normal
            error_efectivo = error_kmh
            Kp_efectivo = self.Kp
            
            # Si está cerca de los límites, reducir ganancia para evitar sobrepasos
            margen = 0.3
            if self.velocidad_actual > (self.velocidad_nominal + margen) or self.velocidad_actual < (self.velocidad_nominal - margen):
                Kp_efectivo *= 0.7  # 30% menos cerca de límites
        
        P = Kp_efectivo * error_efectivo
        
        # 3. INTEGRAL MUY LIMITADA - solo para corrección fina
        if self.esta_en_rango_objetivo() and abs(error_kmh) < 1.0:
            self.integral += error_kmh * self.dt * 0.02  # Muy lenta
            self.integral = np.clip(self.integral, -2, 2)  # Muy limitada
        else:
            self.integral *= 0.9  # Decaimiento rápido
        
        I = self.Ki * self.integral
        
        # 4. DERIVATIVO para suavizar
        # Usa el error_kmh actual y el anterior (en km/h)
        derivativo = (error_kmh - self.error_anterior_kmh) / self.dt if self.dt > 0 else 0
        D = self.Kd * derivativo
        
        control = P + I + D
        
        # 5. ACCIÓN CORRECTIVA INSTANTÁNEA si se detecta sobrepaso inminente
        accion_correctiva = 0
        velocidad_proyectada = self.velocidad_actual + self.tendencia * self.dt
        
        if velocidad_proyectada > rango_max + 0.3:
            # Se proyecta sobrepaso superior - acción correctiva fuerte
            exceso = velocidad_proyectada - (rango_max + 0.2)
            accion_correctiva = -exceso * 10.0
            # print(f"🚨 Prevención sobrepaso superior: {velocidad_proyectada:.1f} km/h")
        
        elif velocidad_proyectada < rango_min - 0.3:
            # Se proyecta sobrepaso inferior - acción correctiva moderada
            deficit = (rango_min - 0.2) - velocidad_proyectada
            accion_correctiva = deficit * 6.0
            # print(f"⚠️ Prevención sobrepaso inferior: {velocidad_proyectada:.1f} km/h")
        
        control += accion_correctiva
        
        # 6. LIMITACIÓN INTELIGENTE de la señal de control
        if self.velocidad_actual > rango_max + 0.2:
            # Muy cerca del límite superior - limitar acciones positivas
            control = np.clip(control, -30, 5)
        elif self.velocidad_actual < rango_min - 0.2:
            # Muy cerca del límite inferior - limitar acciones negativas
            control = np.clip(control, -5, 20)
        else:
            # Zona normal
            control = np.clip(control, -25, 25)
        
        # Guardar el error_kmh para el cálculo derivativo del siguiente paso
        self.error_anterior_kmh = error_kmh
        
        return control, P, I, D, accion_correctiva
    
    def aplicar_perturbacion_atenuada(self, perturbacion):
        """Aplica la perturbación con atenuación para que afecte menos a la salida"""
        # Limitar perturbación a rango -100 a +200 RPM
        perturbacion_limited = np.clip(perturbacion, -100, 200)
        
        # Aplicar factor de atenuación (la perturbación afecta menos)
        perturbacion_atenuada = perturbacion_limited * self.factor_atenuacion_perturbacion
        
        return perturbacion_atenuada
    
    def esta_en_rango_objetivo(self):
        """Verifica si la velocidad está en el rango objetivo (nominal-2, nominal+1) km/h"""
        rango_min, rango_max = self.get_rango_objetivo()
        return rango_min <= self.velocidad_actual <= rango_max
    
    def actualizar_sistema(self, perturbacion, velocidad_nominal=None):
        """Actualiza el estado del sistema"""
        # Actualizar velocidad nominal si se proporciona
        if velocidad_nominal is not None:
            self.velocidad_nominal = velocidad_nominal
        
        # Incrementar tiempo
        self.tiempo_transcurrido += self.dt
        
        # 1. Calcular error de ENTRADA en Volts (para el controlador)
        voltaje_nominal = self.velocidad_a_voltaje(self.velocidad_nominal)
        voltaje_actual = self.velocidad_a_voltaje(self.velocidad_actual)
        self.error_volts = voltaje_nominal - voltaje_actual # NUEVO ERROR EN VOLTS
        
        # 2. Calcular señal de control. Usa error_volts, pero convierte a km/h internamente.
        self.senal_control_interna, P, I, D, accion_correctiva = \
            self.calcular_control_proporcional_inteligente(self.error_volts)
            
        # 3. Conversión de la señal de control interna a 0-5V
        self.senal_control_volts = self.senal_control_a_voltaje(self.senal_control_interna)
        
        # Cálculo del error en km/h (SOLO PARA LOG/HISTORIAL)
        self.error_kmh = self.velocidad_nominal - self.velocidad_actual
        
        # 4. Dinámica de RPM (ahora usa la señal de control CONVERTIDA A VOLTS)
        # La señal de control en volts está centrada en 2.5V. 
        # La diferencia con 2.5V determina el cambio:
        delta_voltaje_ctrl = self.senal_control_volts - self.senal_ctrl_offset
        
        ganancia_base = 18
        
        # Aumentar ganancia si está fuera del rango para respuesta más rápida
        if not self.esta_en_rango_objetivo():
            ganancia_base *= 1.3  # 30% más rápido fuera del rango
            
        # El cambio de RPM ahora es proporcional al delta_voltaje_ctrl
        # Escala: 2.5V representa el límite de MAX_CTRL_INTERNA (25.0)
        # Cambio de RPM = (delta_voltaje_ctrl * (MAX_CTRL_INTERNA / 2.5V)) * ganancia_base
        # Como MAX_CTRL_INTERNA/2.5 = 10, es: delta_voltaje_ctrl * 10 * ganancia_base
        
        # Usamos self.senal_control_interna, que ya contiene el valor de control 
        # sin escalar a 0-5V, ya que es más directo. 
        # NOTA: EL CONTROL SE BASA EN LA SEÑAL INTERNA.
        # Si se desea que se base en la señal de voltaje final, se haría:
        # rpm_cambio = (delta_voltaje_ctrl * (25.0 / 2.5)) * ganancia_base
        
        # Mantenemos la lógica de la señal de control interna para la dinámica, ya que es más natural.
        rpm_cambio = self.senal_control_interna * ganancia_base
        
        self.rpm_ctrl += rpm_cambio * self.dt
        
        # Limitar RPM_ctrl
        self.rpm_ctrl = np.clip(self.rpm_ctrl, 4700, 6300)
        
        # APLICAR PERTURBACIÓN ATENUADA
        perturbacion_atenuada = self.aplicar_perturbacion_atenuada(perturbacion)
        
        # RPM REALES son RPM_ctrl + PERTURBACIÓN ATENUADA
        self.rpm_reales = self.rpm_ctrl + perturbacion_atenuada
        
        # Limitar RPM reales
        self.rpm_reales = np.clip(self.rpm_reales, 4700, 6300)
        
        # La VELOCIDAD se calcula desde los RPM REALES
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
        self.historial_error_kmh.append(self.error_kmh)
        self.historial_error_volts.append(self.error_volts)
        self.historial_control_interna.append(self.senal_control_interna)
        self.historial_control_volts.append(self.senal_control_volts) # Guardar la nueva señal en Volts
        self.historial_perturbacion.append(perturbacion)  # Guardamos la perturbación original para mostrar
        self.historial_retro.append(self.retroalimentacion)
        self.historial_proporcional.append(P)
        self.historial_integral.append(I)
        self.historial_derivativo.append(D)
        self.historial_rpm_ctrl.append(self.rpm_ctrl)
        self.historial_rpm_reales.append(self.rpm_reales)
        self.historial_en_rango.append(self.esta_en_rango_objetivo())
        
        # Log de valores
        if len(self.historial_tiempo) % 5 == 0:
            self.log_valores(accion_correctiva, perturbacion_atenuada)
    
    def log_valores(self, accion_correctiva, perturbacion_atenuada):
        """Muestra los valores actuales en consola"""
        en_rango = "✓" if self.esta_en_rango_objetivo() else "✗"
        correccion = "🛡️" if abs(accion_correctiva) > 2.0 else "  "
        rango_min, rango_max = self.get_rango_objetivo()
        
        # Mostrar tanto la perturbación original como la atenuada
        perturbacion_original = self.historial_perturbacion[-1]
        
        # Calcular voltajes para mostrar
        voltaje_nominal = self.velocidad_a_voltaje(self.velocidad_nominal)
        voltaje_actual = self.velocidad_a_voltaje(self.velocidad_actual)
        
        print(f"T: {self.historial_tiempo[-1]:5.1f}s | "
              f"θi: {self.velocidad_nominal:4.0f}km/h | "
              f"θ₀: {self.velocidad_actual:5.1f}km/h | "
              f"eV: {self.error_volts:5.3f}V | " # Error en Volts
              f"θ₀c: {self.senal_control_volts:5.3f}V | " # Señal de Control en Volts
              f"p: {perturbacion_original:4.0f}RPM | "
              f"Rango: [{rango_min:.0f}-{rango_max:.0f}] {en_rango} {correccion}")

def animar(i, sistema, slider_perturbacion, slider_velocidad, lines, axs, text_time, text_estado, text_rango, text_voltaje):
    """Función de animación"""
    # Obtener valores actuales de los sliders
    perturbacion = slider_perturbacion.val
    velocidad_nominal = slider_velocidad.val
    
    # Actualizar sistema
    sistema.actualizar_sistema(perturbacion, velocidad_nominal)
    
    # Actualizar líneas de los gráficos
    tiempos = sistema.historial_tiempo
    
    if len(tiempos) > 1:
        lines[0].set_data(tiempos, sistema.historial_entrada)
        lines[1].set_data(tiempos, sistema.historial_salida)
        lines[2].set_data(tiempos, sistema.historial_error_volts)  # Error en Volts
        lines[3].set_data(tiempos, sistema.historial_control_volts) # Señal de Control en Volts
        lines[4].set_data(tiempos, sistema.historial_perturbacion)
        lines[5].set_data(tiempos, sistema.historial_retro)
        lines[6].set_data(tiempos, sistema.historial_proporcional)
        lines[7].set_data(tiempos, sistema.historial_integral)
        lines[8].set_data(tiempos, sistema.historial_derivativo)
        
        # Actualizar textos
        text_time.set_text(f'Tiempo: {tiempos[-1]:.1f}s')
        
        # Calcular rango objetivo actual
        rango_min, rango_max = sistema.get_rango_objetivo()
        
        # Calcular voltajes actuales
        voltaje_nominal = sistema.velocidad_a_voltaje(sistema.velocidad_nominal)
        voltaje_actual = sistema.velocidad_a_voltaje(sistema.velocidad_actual)
        
        # Indicador de estado
        if sistema.esta_en_rango_objetivo():
            estado_texto = f"EN RANGO {rango_min:.0f}-{rango_max:.0f} km/h ✅"
            color_fondo = 'lightgreen'
            color_borde = 'green'
        else:
            estado_texto = f"FUERA DE RANGO ⚠️"
            color_fondo = 'lightyellow'
            color_borde = 'orange'
            
        text_estado.set_text(estado_texto)
        text_estado.set_bbox(dict(boxstyle='round', facecolor=color_fondo, 
                                 edgecolor=color_borde, alpha=0.8))
        
        # Texto del rango objetivo
        text_rango.set_text(f'Objetivo: {sistema.velocidad_nominal:.0f} km/h | Rango: {rango_min:.0f}-{rango_max:.0f} km/h')
        
        # Texto de voltajes VSS
        # Ahora mostramos la señal de control en Volts
        text_voltaje.set_text(f'VSS: {voltaje_nominal:.1f}V (obj) → {voltaje_actual:.1f}V (act) | Error: {sistema.error_volts:.3f}V | Ctrl: {sistema.senal_control_volts:.3f}V')
        
        # Ajustar límites del eje x
        x_min = max(0, tiempos[-1] - 15)
        x_max = tiempos[-1] + 1
        
        # Ajustar límites del eje y en gráfico de entrada/salida según la velocidad nominal
        ax1_ymin = max(sistema.velocidad_min, sistema.velocidad_nominal - 10)
        ax1_ymax = min(sistema.velocidad_max, sistema.velocidad_nominal + 10)
        
        axs[0].set_ylim(ax1_ymin, ax1_ymax)
        
        for ax in axs:
            ax.set_xlim(x_min, x_max)
    
    return lines + [text_time, text_estado, text_rango, text_voltaje]

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
    fig.suptitle('Sistema de Control de Velocidad - Pit Lane Assistance (VSS: 0-5V)', 
                 fontsize=14, fontweight='bold')
    
    # Definir posiciones de los subplots
    ax1 = plt.subplot2grid((3, 2), (0, 0))  # Entrada vs Salida
    ax2 = plt.subplot2grid((3, 2), (1, 0))  # Error (VOLTS)
    ax3 = plt.subplot2grid((3, 2), (2, 0))  # Señal de Control (VOLTS)
    ax4 = plt.subplot2grid((3, 2), (0, 1))  # Perturbaciones
    ax5 = plt.subplot2grid((3, 2), (1, 1))  # Retroalimentación
    ax6 = plt.subplot2grid((3, 2), (2, 1))  # Aporte Controladores
    
    axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    
    # Ajustar espaciado
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.90, hspace=0.4, wspace=0.3)
    
    # 1. Entrada nominal y salida del sistema
    line1, = ax1.plot([], [], 'r-', linewidth=2.5, label='Entrada nominal (θi)')
    line2, = ax1.plot([], [], 'b-', linewidth=2, label='Salida del sistema (θ₀)')
    ax1.set_title('Entrada vs Salida - Control de Velocidad Nominal', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Velocidad (km/h)')
    
    # Líneas de referencia se actualizarán en la animación
    line_obj, = ax1.plot([], [], 'r--', alpha=0.5, label='Objetivo')
    rango_patch = ax1.axhspan(78, 81, alpha=0.2, color='green', label='Rango objetivo')
    
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_ylim(70, 90)
    ax1.set_xlim(0, 15)
    
    # 2. Error del Sistema (AHORA EN VOLTS)
    line3, = ax2.plot([], [], 'g-', linewidth=2, label='v_error (V)')
    ax2.set_title('Error (Volts)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Error (Volts)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(-0.5, 0.5)  # Rango típico de error en volts
    ax2.set_xlim(0, 15)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 3. Señal de Control (AHORA EN VOLTS)
    line4, = ax3.plot([], [], 'm-', linewidth=2, label='señal_control (V)')
    ax3.set_title('Señal de Control Adaptada (0-5V, 2.5V=No Modificar)', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Señal (Volts)')
    ax3.set_xlabel('Tiempo (s)')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_ylim(0, 5)
    ax3.set_xlim(0, 15)
    ax3.axhline(y=sistema.senal_ctrl_offset, color='k', linestyle='--', alpha=0.5, label='2.5V (Neutral)')
    
    # 4. Perturbaciones
    line5, = ax4.plot([], [], 'c-', linewidth=2, label='Perturbación (p)')
    ax4.set_title('Perturbaciones (-100 a +200 RPM)', fontweight='bold', fontsize=11)
    ax4.set_ylabel('RPM')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.set_ylim(-120, 220)
    ax4.set_xlim(0, 15)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.axhline(y=-100, color='r', linestyle='--', alpha=0.7, label='Límite inferior')
    ax4.axhline(y=200, color='r', linestyle='--', alpha=0.7, label='Límite superior')
    
    # 5. Retroalimentación
    line6, = ax5.plot([], [], 'y-', linewidth=2, label='Retroalimentación (f)')
    ax5.set_title('Retroalimentación (f = θ₀)', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Retroalimentación (km/h)')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.set_ylim(70, 90)
    ax5.set_xlim(0, 15)
    
    # 6. Aporte de cada controlador
    line7, = ax6.plot([], [], 'r-', linewidth=1.5, label='Proporcional (P)')
    line8, = ax6.plot([], [], 'g-', linewidth=1.5, label='Integral (I)')  
    line9, = ax6.plot([], [], 'b-', linewidth=1.5, label='Derivativo (D)')
    ax6.set_title('Aporte de Controladores (Señal Interna)', fontweight='bold', fontsize=11)
    ax6.set_ylabel('Aporte (Magnitud)')
    ax6.set_xlabel('Tiempo (s)')
    ax6.legend(loc='upper right', fontsize=9)
    ax6.set_ylim(-10, 10)
    ax6.set_xlim(0, 15)
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    lines = [line1, line2, line3, line4, line5, line6, line7, line8, line9]
    
    # Texto para mostrar tiempo actual
    text_time = ax1.text(0.02, 0.95, 'Tiempo: 0.0s', transform=ax1.transAxes, 
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Texto para estado
    text_estado = ax1.text(0.02, 0.85, 'FUERA DE RANGO ⚠️', transform=ax1.transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightyellow', 
                                     edgecolor='orange', alpha=0.8))
    
    # Texto para rango objetivo
    text_rango = ax1.text(0.02, 0.75, 'Objetivo: 80 km/h | Rango: 78-81 km/h', transform=ax1.transAxes,
                          fontsize=9, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Texto para voltajes VSS y Control en Volts
    text_voltaje = ax1.text(0.02, 0.65, f'VSS: {sistema.velocidad_a_voltaje(sistema.velocidad_nominal):.1f}V (obj) → {sistema.velocidad_a_voltaje(sistema.velocidad_actual):.1f}V (act) | Error: {sistema.error_volts:.3f}V | Ctrl: {sistema.senal_ctrl_offset:.3f}V', transform=ax1.transAxes,
                            fontsize=8, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Crear slider para velocidad nominal
    ax_slider_velocidad = plt.axes([0.25, 0.06, 0.5, 0.02])
    slider_velocidad = Slider(
        ax_slider_velocidad, 'Velocidad Nominal (km/h)', 60, 80,
        valinit=80, valstep=1
    )
    
    # Crear slider para perturbación
    ax_slider_perturbacion = plt.axes([0.25, 0.02, 0.5, 0.02])
    slider_perturbacion = Slider(
        ax_slider_perturbacion, 'Perturbación (RPM)', -100, 200,
        valinit=0, valstep=10
    )
    
    # Configurar animación
    ani = animation.FuncAnimation(
        fig, animar, fargs=(sistema, slider_perturbacion, slider_velocidad, lines, axs, text_time, text_estado, text_rango, text_voltaje),
        interval=100, blit=True, cache_frame_data=False
    )
    
    # Función para actualizar el rango visual cuando cambia la velocidad nominal
    def actualizar_rango_visual(val):
        velocidad = slider_velocidad.val
        rango_min = velocidad + sistema.rango_offset_min
        rango_max = velocidad + sistema.rango_offset_max
        
        # Actualizar el patch del rango en ax1
        rango_patch.remove()
        nuevo_rango = ax1.axhspan(rango_min, rango_max, alpha=0.2, color='green', label='Rango objetivo')
        
        # Actualizar línea de objetivo
        line_obj.set_data([ax1.get_xlim()[0], ax1.get_xlim()[1]], [velocidad, velocidad])
        
        fig.canvas.draw_idle()
    
    # Conectar el slider de velocidad a la función de actualización
    slider_velocidad.on_changed(actualizar_rango_visual)
    
    print("=== SISTEMA DE CONTROL PIT LANE ASSISTANCE (VSS 0-5V) ===")
    print("CARACTERÍSTICAS:")
    print("• Error de entrada: V_nominal - V_actual (V)")
    print("• Señal de control: 0V (disminuir máx) a 5V (aumentar máx), 2.5V (neutral)")
    print("• Señal VSS: 0V (50 km/h) a 5V (100 km/h)")
    print("")
    print("CONTROLES:")
    print("• Slider SUPERIOR: Velocidad nominal deseada")
    print("• Slider INFERIOR: Perturbación aplicada")
    print("")
    print("=" * 140)
    print("T (s) | θi (km/h) | θ₀ (km/h) | eV (V) | θ₀c (V) | p (RPM) | Rango")
    print("-" * 140)
    
    plt.show()

if __name__ == "__main__":
    main()