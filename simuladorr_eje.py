# ==========================================
# IMPORTACIÓN DE LIBRERÍAS
# ==========================================

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURACIÓN Y DATOS FIJOS
# ==========================================
F = 100.0          # Alimentación [mol/s]
xf = 0.40          # Composición CO2 en alimentación
Ph = 20.0          # Presión Alimentación [bar]
Pl = 1.0           # Presión Permeado [bar]
alpha = 20.0       # Selectividad
QA = 1e-4          # Permeancia
r = Pl / Ph        # Relación de presiones

# Función para resolver yp dado un xr específico
def resolver_yp(xr_val):
    def ecuacion_yp(yp):
        lhs = yp / (1 - yp)
        rhs = alpha * ((xr_val - r * yp) / ((1 - xr_val) - r * (1 - yp)))
        return lhs - rhs
    # Usamos un valor semilla (ligeramente mayor que xr)
    return fsolve(ecuacion_yp, 0.5)[0]

# Función_ calcular todo el sistema dado un xr (50 INTERACIONES)
# METODO DE NEWTON RAPHSON
def calcular_performance(xr_target):
    yp = resolver_yp(xr_target)
    
    # Corte (Theta)
    theta = (xf - xr_target) / (yp - xr_target)
    
    # Área
    flujo_permeado_CO2 = F * theta * yp
    fuerza_impulsora = (Ph * xr_target) - (Pl * yp)
    
    # El ojo pelao de divisiones por cero
    if fuerza_impulsora <= 0:
        Am = np.nan
    else:
        Am = flujo_permeado_CO2 / (QA * fuerza_impulsora)
        
    return yp, theta, Am

# ==========================================
# GENERACIÓN DE DATOS PARA GRAFICAR
# ==========================================

rango_xr = np.linspace(0.40, 0.02, 50) # 50 puntos entre 0.40 y 0.02

lista_yp = []
lista_theta = []
lista_Am = []

for x in rango_xr:
    y, t, a = calcular_performance(x)
    lista_yp.append(y)
    lista_theta.append(t)
    lista_Am.append(a)

xr_ejercicio = 0.10
yp_ej, theta_ej, Am_ej = calcular_performance(xr_ejercicio)

# ==========================================
# CREACIÓN DE GRÁFICAS 
# ==========================================
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
plt.suptitle(f'Diseño de Membrana: Efecto de la Pureza Requerida ($x_r$)', fontsize=16)

# --- GRÁFICA 1: Composición del Permeado ---
ax[0].plot(rango_xr, lista_yp, label='Curva de Operación', color='blue', linewidth=2)
ax[0].scatter([xr_ejercicio], [yp_ej], color='red', zorder=5, s=100, label='Punto Ejercicio')
ax[0].set_xlabel('Composición Retenido ($x_r$)', fontsize=12)
ax[0].set_ylabel('Composición Permeado ($y_p$)', fontsize=12)
ax[0].set_title('Calidad del Permeado', fontsize=14)
ax[0].invert_xaxis() # Invertimos eje X para mostrar "Mayor pureza" hacia la derecha
ax[0].grid(True, linestyle='--', alpha=0.7)
ax[0].legend()

# --- GRÁFICA 2: Corte de Etapa (Pérdida de gas) ---
ax[1].plot(rango_xr, lista_theta, color='orange', linewidth=2)
ax[1].scatter([xr_ejercicio], [theta_ej], color='red', zorder=5, s=100)
ax[1].set_xlabel('Composición Retenido ($x_r$)', fontsize=12)
ax[1].set_ylabel('Corte de Etapa ($\Theta$)', fontsize=12)
ax[1].set_title('Fracción de Gas que Permea', fontsize=14)
ax[1].invert_xaxis()
ax[1].grid(True, linestyle='--', alpha=0.7)

# --- GRÁFICA 3: Área de Membrana ---
ax[2].plot(rango_xr, lista_Am, color='green', linewidth=2)
ax[2].scatter([xr_ejercicio], [Am_ej], color='red', zorder=5, s=100)
ax[2].set_xlabel('Composición Retenido ($x_r$)', fontsize=12)
ax[2].set_ylabel('Área de Membrana ($m^2$)', fontsize=12)
ax[2].set_title('Área Requerida', fontsize=14)
ax[2].invert_xaxis()
ax[2].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

print(f"--- RESULTADO DEL PUNTO ---")
print(f"Para xr = {xr_ejercicio}:")
print(f"yp = {yp_ej:.3f}")
print(f"Theta = {theta_ej:.3f}")
print(f"Area = {Am_ej:.2f} m^2")