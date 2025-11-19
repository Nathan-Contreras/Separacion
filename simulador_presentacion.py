import numpy as np
from scipy.optimize import fsolve, brentq
import matplotlib.pyplot as plt

# ==========================================
# 1. DATOS DE ENTRADA (EJERCICIO BIOGÁS)
# ==========================================
zf = 0.50          # Alimentación (50% CO2)
theta_target = 0.40 # Corte de etapa (40%)
Ph = 20.0          # Presión Alta [bar]
Pl = 1.0           # Presión Baja [bar]
alpha = 20.0       # Selectividad

# Relación de presiones
r = Pl / Ph

# ==========================================
# 2. SOLVER: MEZCLA PERFECTA (Tus Diapositivas)
# ==========================================
def resolver_mezcla_perfecta():
    def sistema(vars):
        yp, xr = vars
        # Ec 1: Balance Global
        eq1 = zf - ((1 - theta_target) * xr + theta_target * yp)
        # Ec 2: Transporte (Weller-Steiner)
        lhs = yp / (1 - yp)
        rhs = alpha * ((xr - r * yp) / ((1 - xr) - r * (1 - yp)))
        return [eq1, lhs - rhs]

    # Resolver
    yp_mix, xr_mix = fsolve(sistema, [0.8, 0.2])
    return yp_mix, xr_mix

# ==========================================
# 3. SOLVER: COCORRIENTE (Simulación Paso a Paso)
# ==========================================
def resolver_cocorriente():
    # Función local de equilibrio (segura)
    def error_equilibrio(y, x_loc):
        lhs = y / (1 - y)
        den = ((1 - x_loc) - r * (1 - y))
        if abs(den) < 1e-9: den = 1e-9
        rhs = alpha * ((x_loc - r * y) / den)
        return lhs - rhs

    # Integración numérica
    paso = 0.001
    theta_actual = 0.0
    x_curr = zf
    y_acum_num = 0.0
    
    # Listas para graficar
    traj_theta = [0.0]
    traj_x = [x_curr]
    traj_y_avg = [] # Y acumulado (promedio hasta ese punto)
    
    # Para el primer punto y_avg es el equilibrio con la entrada
    y_ini = brentq(error_equilibrio, 0.0001, 0.9999, args=(x_curr))
    traj_y_avg.append(y_ini)

    while theta_actual < theta_target:
        # 1. Hallar y instantáneo local
        try:
            y_local = brentq(error_equilibrio, 0.0001, 0.9999, args=(x_curr))
        except:
            y_local = x_curr # Fallback

        # 2. Balance diferencial (Euler)
        dx = ((x_curr - y_local) / (1 - theta_actual)) * paso
        
        x_curr = x_curr + dx
        y_acum_num += y_local * paso
        theta_actual += paso
        
        # Guardar trayectoria
        y_avg_actual = y_acum_num / theta_actual
        traj_theta.append(theta_actual)
        traj_x.append(x_curr)
        traj_y_avg.append(y_avg_actual)
        
    return traj_theta, traj_x, traj_y_avg

# ==========================================
# 4. EJECUCIÓN Y SALIDA (VERSIÓN SIN PANDAS)
# ==========================================

# A. Calcular Mezcla Perfecta
yp_mp, xr_mp = resolver_mezcla_perfecta()

# B. Calcular Cocorriente
t_vec, x_vec, y_vec = resolver_cocorriente()
yp_coco = y_vec[-1]
xr_coco = x_vec[-1]

# C. Mostrar Resultados con print normal
print("\n" + "="*70)
print(f"COMPARACIÓN DE MODELOS (Alimentación: {zf*100}% CO2)")
print("="*70)

# Encabezados de la tabla manual
encabezado = f"{'MODELO':<35} | {'yp (Permeado)':<15} | {'xr (Retenido)':<15}"
print(encabezado)
print("-" * 70)

# Fila 1: Mezcla Perfecta
print(f"{'Mezcla Perfecta (Diapositivas)':<35} | {yp_mp*100:>14.2f}% | {xr_mp*100:>14.2f}%")

# Fila 2: Cocorriente
print(f"{'Flujo Cocorriente (Simulación)':<35} | {yp_coco*100:>14.2f}% | {xr_coco*100:>14.2f}%")

print("-" * 70)
# Fila 3: Diferencia
diff_yp = (yp_coco - yp_mp) * 100
diff_xr = (xr_coco - xr_mp) * 100
print(f"{'Diferencia (Mejora)':<35} | {diff_yp:>14.2f}% | {diff_xr:>14.2f}%")
print("="*70 + "\n")

# ==========================================
# 5. GRÁFICA COMPARATIVA
# ==========================================
plt.figure(figsize=(12, 7))

# --- Curvas Cocorriente ---
plt.plot(t_vec, y_vec, color='green', linewidth=2.5, label='Permeado Acumulado (Cocorriente)')
plt.plot(t_vec, x_vec, color='blue', linewidth=2.5, label='Retenido (Cocorriente)')

# --- Puntos Mezcla Perfecta ---
# Dibujamos líneas horizontales discontinuas para mostrar que asume valor constante
plt.plot([0, theta_target], [yp_mp, yp_mp], color='lime', linestyle='--', linewidth=2, label='Permeado (Mezcla Perfecta)')
plt.plot([0, theta_target], [xr_mp, xr_mp], color='cyan', linestyle='--', linewidth=2, label='Retenido (Mezcla Perfecta)')

# Puntos finales
plt.scatter([theta_target], [yp_coco], color='darkgreen', s=100, zorder=5)
plt.scatter([theta_target], [xr_coco], color='darkblue', s=100, zorder=5)
plt.scatter([theta_target], [yp_mp], color='lime', edgecolors='black', s=100, zorder=5, marker='s')
plt.scatter([theta_target], [xr_mp], color='cyan', edgecolors='black', s=100, zorder=5, marker='s')

# Decoración
plt.title(f'Comparación: Trayectoria Real vs. Modelo Simplificado\n(Alimentación {zf*100}% CO2, Corte {theta_target*100}%)', fontsize=14)
plt.xlabel('Corte de Etapa ($\Theta$)', fontsize=12)
plt.ylabel('Fracción Molar de CO2', fontsize=12)
plt.axvline(theta_target, color='gray', linestyle=':', label='Corte Final (0.4)')
plt.legend(loc='center right', frameon=True, shadow=True)
plt.grid(True, alpha=0.3)
plt.xlim(0, 0.45)
plt.ylim(0, 1.0)

# Anotaciones
plt.annotate(f'{yp_coco*100:.1f}%', xy=(theta_target, yp_coco), xytext=(theta_target+0.01, yp_coco), color='darkgreen', fontweight='bold')
plt.annotate(f'{yp_mp*100:.1f}%', xy=(theta_target, yp_mp), xytext=(theta_target+0.01, yp_mp-0.03), color='green')
plt.annotate(f'{xr_mp*100:.1f}%', xy=(theta_target, xr_mp), xytext=(theta_target+0.01, xr_mp+0.02), color='teal')
plt.annotate(f'{xr_coco*100:.1f}%', xy=(theta_target, xr_coco), xytext=(theta_target+0.01, xr_coco-0.03), color='darkblue', fontweight='bold')

plt.tight_layout()
plt.show()