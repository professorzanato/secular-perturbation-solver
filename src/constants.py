"""
Constantes astronômicas fundamentais para cálculos de perturbações seculares.
"""

# Unidades: AU, anos, massas solares
AU = 1.0                      # Unidade Astronômica
YEAR = 1.0                    # Ano terrestre
DEGREE = 1.0                  # Grau
RADIAN = 57.29577951308232    # Radiano em graus (180/π)

# Constante gravitacional em AU^3 / (ano^2 * massa_solar)
G = 4 * 3.141592653589793**2  # G = 4π² (quando unidades são AU, ano, massa_solar)

# Conversões úteis
DEG2RAD = 0.017453292519943295  # π/180
RAD2DEG = 57.29577951308232     # 180/π
ARCSEC2DEG = 1/3600.0           # Segundos de arco para graus

# Precisão numérica
EPSILON = 1e-12