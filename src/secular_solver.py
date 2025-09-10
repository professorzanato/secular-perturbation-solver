"""
Solução do sistema secular de Laplace-Lagrange para evolução orbital de longo prazo.

Baseado em Murray & Dermott (1999), Solar System Dynamics, Capítulo 7.
Implementa a solução completa das equações seculares para excentricidades
e inclinações usando a teoria linear de perturbações.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import warnings

# Suprimir warnings de divisão por zero
warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class SecularSolution:
    """Armazena a solução completa do sistema secular."""
    time_array: np.ndarray
    eccentricities: Dict[str, np.ndarray]  # e_j(t) para cada planeta
    inclinations: Dict[str, np.ndarray]    # I_j(t) para cada planeta
    longitudes_peri: Dict[str, np.ndarray] # ϖ_j(t) para cada planeta  
    longitudes_node: Dict[str, np.ndarray] # Ω_j(t) para cada planeta
    parameters: Dict[str, Any]             # Parâmetros do sistema
    modes: Dict[str, Any]                  # Modos normais e frequências

class SecularSolver:
    """
    Resolvedor do sistema secular de Laplace-Lagrange.
    
    Resolve as equações:
    h_j(t) = Σ e_ji sin(g_i t + β_i)
    k_j(t) = Σ e_ji cos(g_i t + β_i)
    p_j(t) = Σ I_ji sin(f_i t + γ_i)  
    q_j(t) = Σ I_ji cos(f_i t + γ_i)
    
    onde h_j = e_j sinϖ_j, k_j = e_j cosϖ_j,
          p_j = I_j sinΩ_j, q_j = I_j cosΩ_j.
    """
    
    def __init__(self, secular_results: Dict[str, Any], planet_names: List[str],
                 initial_conditions: Dict[str, np.ndarray]):
        """
        Inicializa o resolvedor com resultados matriciais e condições iniciais.
        
        Parameters
        ----------
        secular_results : Dict[str, Any]
            Resultados do cálculo matricial (matrizes A, B, autovalores, autovetores)
        planet_names : List[str]
            Nomes dos planetas (ex: ['Jupiter', 'Saturn'])
        initial_conditions : Dict[str, np.ndarray]
            Condições iniciais h_j(0), k_j(0), p_j(0), q_j(0) para cada planeta
        """
        
        self.results = secular_results
        self.planet_names = planet_names
        self.n_planets = len(planet_names)
        self.initial_conditions = initial_conditions
        
        # Extrai autovalores e autovetores
        self.g_freq = secular_results['g_frequencies']  # Frequências excentricidades
        self.f_freq = secular_results['f_frequencies']  # Frequências inclinações
        self.g_modes = secular_results['g_eigenvectors']  # Autovetores excentricidades
        self.f_modes = secular_results['f_eigenvectors']  # Autovetores inclinações
        
        # Determina constantes de integração
        self.amplitudes, self.phases = self._determine_integration_constants()
    
    def _determine_integration_constants(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Determina amplitudes e fases a partir das condições iniciais.
        
        Resolve o sistema:
        h_j(0) = Σ e_ji sin(β_i)
        k_j(0) = Σ e_ji cos(β_i)
        p_j(0) = Σ I_ji sin(γ_i)
        q_j(0) = Σ I_ji cos(γ_i)
        
        Returns
        -------
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
            Amplitudes e fases para excentricidades e inclinações
        """
        
        # Prepara matrizes para resolver sistema linear
        n_modes = self.n_planets
        
        # Para excentricidades
        h0 = np.array([self.initial_conditions['h'][i] for i in range(self.n_planets)])
        k0 = np.array([self.initial_conditions['k'][i] for i in range(self.n_planets)])
        
        # Matriz de autovetores para excentricidades
        E_matrix = self.g_modes
        
        # Resolve: h0 = E · (S sinβ), k0 = E · (S cosβ)
        # onde S_i são as amplitudes dos modos
        sin_beta = np.linalg.solve(E_matrix, h0)
        cos_beta = np.linalg.solve(E_matrix, k0)
        
        # Calcula amplitudes S_i e fases β_i
        S = np.sqrt(sin_beta**2 + cos_beta**2)
        beta = np.arctan2(sin_beta, cos_beta) * 180/np.pi  # Em graus
        
        # Para inclinações
        p0 = np.array([self.initial_conditions['p'][i] for i in range(self.n_planets)])
        q0 = np.array([self.initial_conditions['q'][i] for i in range(self.n_planets)])
        
        # Matriz de autovetores para inclinações
        I_matrix = self.f_modes
        
        # Resolve: p0 = I · (T sinγ), q0 = I · (T cosγ)
        sin_gamma = np.linalg.solve(I_matrix, p0)
        cos_gamma = np.linalg.solve(I_matrix, q0)
        
        # Calcula amplitudes T_i e fases γ_i
        T = np.sqrt(sin_gamma**2 + cos_gamma**2)
        gamma = np.arctan2(sin_gamma, cos_gamma) * 180/np.pi  # Em graus
        
        return (
            {'eccentricity': S, 'inclination': T},
            {'eccentricity': beta, 'inclination': gamma}
        )
    
    def _compute_eccentricity_variables(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula h_j(t) e k_j(t) para um dado tempo.
        
        Parameters
        ----------
        t : float
            Tempo em anos
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Arrays h_j(t) e k_j(t) para todos os planetas
        """
        h = np.zeros(self.n_planets)
        k = np.zeros(self.n_planets)
        
        S = self.amplitudes['eccentricity']
        beta = self.phases['eccentricity'] * np.pi/180  # Converte para radianos
        
        for j in range(self.n_planets):
            for i in range(self.n_planets):
                angle = self.g_freq[i] * t + beta[i]
                contribution = S[i] * self.g_modes[j, i]
                h[j] += contribution * np.sin(angle)
                k[j] += contribution * np.cos(angle)
        
        return h, k
    
    def _compute_inclination_variables(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula p_j(t) e q_j(t) para um dado tempo.
        
        Parameters
        ----------
        t : float
            Tempo em anos
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Arrays p_j(t) e q_j(t) para todos os planetas
        """
        p = np.zeros(self.n_planets)
        q = np.zeros(self.n_planets)
        
        T = self.amplitudes['inclination']
        gamma = self.phases['inclination'] * np.pi/180  # Converte para radianos
        
        for j in range(self.n_planets):
            for i in range(self.n_planets):
                angle = self.f_freq[i] * t + gamma[i]
                contribution = T[i] * self.f_modes[j, i]
                p[j] += contribution * np.sin(angle)
                q[j] += contribution * np.cos(angle)
        
        return p, q
    
    def compute_orbital_elements(self, t: float) -> Dict[str, np.ndarray]:
        """
        Calcula elementos orbitais para um tempo específico.
        
        Parameters
        ----------
        t : float
            Tempo em anos
        
        Returns
        -------
        Dict[str, np.ndarray]
            Elementos orbitais: e, I, ϖ, Ω para todos os planetas
        """
        h, k = self._compute_eccentricity_variables(t)
        p, q = self._compute_inclination_variables(t)
        
        # Calcula elementos orbitais
        eccentricities = np.sqrt(h**2 + k**2)
        inclinations = np.sqrt(p**2 + q**2) * 180/np.pi  # Converte para graus
        
        longitudes_peri = np.arctan2(h, k) * 180/np.pi    # ϖ em graus
        longitudes_node = np.arctan2(p, q) * 180/np.pi    # Ω em graus
        
        # Ajusta ângulos para intervalo [0, 360)
        longitudes_peri = np.mod(longitudes_peri, 360)
        longitudes_node = np.mod(longitudes_node, 360)
        
        return {
            'eccentricity': eccentricities,
            'inclination': inclinations,
            'longitude_peri': longitudes_peri,
            'longitude_node': longitudes_node
        }
    
    def solve(self, time_span: Tuple[float, float], time_step: float = 100.0) -> SecularSolution:
        """
        Resolve o sistema secular para um intervalo de tempo.
        
        Parameters
        ----------
        time_span : Tuple[float, float]
            (t_inicial, t_final) em anos
        time_step : float, optional
            Passo temporal em anos (padrão: 100 anos)
        
        Returns
        -------
        SecularSolution
            Solução completa do sistema secular
        """
        t_start, t_end = time_span
        time_array = np.arange(t_start, t_end + time_step, time_step)
        n_times = len(time_array)
        
        # Inicializa arrays de resultados
        eccentricities = {name: np.zeros(n_times) for name in self.planet_names}
        inclinations = {name: np.zeros(n_times) for name in self.planet_names}
        longitudes_peri = {name: np.zeros(n_times) for name in self.planet_names}
        longitudes_node = {name: np.zeros(n_times) for name in self.planet_names}
        
        # Calcula solução para cada tempo
        for idx, t in enumerate(time_array):
            elements = self.compute_orbital_elements(t)
            
            for j, name in enumerate(self.planet_names):
                eccentricities[name][idx] = elements['eccentricity'][j]
                inclinations[name][idx] = elements['inclination'][j]
                longitudes_peri[name][idx] = elements['longitude_peri'][j]
                longitudes_node[name][idx] = elements['longitude_node'][j]
        
        # Prepara parâmetros para retorno
        parameters = {
            'g_frequencies': self.g_freq,
            'f_frequencies': self.f_freq,
            'time_span': time_span,
            'time_step': time_step
        }
        
        modes = {
            'eccentricity_amplitudes': self.amplitudes['eccentricity'],
            'eccentricity_phases': self.phases['eccentricity'],
            'inclination_amplitudes': self.amplitudes['inclination'],
            'inclination_phases': self.phases['inclination']
        }
        
        return SecularSolution(
            time_array=time_array,
            eccentricities=eccentricities,
            inclinations=inclinations,
            longitudes_peri=longitudes_peri,
            longitudes_node=longitudes_node,
            parameters=parameters,
            modes=modes
        )

# Função de conveniência para criar condições iniciais
def create_initial_conditions(eccentricities: List[float], inclinations: List[float],
                             longitudes_peri: List[float], longitudes_node: List[float]) -> Dict[str, np.ndarray]:
    """
    Cria condições iniciais a partir de elementos orbitais.
    
    Parameters
    ----------
    eccentricities : List[float]
        Excentricidades iniciais
    inclinations : List[float]
        Inclinações iniciais em graus
    longitudes_peri : List[float]
        Longitudes do periélio iniciais em graus
    longitudes_node : List[float]
        Longitudes do nó ascendente iniciais em graus
    
    Returns
    -------
    Dict[str, np.ndarray]
        Condições iniciais h, k, p, q
    """
    # Converte para radianos
    inclinations_rad = np.array(inclinations) * np.pi/180
    longitudes_peri_rad = np.array(longitudes_peri) * np.pi/180
    longitudes_node_rad = np.array(longitudes_node) * np.pi/180
    
    # Calcula variáveis de Poincaré
    h = eccentricities * np.sin(longitudes_peri_rad)
    k = eccentricities * np.cos(longitudes_peri_rad)
    p = inclinations_rad * np.sin(longitudes_node_rad)
    q = inclinations_rad * np.cos(longitudes_node_rad)
    
    return {'h': h, 'k': k, 'p': p, 'q': q}

# Testes unitários
if __name__ == "__main__":
    print("Teste do resolvedor secular para Júpiter-Saturno:")
    
    # Dados de exemplo (valores simplificados)
    from src.matrix_calculations import calculate_secular_system
    
    # Parâmetros do sistema
    central_mass = 1.0
    masses = [9.54786e-4, 2.85837e-4]
    semi_major_axes = [5.202545, 9.554841]
    mean_motions = [30.3374, 12.1890]
    planet_names = ['Jupiter', 'Saturn']
    
    # Condições iniciais (1983)
    eccentricities = [0.0474622, 0.0575481]
    inclinations = [1.30667, 2.48795]  # graus
    longitudes_peri = [13.983865, 88.719425]  # graus
    longitudes_node = [100.0381, 113.1334]  # graus
    
    # Calcula matrizes seculares
    secular_results = calculate_secular_system(
        central_mass, masses, semi_major_axes, mean_motions
    )
    
    # Cria condições iniciais
    initial_conditions = create_initial_conditions(
        eccentricities, inclinations, longitudes_peri, longitudes_node
    )
    
    # Cria e executa resolvedor
    solver = SecularSolver(secular_results, planet_names, initial_conditions)
    solution = solver.solve(time_span=(0, 10000), time_step=100)
    
    print(f"Solução calculada para {len(solution.time_array)} pontos no tempo")
    print(f"Excentricidade de Júpiter no final: {solution.eccentricities['Jupiter'][-1]:.6f}")
    print(f"Inclinação de Saturno no final: {solution.inclinations['Saturn'][-1]:.6f}°")
    
    # Mostra frequências seculares
    print(f"\nFrequências seculares:")
    print(f"Excentricidades (g): {solution.parameters['g_frequencies']} °/ano")
    print(f"Inclinações (f): {solution.parameters['f_frequencies']} °/ano")
    
    # Mostra períodos
    g_periods = 360.0 / np.abs(solution.parameters['g_frequencies'])
    f_periods = 360.0 / np.abs(solution.parameters['f_frequencies'])
    print(f"\nPeríodos seculares:")
    print(f"Excentricidades: {g_periods[0]:.0f} anos, {g_periods[1]:.0f} anos")
    print(f"Inclinações: {f_periods[0]:.0f} anos, {f_periods[1]:.0f} anos")