"""
Cálculo das matrizes seculares A e B para a teoria de Laplace-Lagrange.

Baseado em Murray & Dermott (1999), Solar System Dynamics, Capítulo 7.
As matrizes A e B contêm as frequências e acoplamentos seculares do sistema.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import warnings

# Importa funções de coeficientes de Laplace
from .laplace_coefficients import b32_1, b32_2

# Suprimir warnings de divisão por zero
warnings.filterwarnings('ignore', category=RuntimeWarning)

class SecularMatrixCalculator:
    """
    Calculadora das matrizes seculares A e B para um sistema de N planetas.
    
    As matrizes A (excentricidades) e B (inclinações) são fundamentais
    para a solução das equações seculares de Laplace-Lagrange.
    """
    
    def __init__(self, central_mass: float, masses: List[float], 
                 semi_major_axes: List[float], mean_motions: List[float]):
        """
        Inicializa a calculadora com os parâmetros do sistema.
        
        Parameters
        ----------
        central_mass : float
            Massa do corpo central em massas solares
        masses : List[float]
            Lista de massas dos planetas em massas solares
        semi_major_axes : List[float]
            Lista de semi-eixos maiores em AU
        mean_motions : List[float]
            Lista de movimentos médios em °/ano
        """
        
        self.central_mass = central_mass
        self.masses = np.array(masses)
        self.semi_major_axes = np.array(semi_major_axes)
        self.mean_motions = np.array(mean_motions)
        self.n_planets = len(masses)
        
        # Verifica consistência dos dados
        self._validate_inputs()
        
        # Calcula razões de semi-eixos
        self.alpha_matrix = self._calculate_alpha_matrix()
    
    def _validate_inputs(self) -> None:
        """Valida a consistência dos parâmetros de entrada."""
        if len(self.masses) != self.n_planets:
            raise ValueError("Número de massas inconsistente")
        if len(self.semi_major_axes) != self.n_planets:
            raise ValueError("Número de semi-eixos maiores inconsistente")
        if len(self.mean_motions) != self.n_planets:
            raise ValueError("Número de movimentos médios inconsistente")
        
        if any(m <= 0 for m in self.masses):
            raise ValueError("Massas devem ser positivas")
        if any(a <= 0 for a in self.semi_major_axes):
            raise ValueError("Semi-eixos maiores devem ser positivos")
        if any(n <= 0 for n in self.mean_motions):
            raise ValueError("Movimentos médios devem ser positivos")
    
    def _calculate_alpha_matrix(self) -> np.ndarray:
        """
        Calcula a matriz de razões de semi-eixos α_ij = a_i/a_j.
        
        Para i ≤ j (planeta interno): α_ij = a_i/a_j ≤ 1
        Para i > j (planeta externo): α_ij = a_i/a_j > 1, mas a teoria
        secular usa apenas α_ij < 1, então usamos α_ij = a_j/a_i para i > j.
        
        Returns
        -------
        np.ndarray
            Matriz N×N de razões α_ij (todas ≤ 1)
        """
        alpha_matrix = np.zeros((self.n_planets, self.n_planets))
        
        for i in range(self.n_planets):
            for j in range(self.n_planets):
                if i <= j:
                    # Planeta i é interno ou igual a j
                    alpha_ij = self.semi_major_axes[i] / self.semi_major_axes[j]
                else:
                    # Planeta i é externo a j, usa α = a_j/a_i
                    alpha_ij = self.semi_major_axes[j] / self.semi_major_axes[i]
                
                alpha_matrix[i, j] = alpha_ij
        
        return alpha_matrix
    
    def _get_alpha_tilde(self, i: int, j: int) -> float:
        """
        Determina o fator α̃_ij para correção de perturbação interna/externa.
        
        O fator α̃_ij diferencia entre:
        - Perturbação externa (planeta interno perturbado por externo): α̃ = α
        - Perturbação interna (planeta externo perturbado por interno): α̃ = 1
        
        Parameters
        ----------
        i : int
            Índice do planeta sendo perturbado
        j : int
            Índice do planeta perturbador
        
        Returns
        -------
        float
            Fator α̃_ij (α ou 1)
        """
        if i <= j:
            # Planeta i interno ou igual a j: perturbação externa
            return self.alpha_matrix[i, j]
        else:
            # Planeta i externo a j: perturbação interna
            return 1.0
    
    def calculate_matrix_A(self) -> np.ndarray:
        """
        Calcula a matriz secular A para excentricidades.
        
        A matriz A tem elementos:
        A_ii = +n_i * (1/4) * (m_k/(m_c + m_i)) * α̃ * α * b_{3/2}^{(1)}(α)
        A_ij = -n_i * (1/4) * (m_k/(m_c + m_i)) * α̃ * α * b_{3/2}^{(2)}(α)
        
        onde k ≠ i e α = α_ik.
        
        Returns
        -------
        np.ndarray
            Matriz secular A (N×N) em °/ano
        """
        A = np.zeros((self.n_planets, self.n_planets))
        
        for i in range(self.n_planets):
            for j in range(self.n_planets):
                if i == j:
                    # Elemento diagonal A_ii
                    # Soma sobre todos os outros planetas k ≠ i
                    for k in range(self.n_planets):
                        if k != i:
                            alpha = self.alpha_matrix[i, k]
                            alpha_tilde = self._get_alpha_tilde(i, k)
                            b32_1_val = b32_1(alpha)
                            
                            term = (self.mean_motions[i] / 4) * \
                                   (self.masses[k] / (self.central_mass + self.masses[i])) * \
                                   alpha_tilde * alpha * b32_1_val
                            
                            A[i, i] += term
                
                else:
                    # Elemento não-diagonal A_ij (i ≠ j)
                    alpha = self.alpha_matrix[i, j]
                    alpha_tilde = self._get_alpha_tilde(i, j)
                    b32_2_val = b32_2(alpha)
                    
                    A[i, j] = - (self.mean_motions[i] / 4) * \
                                (self.masses[j] / (self.central_mass + self.masses[i])) * \
                                alpha_tilde * alpha * b32_2_val
        
        return A
    
    def calculate_matrix_B(self) -> np.ndarray:
        """
        Calcula a matriz secular B para inclinações.
        
        A matriz B tem elementos:
        B_ii = -n_i * (1/4) * (m_k/(m_c + m_i)) * α̃ * α * b_{3/2}^{(1)}(α)
        B_ij = +n_i * (1/4) * (m_k/(m_c + m_i)) * α̃ * α * b_{3/2}^{(1)}(α)
        
        onde k ≠ i e α = α_ik.
        
        Returns
        -------
        np.ndarray
            Matriz secular B (N×N) em °/ano
        """
        B = np.zeros((self.n_planets, self.n_planets))
        
        for i in range(self.n_planets):
            for j in range(self.n_planets):
                if i == j:
                    # Elemento diagonal B_ii
                    # Soma sobre todos os outros planetas k ≠ i
                    for k in range(self.n_planets):
                        if k != i:
                            alpha = self.alpha_matrix[i, k]
                            alpha_tilde = self._get_alpha_tilde(i, k)
                            b32_1_val = b32_1(alpha)
                            
                            term = - (self.mean_motions[i] / 4) * \
                                    (self.masses[k] / (self.central_mass + self.masses[i])) * \
                                    alpha_tilde * alpha * b32_1_val
                            
                            B[i, i] += term
                
                else:
                    # Elemento não-diagonal B_ij (i ≠ j)
                    alpha = self.alpha_matrix[i, j]
                    alpha_tilde = self._get_alpha_tilde(i, j)
                    b32_1_val = b32_1(alpha)
                    
                    B[i, j] = + (self.mean_motions[i] / 4) * \
                                (self.masses[j] / (self.central_mass + self.masses[i])) * \
                                alpha_tilde * alpha * b32_1_val
        
        return B
    
    def calculate_secular_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula ambas as matrizes seculares A e B.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Matriz A (excentricidades) e matriz B (inclinações) em °/ano
        """
        A = self.calculate_matrix_A()
        B = self.calculate_matrix_B()
        
        return A, B
    
    def get_secular_frequencies(self) -> Dict[str, Any]:
        """
        Calcula as frequências seculares (autovalores) do sistema.
        
        Returns
        -------
        Dict[str, Any]
            Dicionário com frequências e autovetores
        """
        A, B = self.calculate_secular_matrices()
        
        # Autovalores e autovetores de A (excentricidades)
        g_eigenvalues, g_eigenvectors = np.linalg.eig(A)
        g_indices = np.argsort(g_eigenvalues)  # Ordena por frequência
        g_frequencies = g_eigenvalues[g_indices]
        g_modes = g_eigenvectors[:, g_indices]
        
        # Autovalores e autovetores de B (inclinações)
        f_eigenvalues, f_eigenvectors = np.linalg.eig(B)
        f_indices = np.argsort(f_eigenvalues)  # Ordena por frequência
        f_frequencies = f_eigenvalues[f_indices]
        f_modes = f_eigenvectors[:, f_indices]
        
        return {
            'A_matrix': A,
            'B_matrix': B,
            'g_frequencies': g_frequencies,      # Frequências seculares (excentricidades)
            'g_eigenvectors': g_modes,           # Modos normais (excentricidades)
            'f_frequencies': f_frequencies,      # Frequências seculares (inclinações)
            'f_eigenvectors': f_modes,           # Modos normais (inclinações)
            'alpha_matrix': self.alpha_matrix    # Matriz de razões α
        }

# Função de conveniência para cálculo rápido
def calculate_secular_system(central_mass: float, masses: List[float], 
                           semi_major_axes: List[float], mean_motions: List[float]) -> Dict[str, Any]:
    """
    Função conveniente para calcular as matrizes seculares em um único passo.
    
    Parameters
    ----------
    central_mass : float
        Massa do corpo central
    masses : List[float]
        Lista de massas dos planetas
    semi_major_axes : List[float]
        Lista de semi-eixos maiores
    mean_motions : List[float]
        Lista de movimentos médios
    
    Returns
    -------
    Dict[str, Any]
        Resultados do cálculo secular
    """
    calculator = SecularMatrixCalculator(central_mass, masses, semi_major_axes, mean_motions)
    return calculator.get_secular_frequencies()

# Testes unitários
if __name__ == "__main__":
    # Teste com sistema Júpiter-Saturno
    print("Teste com sistema Júpiter-Saturno:")
    
    # Parâmetros do sistema (Murray & Dermott, 1999)
    central_mass = 1.0  # Massa solar
    masses = [9.54786e-4, 2.85837e-4]  # Júpiter, Saturno
    semi_major_axes = [5.202545, 9.554841]  # AU
    mean_motions = [30.3374, 12.1890]  # °/ano
    
    calculator = SecularMatrixCalculator(central_mass, masses, semi_major_axes, mean_motions)
    
    # Calcula matrizes
    A, B = calculator.calculate_secular_matrices()
    
    print("\nMatriz A (excentricidades):")
    print(A)
    print("\nMatriz B (inclinações):")
    print(B)
    
    # Calcula frequências seculares
    results = calculator.get_secular_frequencies()
    
    print(f"\nFrequências seculares (excentricidades) g_i: {results['g_frequencies']} °/ano")
    print(f"Frequências seculares (inclinações) f_i: {results['f_frequencies']} °/ano")
    
    # Verifica com valores esperados de Murray & Dermott
    print("\nValores esperados (Murray & Dermott):")
    print("A = [[+0.00203738, -0.00132987], [-0.00328007, +0.00502513]] °/ano")
    print("B = [[-0.00203738, +0.00203738], [+0.00502513, -0.00502513]] °/ano")
    print("g1 ≈ 9.63435e-4 °/ano, g2 ≈ 6.09908e-3 °/ano")
    print("f1 = 0 °/ano, f2 ≈ -7.06251e-3 °/ano")
    
    # Calcula períodos seculares em anos
    g_periods = 360.0 / np.abs(results['g_frequencies'])
    f_periods = 360.0 / np.abs(results['f_frequencies'])
    
    print(f"\nPeríodos seculares:")
    print(f"Excentricidades: {g_periods[0]:.0f} anos, {g_periods[1]:.0f} anos")
    print(f"Inclinações: {f_periods[0]:.0f} anos, {f_periods[1]:.0f} anos")