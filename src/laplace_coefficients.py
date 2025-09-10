"""
Implementação dos coeficientes de Laplace para a teoria de perturbações seculares.

Baseado em Murray & Dermott (1999), Solar System Dynamics, Capítulo 6.
Os coeficientes de Laplace são integrais fundamentais na expansão do potencial perturbador.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1
import warnings
from typing import Union

# Suprimir warnings de divisão por zero que são tratados manualmente
warnings.filterwarnings('ignore', category=RuntimeWarning)

def laplace_coefficient(s: float, j: int, alpha: float, 
                       method: str = 'quadrature', 
                       tol: float = 1e-10) -> float:
    """
    Calcula o coeficiente de Laplace b_s^{(j)}(α).
    
    O coeficiente de Laplace é definido pela integral:
    b_s^{(j)}(α) = (1/π) ∫₀²π [cos(jψ) / (1 - 2α cosψ + α²)^s] dψ
    
    Parameters
    ----------
    s : float
        Expoente no denominador (geralmente semi-inteiro: 1/2, 3/2, 5/2, ...)
    j : int
        Ordem harmônica (inteiro não-negativo)
    alpha : float
        Razão de semi-eixos maiores (0 ≤ α < 1)
    method : str, optional
        Método de cálculo: 'quadrature' (padrão) ou 'hypergeometric'
    tol : float, optional
        Tolerância para integração numérica (padrão: 1e-10)
    
    Returns
    -------
    float
        Valor do coeficiente de Laplace b_s^{(j)}(α)
    
    Raises
    ------
    ValueError
        Se α não estiver no intervalo [0, 1) ou j for negativo
    """
    
    # Validação dos parâmetros de entrada
    if not 0 <= alpha < 1:
        raise ValueError(f"α deve estar no intervalo [0, 1), recebido: {alpha}")
    if j < 0:
        raise ValueError(f"j deve ser não-negativo, recebido: {j}")
    
    # Caso especial: α = 0
    if alpha == 0:
        return 1.0 if j == 0 else 0.0
    
    # Escolhe o método de cálculo
    if method == 'hypergeometric':
        return _laplace_hypergeometric(s, j, alpha)
    else:
        return _laplace_quadrature(s, j, alpha, tol)

def _laplace_quadrature(s: float, j: int, alpha: float, tol: float = 1e-10) -> float:
    """
    Calcula o coeficiente de Laplace usando quadratura numérica.
    
    Este método é mais geral mas pode ser mais lento que a forma hipergeométrica.
    Usado quando a forma fechada não está disponível ou para verificação.
    
    Parameters
    ----------
    s : float
        Expoente no denominador
    j : int
        Ordem harmônica
    alpha : float
        Razão de semi-eixos maiores
    tol : float
        Tolerância para a integração
    
    Returns
    -------
    float
        Valor do coeficiente de Laplace
    """
    
    def integrand(psi: float) -> float:
        """
        Integrando do coeficiente de Laplace.
        
        Parameters
        ----------
        psi : float
            Ângulo de integração [0, 2π]
        
        Returns
        -------
        float
            Valor do integrando em ψ
        """
        denominator = 1 - 2 * alpha * np.cos(psi) + alpha**2
        # Evita divisão por zero (ψ = 0, α = 1)
        if denominator <= 0:
            return 0.0
        return np.cos(j * psi) / (denominator ** s)
    
    # Calcula a integral numericamente
    integral, error = quad(integrand, 0, 2 * np.pi, epsabs=tol, epsrel=tol)
    
    return integral / np.pi

def _laplace_hypergeometric(s: float, j: int, alpha: float) -> float:
    """
    Calcula o coeficiente de Laplace usando a função hipergeométrica.
    
    Para muitos casos comuns, existe uma forma fechada usando a função
    hipergeométrica ₂F₁. Este método é mais rápido e preciso que a quadratura.
    
    A fórmula geral é:
    b_s^{(j)}(α) = \frac{(s)_j}{j!} α^j ₂F₁(s, s+j; j+1; α²)
    
    onde (s)_j é o símbolo de Pochhammer.
    
    Parameters
    ----------
    s : float
        Expoente no denominador
    j : int
        Ordem harmônica
    alpha : float
        Razão de semi-eixos maiores
    
    Returns
    -------
    float
        Valor do coeficiente de Laplace
    
    References
    ----------
    Murray & Dermott (1999), Appendix B
    """
    
    from scipy.special import poch, hyp2f1
    
    # Caso j = 0 (termo secular)
    if j == 0:
        return hyp2f1(s, s, 1, alpha**2)
    
    # Calcula o símbolo de Pochhammer (s)_j = s(s+1)...(s+j-1)
    pochhammer = poch(s, j)
    
    # Calcula o fatorial de j
    j_factorial = np.math.factorial(j)
    
    # Calcula a função hipergeométrica
    hyp_val = hyp2f1(s, s + j, j + 1, alpha**2)
    
    return (pochhammer / j_factorial) * (alpha ** j) * hyp_val

def laplace_coefficient_derivative(s: float, j: int, alpha: float, 
                                  n: int = 1, method: str = 'quadrature') -> float:
    """
    Calcula a n-ésima derivada do coeficiente de Laplace em relação a α.
    
    A derivada é útil para calcular os termos D^n b_s^{(j)} que aparecem
    na expansão da função perturbadora.
    
    Parameters
    ----------
    s : float
        Expoente no denominador
    j : int
        Ordem harmônica
    alpha : float
        Razão de semi-eixos maiores
    n : int, optional
        Ordem da derivada (padrão: 1)
    method : str, optional
        Método de cálculo
    
    Returns
    -------
    float
        Valor da n-ésima derivada d^n/dα^n [b_s^{(j)}(α)]
    """
    
    # Para primeira derivada, usa fórmula de recorrência
    if n == 1:
        # Fórmula: D b_s^{(j)} = (s/α)[b_{s+1}^{(j-1)} - 2α b_{s+1}^{(j)} + b_{s+1}^{(j+1)}]
        if j == 0:
            term1 = 0.0
        else:
            term1 = laplace_coefficient(s + 1, j - 1, alpha, method)
        
        term2 = 2 * alpha * laplace_coefficient(s + 1, j, alpha, method)
        term3 = laplace_coefficient(s + 1, j + 1, alpha, method)
        
        return (s / alpha) * (term1 - term2 + term3)
    
    # Para derivadas de ordem superior, usa diferenciação numérica
    # (poderia implementar fórmulas analíticas mas fica complexo)
    elif n > 1:
        # Usa diferenciação numérica de segunda ordem
        h = 1e-6
        if alpha > h:
            h = min(h, alpha * 0.01)
        
        # Coeficientes para diferenciação de segunda ordem
        if alpha - h > 0:
            f_minus = laplace_coefficient_derivative(s, j, alpha - h, n - 1, method)
            f_plus = laplace_coefficient_derivative(s, j, alpha + h, n - 1, method)
            return (f_plus - f_minus) / (2 * h)
        else:
            f = laplace_coefficient_derivative(s, j, alpha, n - 1, method)
            f_plus = laplace_coefficient_derivative(s, j, alpha + h, n - 1, method)
            f_plus2 = laplace_coefficient_derivative(s, j, alpha + 2 * h, n - 1, method)
            return (-3 * f + 4 * f_plus - f_plus2) / (2 * h)
    
    else:
        return laplace_coefficient(s, j, alpha, method)

# Coeficientes comuns pré-definidos para fácil acesso
def b12_0(alpha: float) -> float:
    """Coeficiente de Laplace b_{1/2}^{(0)}(α)"""
    return laplace_coefficient(0.5, 0, alpha)

def b12_1(alpha: float) -> float:
    """Coeficiente de Laplace b_{1/2}^{(1)}(α)"""
    return laplace_coefficient(0.5, 1, alpha)

def b32_1(alpha: float) -> float:
    """Coeficiente de Laplace b_{3/2}^{(1)}(α) - muito usado na teoria secular"""
    return laplace_coefficient(1.5, 1, alpha)

def b32_2(alpha: float) -> float:
    """Coeficiente de Laplace b_{3/2}^{(2)}(α) - usado em termos de acoplamento"""
    return laplace_coefficient(1.5, 2, alpha)

def b32_3(alpha: float) -> float:
    """Coeficiente de Laplace b_{3/2}^{(3)}(α)"""
    return laplace_coefficient(1.5, 3, alpha)

# Testes unitários básicos
if __name__ == "__main__":
    # Teste com valores conhecidos
    alpha_test = 0.5
    
    print("Teste dos coeficientes de Laplace:")
    print(f"b_{{1/2}}^{{(0)}}({alpha_test}) = {b12_0(alpha_test):.6f}")
    print(f"b_{{1/2}}^{{(1)}}({alpha_test}) = {b12_1(alpha_test):.6f}")
    print(f"b_{{3/2}}^{{(1)}}({alpha_test}) = {b32_1(alpha_test):.6f}")
    print(f"b_{{3/2}}^{{(2)}}({alpha_test}) = {b32_2(alpha_test):.6f}")
    
    # Teste de derivada
    deriv = laplace_coefficient_derivative(1.5, 1, alpha_test)
    print(f"D b_{{3/2}}^{{(1)}}({alpha_test}) = {deriv:.6f}")
    
    # Comparação com valores de Murray & Dermott para α = 0.544493
    alpha_js = 0.544493
    print(f"\nPara Júpiter-Saturn (α = {alpha_js}):")
    print(f"b_{{3/2}}^{{(1)}} = {b32_1(alpha_js):.6f} (esperado: ~3.17296)")
    print(f"b_{{3/2}}^{{(2)}} = {b32_2(alpha_js):.6f} (esperado: ~2.07110)")