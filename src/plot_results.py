"""
Visualização dos resultados da teoria secular de Laplace-Lagrange.

Gera gráficos da evolução temporal de excentricidades, inclinações,
e outros elementos orbitais baseados na solução secular.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import matplotlib.gridspec as gridspec
from dataclasses import dataclass

# Configuração do estilo matplotlib
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman'],
    'mathtext.fontset': 'cm',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (10, 8),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

class SecularPlotter:
    """
    Classe para visualização dos resultados da teoria secular.
    
    Gera gráficos similares à Figura 7.1 de Murray & Dermott (1999),
    mostrando a evolução secular de excentricidades e inclinações.
    """
    
    def __init__(self, solution):
        """
        Inicializa o plotter com a solução secular.
        
        Parameters
        ----------
        solution : SecularSolution
            Solução do sistema secular
        """
        self.solution = solution
        self.planet_names = list(solution.eccentricities.keys())
        self.n_planets = len(self.planet_names)
        
    def plot_eccentricity_evolution(self, ax: plt.Axes = None, 
                                   colors: List[str] = None,
                                   **plot_kwargs) -> plt.Axes:
        """
        Plota a evolução temporal das excentricidades.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        if colors is None:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:self.n_planets]
        
        time_years = self.solution.time_array
        
        for i, name in enumerate(self.planet_names):
            ecc = self.solution.eccentricities[name]
            ax.plot(time_years, ecc, color=colors[i], 
                   label=name, linewidth=2, **plot_kwargs)
        
        # Configurações do gráfico
        ax.set_xlabel('Tempo (anos)')
        ax.set_ylabel('Excentricidade')
        ax.set_title('Evolução Secular das Excentricidades')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Formatação do eixo x para grandes intervalos de tempo
        if time_years[-1] > 10000:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, p: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'
            ))
            ax.set_xlabel('Tempo (mil anos)')
        
        return ax
    
    def plot_inclination_evolution(self, ax: plt.Axes = None,
                                  colors: List[str] = None,
                                  **plot_kwargs) -> plt.Axes:
        """
        Plota a evolução temporal das inclinações.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        if colors is None:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:self.n_planets]
        
        time_years = self.solution.time_array
        
        for i, name in enumerate(self.planet_names):
            inc = self.solution.inclinations[name]
            ax.plot(time_years, inc, color=colors[i],
                   label=name, linewidth=2, **plot_kwargs)
        
        # Configurações do gráfico
        ax.set_xlabel('Tempo (anos)')
        ax.set_ylabel('Inclinação (graus)')
        ax.set_title('Evolução Secular das Inclinações')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Formatação do eixo x para grandes intervalos de tempo
        if time_years[-1] > 10000:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, p: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'
            ))
            ax.set_xlabel('Tempo (mil anos)')
        
        return ax
    
    def plot_phase_space(self, fig: plt.Figure = None,
                        colors: List[str] = None) -> plt.Figure:
        """
        Plota o espaço de fase das variáveis seculares.
        """
        if fig is None:
            fig = plt.figure(figsize=(12, 10))
        
        if colors is None:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:self.n_planets]
        
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Plot (h, k) - espaço de fase das excentricidades
        ax1 = fig.add_subplot(gs[0, 0])
        for i, name in enumerate(self.planet_names):
            h = np.array([e * np.sin(w * np.pi/180) for e, w in 
                         zip(self.solution.eccentricities[name], 
                             self.solution.longitudes_peri[name])])
            k = np.array([e * np.cos(w * np.pi/180) for e, w in 
                         zip(self.solution.eccentricities[name], 
                             self.solution.longitudes_peri[name])])
            ax1.plot(k, h, color=colors[i], label=name, linewidth=1.5)
            ax1.plot(k[0], h[0], 'o', color=colors[i], markersize=6)  # Ponto inicial
        
        ax1.set_xlabel('$k = e \cos\\varpi$')
        ax1.set_ylabel('$h = e \sin\\varpi$')
        ax1.set_title('Espaço de Fase: Excentricidades')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axis('equal')
        
        # Plot (p, q) - espaço de fase das inclinações
        ax2 = fig.add_subplot(gs[0, 1])
        for i, name in enumerate(self.planet_names):
            p = np.array([i * np.sin(w * np.pi/180) for i, w in 
                         zip(self.solution.inclinations[name], 
                             self.solution.longitudes_node[name])])
            q = np.array([i * np.cos(w * np.pi/180) for i, w in 
                         zip(self.solution.inclinations[name], 
                             self.solution.longitudes_node[name])])
            ax2.plot(q, p, color=colors[i], label=name, linewidth=1.5)
            ax2.plot(q[0], p[0], 'o', color=colors[i], markersize=6)  # Ponto inicial
        
        ax2.set_xlabel('$q = I \cos\\Omega$')
        ax2.set_ylabel('$p = I \sin\\Omega$')
        ax2.set_title('Espaço de Fase: Inclinações')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.axis('equal')
        
        # Plot longitude do periélio
        ax3 = fig.add_subplot(gs[1, 0])
        for i, name in enumerate(self.planet_names):
            time_kyr = self.solution.time_array / 1000  # mil anos
            ax3.plot(time_kyr, self.solution.longitudes_peri[name], 
                    color=colors[i], label=name, linewidth=1.5)
        
        ax3.set_xlabel('Tempo (mil anos)')
        ax3.set_ylabel('Longitude do Periélio (graus)')
        ax3.set_title('Evolução da Longitude do Periélio')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot longitude do nó
        ax4 = fig.add_subplot(gs[1, 1])
        for i, name in enumerate(self.planet_names):
            time_kyr = self.solution.time_array / 1000  # mil anos
            ax4.plot(time_kyr, self.solution.longitudes_node[name], 
                    color=colors[i], label=name, linewidth=1.5)
        
        ax4.set_xlabel('Tempo (mil anos)')
        ax4.set_ylabel('Longitude do Nó (graus)')
        ax4.set_title('Evolução da Longitude do Nó')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_murray_figure_7_1(self, save_path: str = None) -> plt.Figure:
        """
        Recria a Figura 7.1 de Murray & Dermott (1999).
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Cores no estilo Murray & Dermott
        colors = ['#0066cc', '#cc6600']  # Azul e laranja
        
        time_kyr = self.solution.time_array / 1000  # mil anos
        
        # Painel superior: Excentricidades
        for i, name in enumerate(self.planet_names):
            ax1.plot(time_kyr, self.solution.eccentricities[name],
                    color=colors[i], linewidth=2, label=name)
        
        ax1.set_ylabel('Excentricidade')
        ax1.set_title('Evolução Secular de Júpiter e Saturno')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(time_kyr[0], time_kyr[-1])
        
        # Painel inferior: Inclinações
        for i, name in enumerate(self.planet_names):
            ax2.plot(time_kyr, self.solution.inclinations[name],
                    color=colors[i], linewidth=2, label=name)
        
        ax2.set_xlabel('Tempo (mil anos)')
        ax2.set_ylabel('Inclinação (graus)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim(time_kyr[0], time_kyr[-1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura salva em: {save_path}")
        
        return fig

def plot_secular_evolution(solution, 
                          plot_type: str = 'combined',
                          save_path: str = None) -> plt.Figure:
    """
    Função conveniente para plotar resultados seculares.
    """
    plotter = SecularPlotter(solution)
    
    if plot_type == 'combined':
        fig = plotter.plot_murray_figure_7_1(save_path)
    elif plot_type == 'eccentricity':
        fig, ax = plt.subplots(figsize=(10, 6))
        plotter.plot_eccentricity_evolution(ax)
    elif plot_type == 'inclination':
        fig, ax = plt.subplots(figsize=(10, 6))
        plotter.plot_inclination_evolution(ax)
    elif plot_type == 'phase':
        fig = plotter.plot_phase_space()
    else:
        raise ValueError("Tipo de plot não reconhecido")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# Testes unitários
if __name__ == "__main__":
    print("Teste do módulo de plotagem secular")
    print("Este módulo requer uma solução secular para testar.")