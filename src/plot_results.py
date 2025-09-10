"""
Programa principal para cálculo e visualização de perturbações seculares.

Integra todos os módulos para calcular a evolução secular de sistemas planetários
usando a teoria de Laplace-Lagrange.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Adiciona o diretório src ao path
src_path = Path(__file__).parent
sys.path.append(str(src_path))

# Importa módulos próprios
from parameters import ParameterLoader, BodyParameters, SystemParameters
from matrix_calculations import SecularMatrixCalculator, calculate_secular_system
from secular_solver import SecularSolver, create_initial_conditions
from plot_results import SecularPlotter, plot_secular_evolution

def load_system_parameters(config_file: str) -> SystemParameters:
    """
    Carrega parâmetros do sistema a partir de arquivo JSON.
    
    Parameters
    ----------
    config_file : str
        Caminho para o arquivo de configuração JSON
    
    Returns
    -------
    SystemParameters
        Parâmetros do sistema carregados
    """
    loader = ParameterLoader()
    return loader.load_from_file(config_file)

def run_secular_analysis(config_file: str, output_dir: str = None) -> dict:
    """
    Executa análise secular completa para um sistema planetário.
    
    Parameters
    ----------
    config_file : str
        Caminho para o arquivo de configuração JSON
    output_dir : str, optional
        Diretório para salvar resultados (se None, não salva)
    
    Returns
    -------
    dict
        Resultados completos da análise
    """
    # Carrega parâmetros
    print("Carregando parâmetros do sistema...")
    system_params = load_system_parameters(config_file)
    
    # Extrai arrays para cálculo
    masses = [body.mass for body in system_params.bodies]
    semi_major_axes = [body.semi_major_axis for body in system_params.bodies]
    mean_motions = [body.mean_motion for body in system_params.bodies]
    planet_names = [body.name for body in system_params.bodies]
    
    # Condições iniciais
    eccentricities = [body.eccentricity for body in system_params.bodies]
    inclinations = [body.inclination for body in system_params.bodies]
    longitudes_peri = [body.longitude_peri for body in system_params.bodies]
    longitudes_node = [body.longitude_node for body in system_params.bodies]
    
    # Calcula matrizes seculares
    print("Calculando matrizes seculares...")
    secular_results = calculate_secular_system(
        system_params.central_body_mass,
        masses,
        semi_major_axes,
        mean_motions
    )
    
    # Cria condições iniciais
    initial_conditions = create_initial_conditions(
        eccentricities, inclinations, longitudes_peri, longitudes_node
    )
    
    # Resolve sistema secular
    print("Resolvendo sistema secular...")
    solver = SecularSolver(secular_results, planet_names, initial_conditions)
    solution = solver.solve(
        time_span=system_params.time_span,
        time_step=system_params.time_step
    )
    
    # Gera gráficos
    print("Gerando gráficos...")
    plotter = SecularPlotter(solution)
    
    # Figura principal (estilo Murray & Dermott)
    fig_main = plotter.plot_murray_figure_7_1()
    
    # Gráficos adicionais
    fig_phase = plotter.plot_phase_space()
    stats = plotter.plot_summary_statistics()
    
    # Salva resultados se solicitado
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Salva figuras
        fig_main.savefig(output_path / "secular_evolution.png", dpi=300)
        fig_phase.savefig(output_path / "phase_space.png", dpi=300)
        
        # Salva dados numéricos
        np.savez(output_path / "secular_solution.npz",
                time=solution.time_array,
                eccentricities=solution.eccentricities,
                inclinations=solution.inclinations,
                longitudes_peri=solution.longitudes_peri,
                longitudes_node=solution.longitudes_node)
        
        # Salva estatísticas
        with open(output_path / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Resultados salvos em: {output_path}")
    
    return {
        'system_params': system_params,
        'secular_results': secular_results,
        'solution': solution,
        'statistics': stats
    }

def main():
    """Função principal do programa."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Solver de Perturbações Seculares')
    parser.add_argument('config', help='Arquivo de configuração JSON')
    parser.add_argument('-o', '--output', help='Diretório de saída')
    parser.add_argument('-p', '--plot', action='store_true', help='Mostrar gráficos interativos')
    
    args = parser.parse_args()
    
    try:
        # Executa análise
        results = run_secular_analysis(args.config, args.output)
        
        # Mostra resumo
        print("\n" + "="*50)
        print("RESUMO DA ANÁLISE SECULAR")
        print("="*50)
        
        print(f"\nSistema: {len(results['system_params'].bodies)} planetas")
        for body in results['system_params'].bodies:
            print(f"  - {body.name}: a={body.semi_major_axis:.3f} AU, e={body.eccentricity:.4f}")
        
        print(f"\nFrequências seculares:")
        g_freq = results['secular_results']['g_frequencies']
        f_freq = results['secular_results']['f_frequencies']
        
        for i, freq in enumerate(g_freq):
            period = 360 / abs(freq) if abs(freq) > 1e-10 else np.inf
            print(f"  g_{i+1}: {freq:.6f} °/ano (período: {period:.0f} anos)")
        
        for i, freq in enumerate(f_freq):
            period = 360 / abs(freq) if abs(freq) > 1e-10 else np.inf
            print(f"  f_{i+1}: {freq:.6f} °/ano (período: {period:.0f} anos)")
        
        # Mostra gráficos se solicitado
        if args.plot:
            plt.show()
        
        print("\nAnálise concluída com sucesso!")
            
    except Exception as e:
        print(f"Erro durante a execução: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    # Exemplo de uso direto (para teste)
    try:
        # Tenta carregar exemplo de Júpiter-Saturno
        config_file = "../examples/jupiter_saturn/jupiter_saturn_input.json"
        results = run_secular_analysis(config_file)
        plt.show()
        
    except FileNotFoundError:
        print("Arquivo de exemplo não encontrado. Execute via linha de comando:")
        print("python src/main.py examples/jupiter_saturn/jupiter_saturn_input.json -p")
        
    except Exception as e:
        print(f"Erro: {e}")