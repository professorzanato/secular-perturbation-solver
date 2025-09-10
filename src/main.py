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
try:
    from parameters import ParameterLoader
    from matrix_calculations import calculate_secular_system
    from secular_solver import SecularSolver, create_initial_conditions
    from plot_results import SecularPlotter
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print("Certifique-se de que todos os arquivos estão na pasta src/")
    sys.exit(1)

def load_system_parameters(config_file: str):
    """
    Carrega parâmetros do sistema a partir de arquivo JSON.
    """
    loader = ParameterLoader()
    return loader.load_from_file(config_file)

def run_secular_analysis(config_file: str, output_dir: str = None):
    """
    Executa análise secular completa para um sistema planetário.
    """
    print("Carregando parâmetros do sistema...")
    
    try:
        system_params = load_system_parameters(config_file)
    except FileNotFoundError:
        print(f"Arquivo de configuração não encontrado: {config_file}")
        return None
    
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
    
    print("Calculando matrizes seculares...")
    try:
        secular_results = calculate_secular_system(
            system_params.central_body_mass,
            masses,
            semi_major_axes,
            mean_motions
        )
    except Exception as e:
        print(f"Erro no cálculo das matrizes: {e}")
        return None
    
    # Cria condições iniciais
    initial_conditions = create_initial_conditions(
        eccentricities, inclinations, longitudes_peri, longitudes_node
    )
    
    print("Resolvendo sistema secular...")
    try:
        solver = SecularSolver(secular_results, planet_names, initial_conditions)
        solution = solver.solve(
            time_span=system_params.time_span,
            time_step=system_params.time_step
        )
    except Exception as e:
        print(f"Erro na solução do sistema: {e}")
        return None
    
    print("Gerando gráficos...")
    try:
        plotter = SecularPlotter(solution)
        fig_main = plotter.plot_murray_figure_7_1()
    except Exception as e:
        print(f"Erro na geração de gráficos: {e}")
        return None
    
    return {
        'system_params': system_params,
        'secular_results': secular_results,
        'solution': solution
    }

def main():
    """Função principal do programa."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Solver de Perturbações Seculares')
    parser.add_argument('config', help='Arquivo de configuração JSON')
    parser.add_argument('-p', '--plot', action='store_true', help='Mostrar gráficos interativos')
    
    args = parser.parse_args()
    
    # Verifica se o arquivo de configuração existe
    if not Path(args.config).exists():
        print(f"Arquivo de configuração não encontrado: {args.config}")
        print("Crie primeiro o arquivo examples/jupiter_saturn/jupiter_saturn_input.json")
        return 1
    
    try:
        results = run_secular_analysis(args.config)
        
        if results is None:
            return 1
        
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
            period = 360 / abs(freq) if abs(freq) > 1e-10 else float('inf')
            print(f"  g_{i+1}: {freq:.6f} °/ano (período: {period:.0f} anos)")
        
        for i, freq in enumerate(f_freq):
            period = 360 / abs(freq) if abs(freq) > 1e-10 else float('inf')
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
    # Teste simples se executado diretamente
    print("Solver de Perturbações Seculares")
    print("Use: python src/main.py arquivo_config.json [-p]")
    
    # Tenta encontrar um arquivo de exemplo
    example_files = [
        "examples/jupiter_saturn/jupiter_saturn_input.json",
        "../examples/jupiter_saturn/jupiter_saturn_input.json",
        "jupiter_saturn_input.json"
    ]
    
    for example_file in example_files:
        if Path(example_file).exists():
            print(f"\nExemplo encontrado: {example_file}")
            print(f"Execute: python src/main.py {example_file} -p")
            break
    else:
        print("\nNenhum arquivo de exemplo encontrado.")
        print("Crie primeiro o arquivo examples/jupiter_saturn/jupiter_saturn_input.json")