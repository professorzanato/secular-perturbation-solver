"""
Programa principal para cálculo e visualização de perturbações seculares.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Adiciona o diretório atual ao path para importar módulos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Tentando importar módulos...")

# Importa módulos próprios com verificação individual
try:
    print("1. Importando parameters...")
    from parameters import ParameterLoader
    print("   ✓ parameters importado com sucesso!")
except ImportError as e:
    print(f"   ✗ Erro ao importar parameters: {e}")

try:
    print("2. Importando matrix_calculations...")
    from matrix_calculations import calculate_secular_system
    print("   ✓ matrix_calculations importado com sucesso!")
except ImportError as e:
    print(f"   ✗ Erro ao importar matrix_calculations: {e}")

try:
    print("3. Importando secular_solver...")
    from secular_solver import SecularSolver, create_initial_conditions
    print("   ✓ secular_solver importado com sucesso!")
except ImportError as e:
    print(f"   ✗ Erro ao importar secular_solver: {e}")

try:
    print("4. Importando plot_results...")
    from plot_results import SecularPlotter
    print("   ✓ plot_results importado com sucesso!")
except ImportError as e:
    print(f"   ✗ Erro ao importar plot_results: {e}")

# Verifica se todos os módulos foram importados com sucesso
try:
    from parameters import ParameterLoader
    from matrix_calculations import calculate_secular_system
    from secular_solver import SecularSolver, create_initial_conditions
    from plot_results import SecularPlotter
    print("\nTodos os módulos importados com sucesso!")
except ImportError as e:
    print(f"\nErro ao importar módulos: {e}")
    print("Arquivos necessários na pasta src/:")
    print("- parameters.py")
    print("- matrix_calculations.py") 
    print("- laplace_coefficients.py")
    print("- secular_solver.py")
    print("- plot_results.py")
    sys.exit(1)

def load_system_parameters(config_file: str):
    """Carrega parâmetros do sistema a partir de arquivo JSON."""
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    
    loader = ParameterLoader()
    return loader.load_from_file(str(config_path))

def run_secular_analysis(config_file: str, output_dir: str = None):
    """Executa análise secular completa para um sistema planetário."""
    print("Carregando parâmetros do sistema...")
    
    try:
        system_params = load_system_parameters(config_file)
        print(f"Parâmetros carregados: {len(system_params.bodies)} planetas")
    except Exception as e:
        print(f"Erro ao carregar parâmetros: {e}")
        return None
    
    # Extrai arrays para cálculo
    masses = [body.mass for body in system_params.bodies]
    semi_major_axes = [body.semi_major_axis for body in system_params.bodies]
    mean_motions = [body.mean_motion for body in system_params.bodies]
    planet_names = [body.name for body in system_params.bodies]
    
    print("Calculando matrizes seculares...")
    try:
        secular_results = calculate_secular_system(
            system_params.central_body_mass,
            masses,
            semi_major_axes,
            mean_motions
        )
        print("Matrizes calculadas com sucesso!")
    except Exception as e:
        print(f"Erro no cálculo das matrizes: {e}")
        return None
    
    # Condições iniciais
    eccentricities = [body.eccentricity for body in system_params.bodies]
    inclinations = [body.inclination for body in system_params.bodies]
    longitudes_peri = [body.longitude_peri for body in system_params.bodies]
    longitudes_node = [body.longitude_node for body in system_params.bodies]
    
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
        print("Sistema resolvido com sucesso!")
    except Exception as e:
        print(f"Erro na solução do sistema: {e}")
        return None
    
    print("Gerando gráficos...")
    try:
        plotter = SecularPlotter(solution)
        fig_main = plotter.plot_murray_figure_7_1()
        print("Gráficos gerados com sucesso!")
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
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Arquivo de configuração não encontrado: {args.config}")
        return 1
    
    try:
        results = run_secular_analysis(args.config)
        
        if results is None:
            print("Falha na análise secular.")
            return 1
        
        # Mostra resumo
        print("\n" + "="*50)
        print("RESUMO DA ANÁLISE SECULAR")
        print("="*50)
        
        print(f"\nSistema: {len(results['system_params'].bodies)} planetas")
        for body in results['system_params'].bodies:
            print(f"  - {body.name}: a={body.semi_major_axis:.3f} AU, e={body.eccentricity:.4f}, I={body.inclination:.3f}°")
        
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
            print("\nMostrando gráficos... Feche as janelas para continuar.")
            plt.show()
        
        print("\nAnálise concluída com sucesso!")
            
    except Exception as e:
        print(f"Erro durante a execução: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())