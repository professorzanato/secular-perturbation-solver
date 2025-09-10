"""
Classe para carregar e validar parâmetros do sistema.
"""

import json
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class BodyParameters:
    """Parâmetros orbitais de um corpo celeste."""
    name: str
    mass: float              # Em massas solares
    semi_major_axis: float   # Em AU
    mean_motion: float       # Em °/ano
    eccentricity: float
    inclination: float       # Em graus
    longitude_peri: float    # ϖ em graus
    longitude_node: float    # Ω em graus

@dataclass
class SystemParameters:
    """Parâmetros do sistema completo."""
    central_body_mass: float  # Massa do corpo central em massas solares
    bodies: List[BodyParameters]
    time_span: tuple         # (t_inicial, t_final) em anos
    time_step: float         # Passo temporal em anos

class ParameterLoader:
    """Carrega e valida parâmetros de arquivo JSON."""
    
    def __init__(self):
        self.parameters = None
    
    def load_from_file(self, filename: str) -> SystemParameters:
        """Carrega parâmetros de arquivo JSON."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return self._parse_parameters(data)
    
    def _parse_parameters(self, data: Dict[str, Any]) -> SystemParameters:
        """Parse dos parâmetros do dicionário."""
        # Verifica corpo central
        if 'central_body' not in data:
            raise ValueError("Parâmetro 'central_body' não encontrado")
        
        central_mass = data['central_body'].get('mass', 1.0)
        
        # Verifica corpos orbitantes
        if 'orbiting_bodies' not in data or not data['orbiting_bodies']:
            raise ValueError("Nenhum corpo orbitante especificado")
        
        bodies = []
        for body_data in data['orbiting_bodies']:
            body = BodyParameters(
                name=body_data.get('name', 'Unnamed'),
                mass=body_data['mass'],
                semi_major_axis=body_data['semi_major_axis'],
                mean_motion=body_data['mean_motion'],
                eccentricity=body_data['eccentricity'],
                inclination=body_data['inclination'],
                longitude_peri=body_data['longitude_peri'],
                longitude_node=body_data['longitude_node']
            )
            bodies.append(body)
        
        # Configuração temporal
        time_config = data.get('time_config', {})
        time_span = time_config.get('time_span', [0, 100000])
        time_step = time_config.get('time_step', 100)
        
        return SystemParameters(
            central_body_mass=central_mass,
            bodies=bodies,
            time_span=tuple(time_span),
            time_step=time_step
        )