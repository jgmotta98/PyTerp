# PyTerp

Um interpolador 3D para **Python** projetado para máxima velocidade em grandes conjuntos de dados. Acelera o algoritmo IDW com um núcleo C++ paralelizado (`OpenMP`) e buscas k-NN otimizadas (`nanoflann`).

## Explicação Teórica

A interpolação é realizada em um processo de duas etapas que combina os algoritmos k-NN e IDW.

1. **Seleção de Vizinhos (k-NN)**: Para cada ponto onde se deseja estimar um valor, o algoritmo _k-Nearest Neighbors_ primeiro encontra os k pontos de origem conhecidos mais próximos no espaço. A eficiência desta busca é garantida por uma estrutura de dados otimizada (`k-d tree`).

2. **Cálculo do Valor (IDW)**: Em seguida, o método de _Ponderação pelo Inverso da Distância_ calcula o valor final como uma média ponderada dos k vizinhos encontrados. O peso de cada vizinho é inversamente proporcional à sua distância ($ \text{peso} = \frac{1}{\text{distância}^{p}} $, onde `p` é uma variável de potência), fazendo com que pontos mais próximos tenham uma influência muito maior no resultado.

## Uso

### Pré-requisitos

Antes de começar, garanta que você tenha os seguintes softwares instalados:

* **Python 3.10+**
* **Git**
* **Um compilador C++**: Este pacote contém código C++ que precisa ser compilado durante a instalação.
    * **Windows**: Instale as Ferramentas de Build do Visual Studio (selecione a carga de trabalho "Desenvolvimento para desktop com C++").
    * **Linux (Debian/Ubuntu)**: Instale o essencial de build com: sudo apt-get install build-essential.

---

### Instalação

#### 1. Clone o repositório:

```bash
git clone https://github.com/jgmotta98/PyTerp.git
cd PyTerp
```

#### 2. Crie e ative um ambiente virtual:

```bash
# Crie o ambiente
python -m venv .venv

# Ative o ambiente
# No Windows (cmd.exe):
.venv\Scripts\activate
# No macOS/Linux (bash/zsh):
source .venv/bin/activate
```

#### 3. Instale os requerimentos:

```bash
pip install -r requirements.txt
```

#### 4. Instale o pacote:

```bash
pip install .
```

---

### Exemplo de Uso

```py
import numpy as np
import pyterp as pt

# Supondo que 'source_points', 'source_values' e 'target_points' 
# são arrays NumPy devidamente preparados.
interpolated_values = pt.interpolate(
    source_points=source_points,
    source_values=source_values,
    target_points=target_points,
    k_neighbors=10,
    power=2
)

print("Valores interpolados:", interpolated_values)
```

Para um exemplo completo e executável, incluindo a criação e preparação dos dados de entrada, consulte o script na pasta [examples](examples/basic_usage.py).

## Créditos

Este projeto utiliza `nanoflann`, uma biblioteca C++ de alta performance para o algoritmo _k-Nearest Neighbors_. A eficiência da implementação da árvore k-d do nanoflann é fundamental para o desempenho deste interpolador.

* **Repositório Oficial:** [Nanoflann](https://github.com/jlblancoc/nanoflann)