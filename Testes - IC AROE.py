# %% [markdown]
# # **Resolução de Problemas de Otimização sob Incerteza em Cadeias de Suprimentos através de Técnicas de Aprendizado por Reforço e Otimização Estocástica**
# 
# ## **Iniciação Científca - CNPq**
# 
# ### **Membros do Projeto**
# 
# * Julio César Alves
# 
# * Dilson Lucas Pereira
# 
# * Marcos Carvalho Ferreira
# 

# %% [markdown]
# ### Pip installs

# %%
# %pip install tqdm
# %pip install gymnasium
# %pip install git+https://github.com/mimoralea/gym-walk#egg=gym-walk
# %pip install tabulate
# %pip install git+https://github.com/ibm-cds-labs/pixiedust
# %pip install -e ./pixiedust

# %% [markdown]
# **Atenção**: após as instalações é recomendável **reiniciar o Kernel**. No VS Code, acesse a paleta de comandos (teclando `F1` OU `Ctrl+Shift+P`) e procurando a opção *Notebook: Restart Kernel* (dica: digite *kernel* na paleta para filtrar os comandos).

# %% [markdown]
# ### Imports

# %%
#   Você pode usar a biblioteca externa do Python tqdm , para criar barras de progresso simples e sem complicações
# que você pode adicionar ao seu código e torná-lo mais animado!
from tqdm import tqdm

# Como implementar? --> Ex: "for e in tqdm(range(n_episodios), desc='Episodes for: ' + nome(string), leave=False):"

# tqdm("Intervalo desejado", "Descrição (desc)", "Leave" Se [padrão: True], mantém
# todos os traços da barra de progresso após o término da iteração):

import gymnasium as gym
import random
import gym_walk
import gym as old_gym
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import sys

# %% [markdown]
# ### Configurações

# %%
SEMENTES = (5, 80, 78, 27, 17)
# SEMENTES = (12, 34, 56, 78, 90) # usadas pelo autor do livro

# %% [markdown]
# ### **Funções úteis**
# 
# * **Funções utilizadas nas estratégias:**

# %%
# Função utilizada para decair alfa e epsilon exponencialmente.
# Você dá um valor inicial, um valor mínimo e a porcentagem de max_iteracoes para decair os valores de inicial para mínimo.
def decaimento_exponencial(valor_ini, valor_min, taxa_decaimento, max_iteracoes, log_inicio=-2, log_base=10):
    # Índice onde o decaimento de valores termina e o valor_min vai até max_iteracoes.
    decaimento_iteracoes = int(max_iteracoes * taxa_decaimento)

    # Diferença entre o máximo de iterações e as iterações de decaimento
    iteracoes_remanescentes = max_iteracoes - decaimento_iteracoes

    # Usando o logspace para gerar uma escala logaritimica começando em log_start (padrão = -2), e terminando em 0. 
    # O número total de valores é decaimento_iteracoes e a base é log_base (padrão é 10). 
    # Perceba que está invertido com [::-1]!
    valor = np.logspace(log_inicio, 0, decaimento_iteracoes, base=log_base, endpoint=True)[::-1]

    # Os valores podem não terminar exatamente em 0, por causa do logaritmo
    # Alteramos para ficar entre 0 e 1 e assim a curva fica suave e bonita
    valor = (valor - valor.min()) / (valor.max() - valor.min())

    # Fazemos a transformação linear entre valor_ini e valor_min.
    valor = (valor_ini - valor_min) * valor + valor_min

    # E então enchemos um array com "valor", a função "pad" repete o valor mais à direita iteracoes_remanescentes número de vezes.
    valor = np.pad(valor, (0, iteracoes_remanescentes), 'edge')
    
    return valor

# %%
import itertools

# Função utilizada para criar uma trajetória necessária para o controle de Monte Carlo

# Esta versão do gerar_trajetoria é um pouco diferente. Agora precisamos levar em uma estratégia de seleção de ação, em vez de uma política gananciosa.

# Você passa a estratégia de seleção de ação, a função Q, o epsilon (Quão gancioso o agente vai ser?), o ambiente e a quantidade maxima de iterações 
# (caso o número de iterações ultrapasse este número a trajetória é retornada vazia (truncar)).
def gerar_trajetoria_epsilon_ganancioso(seleciona_acao, Q, epsilon, amb, max_iteracoes=200):
    # Criamos a variavel bolleana "terminado" que somente se tornará True quando o resultado da iteração levar a um estado terminal
    # Inicializamos o array que irá armazenar a trajetória (vazio)
    terminado, trajetoria = False, []

    # Loop principal
    while not terminado:
        # Resetamos o ambiente para garantir que ele esteja pronto para gerar a trajetória
        estado = amb.reset()

        # Entramos em loop infinito -> só para com break
        for t in itertools.count():
            # Selecionamos a acao para o estado atual de acordo com a estratégia passada
            acao = seleciona_acao(estado, Q, epsilon)

            # Realizamos uma iteração
            prox_estado, recompensa, terminado, _ = amb.step(acao)
            # Guardamos o resultado da iteração em um array
            experiencia = (estado, acao, recompensa, prox_estado, terminado)
            # append --> Acrescentamos a experiencia no final do array "trajetória"
            trajetoria.append(experiencia)

            # Para o loop quando chegamos em um estado terminal
            if terminado:
                break

            # Truncamos a trajetória quando atinje o máximo de iterações
            if t >= max_iteracoes - 1:
                trajetoria = []
                break

            # Passamos para o proximo estado
            estado = prox_estado
            
        return np.array(trajetoria, np.object)

# %% [markdown]
# * **Funções utilizadas para testes:**

# %%
def imprima_funcao_valor_de_estado(V, P, n_cols=4, prec=3, title='Função valor de estado:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

# %%
def imprima_funcao_valor_de_acao(Q, 
                                optimal_Q=None, 
                                action_symbols=('<', '>'), 
                                prec=3, 
                                title='Função Q (valor de ação):'):
    vf_types=('',) if optimal_Q is None else ('', '*', 'err')
    headers = ['s',] + [' '.join(i) for i in list(itertools.product(vf_types, action_symbols))]
    print(title)
    states = np.arange(len(Q))[..., np.newaxis]
    arr = np.hstack((states, np.round(Q, prec)))
    if not (optimal_Q is None):
        arr = np.hstack((arr, np.round(optimal_Q, prec), np.round(optimal_Q-Q, prec)))
    print(tabulate(arr, headers, tablefmt="fancy_grid"))

# %%
def imprima_politica(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

# %%
from itertools import cycle

def plotar_funcao_de_valor(title, V_track, V_true=None, log=False, limit_value=0.05, limit_items=5):
    np.random.seed(123)
    per_col = 25
    linecycler = cycle(["-","--",":","-."])
    legends = []

    valid_values = np.argwhere(V_track[-1] > limit_value).squeeze()
    items_idxs = np.random.choice(valid_values, 
                                  min(valid_values.size, limit_items), 
                                  replace=False)
    # draw the true values first
    if V_true is not None:
        for i, state in enumerate(V_track.T):
            if i not in items_idxs:
                continue
            if state[-1] < limit_value:
                continue

            label = 'v*({})'.format(i)
            plt.axhline(y=V_true[i], color='k', linestyle='-', linewidth=1)
            plt.text(int(len(V_track)*1.02), V_true[i]+.01, label)

    # then the estimates
    for i, state in enumerate(V_track.T):
        if i not in items_idxs:
            continue
        if state[-1] < limit_value:
            continue
        line_type = next(linecycler)
        label = 'V({})'.format(i)
        p, = plt.plot(state, line_type, label=label, linewidth=3)
        legends.append(p)
        
    legends.reverse()

    ls = []
    for loc, idx in enumerate(range(0, len(legends), per_col)):
        subset = legends[idx:idx+per_col]
        l = plt.legend(subset, [p.get_label() for p in subset], 
                       loc='center right', bbox_to_anchor=(1.25, 0.5))
        ls.append(l)
    [plt.gca().add_artist(l) for l in ls[:-1]]
    if log: plt.xscale('log')
    plt.title(title)
    plt.ylabel('State-value function')
    plt.xlabel('Episodes (log scale)' if log else 'Episodes')
    plt.show()

# %% [markdown]
# ## **Estratégias de aprendizagem por Reforço**

# %%
# Implementação do algoritmo de controle de Monte Carlo

# Você passa a política o ambiente para qual o algoritmo criará a política, desconto de recompensa, especificações para a criação do array de 
# decaimento exponencial de alpha e epsilon, número de episódios considerados, quantidade maxima de iterações e se utilizaremos a primeira 
# vista Monte Carlo ou não
def controle_mc(amb, gamma=1.0, ini_alpha=0.5, min_alpha=0.01, taxa_decay_alpha=0.5, ini_epsilon=1.0, min_epsilon=0.1, 
               taxa_decay_epsilon=0.9, n_episodios=3000, max_iteracoes=200, primeira_visita=True):
    # Obtemos o número de estados do ambiente (nS) e o número de ações disponíveis (nA)
    nS, nA = amb.observation_space.n, amb.action_space.n

    # Calculando todos os descontos possíveis de uma vez. 
    # Esta função logspace para uma gama de 0,99 e um max_step de 100 retorna um vetor de número 100: [1, 0,99, 0,9801, . . ., 0,3697].
    descontos = np.logspace(0, max_iteracoes, num=max_iteracoes, base=gamma, endpoint=False)
    # Usando a função criada anteriormente para criar o decaimento exponencial de alpha 
    alphas = decaimento_exponencial(ini_alpha, min_alpha, taxa_decay_alpha, n_episodios)
    # Usando a função criada anteriormente para criar o decaimento exponencial de epsilon 
    epsilons = decaimento_exponencial(ini_epsilon, min_epsilon, taxa_decay_epsilon, n_episodios)

    # Array utilizado para ver o progresso da avaliação da política
    pi_historico = []
    # Inicializando a função com as dimensões do ambiente analisado e os valores zerados
    Q = np.zeros((nS, nA), dtype=np.float64)
    # Array utilizado para ver o progresso da função Q a cada episodio
    Q_historico = np.zeros((n_episodios, nS, nA), dtype=np.float64)

    # Criando a estratégia de seleção de ação: Utilizamos Epsilon Ganancioso
    #     Funções anônimas (Lambda): são  funções que o usuário não precisa definir, ou seja, não vai precisar
    # escrever a função e depois utilizá-la dentro do código.
    seleciona_acao = lambda estado, Q, epsilon: \
        np.argmax(Q[estado]) \
        if np.random.random() > epsilon \
        else np.random.randint(len(Q[estado]))

    # Loop principal com tqdm
    # e --> Número da iteração
    for e in tqdm(range(n_episodios), leave=False):
        # Gerando a trajetória com a função criada anteriormente
        trajetoria = gerar_trajetoria_epsilon_ganancioso(seleciona_acao, Q, epsilons[e], amb, max_iteracoes)

        # Criamos o array que verifica se a trajetória passou pelo mesmo estado
        visitado = np.zeros((nS,nA), dtype=np.bool)

        # Loop secundário
        # t --> Número da iteração
        # (estado, acao, recompensa, _, _) --> Tupla de experiencia da trajetória
        for t, (estado, acao, recompensa, _, _) in enumerate(trajetoria):
            # Caso o estado já foi visitado e se estivermos usando a primeira visita MC
            if visitado[estado][acao] and primeira_visita:
                # Ignoramos tudo e vamos para a próxima iteração do loop
                continue

            # Marcamos que estado foi visitado
            visitado[estado][acao] = True

            # Descobrindo a quantidade de passos até o estado terminal
            n_passos = len(trajetoria[t:])
            # Calculando o Retorno: Somando todas recompensas (com seus respectivos descontos)
            G = np.sum(descontos[:n_passos] * trajetoria[t:, 2])
            # Atualizamos a Função Q (Aplicação da equação de Monte Carlo)
            Q[estado][acao] = Q[estado][acao] + alphas[e] * (G - Q[estado][acao])

        # Guardamos a Função Q atual no histórico
        Q_historico[e] = Q
        # Guardamos a política atual no histórico
        pi_historico.append(np.argmax(Q, axis=1))

    # Extraímos a Função de Valor de Estado selecionando as melhores ações de Q
    V = np.max(Q, axis=1)
    # Com a Função de Valor Feita podemos obter uma política ótima
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return Q, V, pi, Q_historico, pi_historico

# %%
# Implementação do algoritmo SARSA

# Você passa a política o ambiente para qual o algoritmo criará a política, desconto de recompensa, especificações para a criação do array de 
# decaimento exponencial de alpha e epsilon, número de episódios considerados
def sarsa(amb, gamma=1.0, ini_alpha=0.5, min_alpha=0.01, taxa_decay_alpha=0.5, ini_epsilon=1.0, 
          min_epsilon=0.1, taxa_decay_epsilon=0.9, n_episodios=3000):
    # Obtemos o número de estados do ambiente (nS) e o número de ações disponíveis (nA)
    nS, nA = env.observation_space.n, env.action_space.n

    # Array utilizado para ver o progresso da avaliação da política
    pi_historico = []
    # Inicializando a função com as dimensões do ambiente analisado e os valores zerados
    Q = np.zeros((nS, nA), dtype=np.float64)
    # Array utilizado para ver o progresso da função Q a cada episodio
    Q_historico = np.zeros((n_episodios, nS, nA), dtype=np.float64)

    # Criando a estratégia de seleção de ação: Utilizamos Epsilon Ganancioso
    #     Funções anônimas (Lambda): são  funções que o usuário não precisa definir, ou seja, não vai precisar
    # escrever a função e depois utilizá-la dentro do código.
    seleciona_acao = lambda estado, Q, epsilon: \
        np.argmax(Q[estado]) \
        if np.random.random() > epsilon \
        else np.random.randint(len(Q[estado]))

    # Usando a função criada anteriormente para criar o decaimento exponencial de alpha 
    alphas = decaimento_exponencial(ini_alpha, min_alpha, taxa_decay_alpha, n_episodios)
    # Usando a função criada anteriormente para criar o decaimento exponencial de epsilon 
    epsilons = decaimento_exponencial(ini_epsilon, min_epsilon, taxa_decay_epsilon, n_episodios)

    # Loop principal com tqdm
    # e --> Número da iteração
    for e in tqdm(range(n_episodios), leave=False):
        # Resetamos o ambiente para garantir que ele esteja pronto para ser analisado
        # Obtemos o estado inicial e se o estado é terminal
        estado, terminado = amb.reset(), False
        # Selecionamos a acao para o estado atual de acordo com a estratégia passada
        acao = seleciona_acao(estado, Q, epsilons[e])
        
        # Loop secundário --> Só para se entrarmos em um estado terminal
        while not terminado:
            # Obtemos as informações de retorno do passo dado
            prox_estado, recompensa, terminado, _ = amb.step(acao)

            # Selecionamos a acao para o estado atual de acordo com a estratégia passada
            # Observe que antes de fazer qualquer cálculo, precisamos obter a ação para a próxima etapa.
            prox_acao = seleciona_acao(prox_estado, Q, epsilons[e])

            # Implementação da função TD: Calculando o objetivo (Caso o estado seja terminal temos que zerar)
            objetivo_td = recompensa + gamma * Q[prox_estado][prox_acao] * (not terminado)
            # Implementação da função TD: Calculando o erro
            erro_td = objetivo_td - Q[estado][acao]
            # Implementação da função TD: Calculando a função Q
            Q[estado][acao] = Q[estado][acao] + alphas[e] * erro_td

            # Passamos para o proximo estado e proxima ação
            estado, acao = prox_estado, prox_acao

        # Guardamos a Função Q atual no histórico
        Q_historico[e] = Q
        # Guardamos a política atual no histórico
        pi_historico.append(np.argmax(Q, axis=1))

    # Extraímos a Função de Valor de Estado selecionando as melhores ações de Q
    V = np.max(Q, axis=1)
    # Com a Função de Valor Feita podemos obter uma política ótima
    #pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    pi = np.argmax(Q, axis=1)
    return Q, V, pi, Q_historico, pi_historico

# %%
# Implementação do algoritmo Q-Learning

from gymnasium.spaces import Discrete, MultiDiscrete
from gymnasium.spaces.utils import flatdim

def indice_estado(estado: int | list[int], obs_space) -> int:
    """ Converte um estado em um índice para o array Q 
        Faz a conta: 550*500*20*a + 500*20*b + 20*c + d
    """
    if isinstance(obs_space, MultiDiscrete):
        indice = estado[-1]
        fator = 1
        for i in range(len(estado)-2,-1,-1):
            fator *= obs_space.nvec[i+1]
            indice += fator*estado[i]
        return indice
    else:
        return estado

# Você passa a política o ambiente para qual o algoritmo criará a política, desconto de recompensa, especificações para a criação do array de 
# decaimento exponencial de alpha e epsilon, número de episódios considerados
def q_learning(amb, gamma=1.0, ini_alpha=0.5, min_alpha=0.01, taxa_decay_alpha=0.5, ini_epsilon=1.0, 
          min_epsilon=0.1, taxa_decay_epsilon=0.9, n_episodios=3000, verbose=False, guardar_historico=False):
    # Obtemos o número de estados do ambiente (nS) e o número de ações disponíveis (nA)
    nS = np.prod(amb.observation_space.nvec) if isinstance(amb.observation_space, MultiDiscrete) else amb.observation_space.n
    # nA = flatdim(amb.action_space) if isinstance(amb.action_space, MultiDiscrete) else amb.action_space.n
    nA = np.prod(amb.action_space.nvec) if isinstance(amb.action_space, MultiDiscrete) else amb.action_space.n
    if verbose: print(f"{nS=} {nA=}")

    # Lista de retornos
    retornos = []

    # Inicializando a função com as dimensões do ambiente analisado e os valores zerados
    Q = np.zeros((nS, nA), dtype=np.float64)
    if verbose: print(f'{Q.shape=}')
    # Array utilizado para ver o progresso da função Q a cada episodio
    if guardar_historico:
        Q_historico = np.zeros((n_episodios, nS, nA), dtype=np.float64)
        # Array utilizado para ver o progresso da avaliação da política
        pi_historico = []
    else:
        Q_historico = None
        pi_historico = None

    # Criando a estratégia de seleção de ação: Utilizamos Epsilon Ganancioso
    #     Funções anônimas (Lambda): são  funções que o usuário não precisa definir, ou seja, não vai precisar
    # escrever a função e depois utilizá-la dentro do código.
    seleciona_acao = lambda estado, Q, epsilon: \
        np.argmax(Q[indice_estado(estado,amb.observation_space)]) \
        if np.random.random() > epsilon \
        else np.random.randint(len(Q[indice_estado(estado,amb.observation_space)]))

    # Usando a função criada anteriormente para criar o decaimento exponencial de alpha 
    alphas = decaimento_exponencial(ini_alpha, min_alpha, taxa_decay_alpha, n_episodios)
    # Usando a função criada anteriormente para criar o decaimento exponencial de epsilon 
    epsilons = decaimento_exponencial(ini_epsilon, min_epsilon, taxa_decay_epsilon, n_episodios)

    # Loop principal com tqdm
    # e --> Número da iteração
    # for e in range(n_episodios):
    for e in tqdm(range(n_episodios), leave=False):
        if verbose: print(f'{e=}')
        # Resetamos o ambiente para garantir que ele esteja pronto para ser analisado
        # Obtemos o estado inicial e se o estado é terminal
        estado, terminado = amb.reset(), False
        if verbose: print(f'{terminado=}')
        # Loop secundário --> Só para se entrarmos em um estado terminal
        soma = 0
        while not terminado:            
            # Selecionamos a acao para o estado atual de acordo com a estratégia passada
            acao = seleciona_acao(estado, Q, epsilons[e])
            if verbose: print(f'ANTES: {estado=} {acao=} {Q[indice_estado(estado,amb.observation_space)][acao]=}')
            # Obtemos as informações de retorno do passo dado
            prox_estado, recompensa, terminado, _ = amb.step(acao)

            # Soma das recompensas
            soma += recompensa

            if verbose: print(f'{acao=} {prox_estado=} {recompensa=} {terminado=}')
            # Implementação da função Q-learning: Calculando o objetivo (Caso o estado seja terminal temos que zerar)
            objetivo_td = recompensa + gamma * Q[indice_estado(prox_estado,amb.observation_space)].max() * (not terminado)
            # Implementação da função Q-learning: Calculando o erro
            erro_td = objetivo_td - Q[indice_estado(estado,amb.observation_space)][acao]            
            # Implementação da função Q-learning: Calculando a Função Q
            Q[indice_estado(estado,amb.observation_space)][acao] = Q[indice_estado(estado,amb.observation_space)][acao] + alphas[e] * erro_td
            if verbose: print(f'{objetivo_td=} {erro_td=} {Q[indice_estado(estado,amb.observation_space)][acao]=}')

            # Passamos para o proximo estado
            estado = prox_estado      

        # Guardamos a Função Q e politica atuais no histórico
        if guardar_historico:
            Q_historico[e] = Q
            pi_historico.append(np.argmax(Q, axis=1))

        # Guardamos o historico de retorno
        retornos.append(soma)

    # Extraímos a Função de Valor de Estado selecionando as melhores ações de Q
    V = np.max(Q, axis=1)
    # Com a Função de Valor Feita podemos obter uma política ótima
    #pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[indice_estado(s,amb.observation_space)]
    pi = np.argmax(Q, axis=1)
    return Q, V, pi, Q_historico, pi_historico, retornos

# %%
# Implementação do algoritmo Q-Learning duplo

# Você passa a política o ambiente para qual o algoritmo criará a política, desconto de recompensa, especificações para a criação do array de 
# decaimento exponencial de alpha e epsilon, número de episódios considerados
def double_q_learning(amb, gamma=1.0, ini_alpha=0.5, min_alpha=0.01, taxa_decay_alpha=0.5, ini_epsilon=1.0,
                      min_epsilon=0.1, taxa_decay_epsilon=0.9, n_episodios=3000):
    # Obtemos o número de estados do ambiente (nS) e o número de ações disponíveis (nA)
    nS, nA = amb.observation_space.n, amb.action_space.n
    
    # Array utilizado para ver o progresso da avaliação da política
    pi_historico = []

     # Inicializando a função Q1 e Q2 com as dimensões do ambiente analisado e os valores zerados
    Q1 = np.zeros((nS, nA), dtype=np.float64)
    Q2 = np.zeros((nS, nA), dtype=np.float64)
    # Arrays utilizados para ver o progresso da funções Q1 e Q2 a cada episodio
    Q1_historico = np.zeros((n_episodios, nS, nA), dtype=np.float64)
    Q2_historico = np.zeros((n_episodios, nS, nA), dtype=np.float64)

    # Criando a estratégia de seleção de ação: Utilizamos Epsilon Ganancioso
    #     Funções anônimas (Lambda): são  funções que o usuário não precisa definir, ou seja, não vai precisar
    # escrever a função e depois utilizá-la dentro do código.
    seleciona_acao = lambda estado, Q, epsilon: \
        np.argmax(Q[estado]) \
        if np.random.random() > epsilon \
        else np.random.randint(len(Q[estado]))

    # Usando a função criada anteriormente para criar o decaimento exponencial de alpha 
    alphas = decaimento_exponencial(ini_alpha, min_alpha, taxa_decay_alpha, n_episodios)
    # Usando a função criada anteriormente para criar o decaimento exponencial de epsilon 
    epsilons = decaimento_exponencial(ini_epsilon, min_epsilon, taxa_decay_epsilon, n_episodios)

    # Loop principal com tqdm
    # e --> Número da iteração
    for e in tqdm(range(n_episodios), leave=False):
        # Resetamos o ambiente para garantir que ele esteja pronto para ser analisado
        # Obtemos o estado inicial e se o estado é terminal
        estado, terminado = amb.reset(), False

        # Loop secundário --> Só para se entrarmos em um estado terminal
        while not terminado:
            # Selecionamos a acao para o estado atual de acordo com a estratégia passada
            # Precisamos passar somente uma função Q por isso criamos um array com a média de nossas duas funções Q
            # Também poderíamos usar a soma de nossas funções Q aqui, eles darão resultados semelhantes.
            acao = seleciona_acao(estado, (Q1 + Q2)/2., epsilons[e])

            # Obtemos as informações de retorno do passo dado
            prox_estado, recompensa, terminado, _ = amb.step(acao)

            # Escolhemos aleatoriamente (50%) qual função Q usar
            if np.random.randint(2):
                # Vamos usar a ação que Q1 acha melhor
                argmax_Q1 = np.argmax(Q1[prox_estado])

                # Mas usamos o valor de Q2 para calcular o objetivo TD
                objetivo_td = recompensa + gamma * Q2[prox_estado][argmax_Q1] * (not terminado)
                # Resto da implementação da função do Q-learning: Usando Q1
                erro_td = objetivo_td - Q1[estado][acao]
                Q1[estado][acao] = Q1[estado][acao] + alphas[e] * erro_td

            else:
                # Vamos usar a ação que Q2 acha melhor
                argmax_Q2 = np.argmax(Q2[prox_estado])

                # Mas usamos o valor de Q1 para calcular o objetivo TD
                objetivo_td = recompensa + gamma * Q1[prox_estado][argmax_Q2] * (not terminado)
                # Resto da implementação da função do Q-learning: Usando Q2
                erro_td = objetivo_td - Q2[estado][acao]
                Q2[estado][acao] = Q2[estado][acao] + alphas[e] * erro_td

            # Passamos para o proximo estado
            estado = prox_estado

        # Guardamos as Funções Q1 e Q2 atuais no histórico
        Q1_historico[e] = Q1
        Q2_historico[e] = Q2
        # Guardamos a política atual no histórico, perceba que utilizaremos a avaliação média entre Q1 e Q2
        pi_historico.append(np.argmax((Q1 + Q2)/2., axis=1))

    # O Q final é média do 1 com o 2
    Q = (Q1 + Q2)/2.
    # Obtemos a Função valor estado
    V = np.max(Q, axis=1)
    # Obtemos uma política ótima
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    # Perceba que retornamos o histórico médio de Q
    return Q, V, pi, (Q1_historico + Q2_historico)/2., pi_historico

# %% [markdown]
# ## **Implementação de Ambientes customizados**

# %%
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Optional
from collections import deque

class BeerGame(gym.Env):

    # Método 'construtor'
    def __init__(self, seed : Optional[int] = None) -> None: 
       # 1-> Definindo o espaço de ações 
       self.pedido_max = 20 # Podem ser pedidos somente 20 cervejas por vez
       self.pedido_min = 0 # Não é possível realizar pedidos negativos (Isso não faz sentido!!)
       self.pedido_ini = 4 # O pedido inicial sempre é 4
       self.pedido_ant = self.pedido_ini # Essa variável 'pedido_ant' é uma variável auxiliar que guarda a informação sobre o pedido feito anteriormente (será atualizada a cada iteração)
       # Declarando espaço de ações:
       self.action_space = spaces.MultiDiscrete(np.array(4*[self.pedido_max]))
       
       # 2-> Definindo uma seed
       self.seed(seed) # Chamamos o método que cria uma seed
       
       # 3-> Definindo o espaço de estados
       self.estoque_ini = 12 # A quantidade inicial em estoque sempre é 12
       self.transporte_ini = 4 # A quantidade inicial em transporte sempre é 4
       self.capacidade_max = 50 # A capacidade máxima de estoque em qualquer um dos membros da cadeia de suprimentos
       self.capacidade_min = 0 # Não é possível realizar guardar uma quantidade de cerveja negativa (Isso não faz sentido!!)
       # Definindo o observation space: limites, tipo de espaço e forma
       self.observation_space = spaces.MultiDiscrete(np.array(12*[self.capacidade_max]+[self.pedido_max]))

       # 4-> Inicializando os arrays dos estoques, transporte e produção
       self.estoques = 4*[0] # Simples 4 vetores de 1 posição uma para cada (R, W, D, F)
       self.entrega_retailer = deque(2*[0]) # Fila com 2 posições Retailer - Transporte
       self.entrega_wholesailer = deque(2*[0]) # Fila com 2 posições Wholesaler - Transporte
       self.entrega_distributor = deque(2*[0]) # Fila com 2 posições Distributor - Transporte
       self.producao_factory = deque(2*[0]) # Fila com 2 posições Factory - Produção
       
       # 5-> Definimos os arrays com os valores iniciais correspondentes
       self._inicializar_cadeia()
       
       # 6-> Definimos os preços 
       self.custo_de_estoque = 0.5
       self.custo_de_backlog = 1
       
       # 6-> Definimos o oeríodo em que o BeerGame será rodado
       self.periodo_max = 50
       self.periodo_atual = 0 # Variável auxiliar será atualizada a cada iteração

    # Define os arrays com os valores iniciais correspondentes
    def _inicializar_cadeia(self) -> None:
       self.estoques = 4*[self.estoque_ini] # Estoques começam com 12 cervejas
       self.entrega_retailer = deque(2*[self.transporte_ini]) # Estão transportando 4 cervejas em cada slot de entrega
       self.entrega_wholesaler = deque(2*[self.transporte_ini]) # Estão transportando 4 cervejas em cada slot de entrega
       self.entrega_distributor = deque(2*[self.transporte_ini]) # Estão transportando 4 cervejas em cada slot de entrega
       self.producao_factory = deque(2*[self.transporte_ini]) # Estão produzindo 4 cervejas em cada slot de produção

    #   Esse método retorna um estado, ou seja, uma observação do estado, em que a 
    # posição 0 é estoque do Retailer e a posicao 11 é slot de producao da fábrica
    # a posição 12 é na verdade a informação sobre o pedido feito ao Retailer anteriormente
    def _montar_estado(self) -> list[int]:
       estado = 13*[0]
       estado[0] = self.estoques[0]
       estado[1] = self.entrega_retailer[0]
       estado[2] = self.entrega_retailer[1]
       estado[3] = self.estoques[1]
       estado[4] = self.entrega_wholesaler[0]
       estado[5] = self.entrega_wholesaler[1]
       estado[6] = self.estoques[2]
       estado[7] = self.entrega_distributor[0]
       estado[8] = self.entrega_distributor[1]
       estado[9] = self.estoques[3]
       estado[10] = self.producao_factory[0]
       estado[11] = self.producao_factory[1]
       estado[12] = self.pedido_ant       

       return estado

    # Esse método reseta o ambiente, reinicializando os arrays com os valores iniciais definidos no 'construtor'
    def reset(self) -> list[int]:
       self._inicializar_cadeia() # Reinicializa os arrays com os valores iniciais
       self.periodo_atual = 0 # Reinicia a contagem do período
       return np.array(self._montar_estado()) # retorna uma observação do estado inicial
    
    # Esse método executa a ação escolhida no ambiente e retorna uma observação do estado, recompensa, se o estado é terminal e uma informação
    def step(self, action : list[int]) -> tuple[list[int], float, bool, bool, dict]:
       self.periodo_atual += 1 # Cada vez que uma ação é tomada é considerado um novo período
       
       custo = 0 # Inicializamos o valor do custo
       self.pedido_ant = self.rand_generator.randint(self.pedido_max + 1) # Sorteamos um número (de 1 a 20) que será o número de cervejas solicitados pelo cliente ao Retailer
       
       # Chegaram os caminhões: Os pedidos feitos anteriormente que estavam aguardando nas filas de entrega é adicionado aos estoques
       self.estoques[0] += self.entrega_retailer.popleft()
       self.estoques[1] += self.entrega_wholesaler.popleft()
       self.estoques[2] += self.entrega_distributor.popleft()
       self.estoques[3] += self.producao_factory.popleft()
       
       # Decisão Retailer:
       if (self.estoques[0] >= self.pedido_ant):
         # O número de unidades estocadas é subtraído pelo pedido quando o estoque é maior ou igual ao pedido
         self.estoques[0] = self.estoques[0] - self.pedido_ant
       else:
         # Quando pedido é maior cobra-se um custo de backlog a cada unidade de cerveja não atendida, esse custo é armazenado como recompensa
         custo += (self.pedido_ant - self.estoques[0])*self.custo_de_backlog 
         self.estoques[0] = 0 # E o estoque é zerado
       
       # Decisão Wholesaler:
       if (self.estoques[1] >= action[0]):
         # O número de unidades estocadas é subtraído pelo pedido quando o estoque é maior ou igual ao pedido
         self.estoques[1] = self.estoques[1] - action[0]
         # Como Wholesaler não lida diretamente com cliente ele manda transportar as unidades de cerveja: o número pedido é adicionado a fila
         self.entrega_retailer.append(action[0])
       else:
         # Quando pedido é maior cobra-se um custo de backlog a cada unidade de cerveja não atendida, esse custo é armazenado como recompensa
         custo += (action[0] - self.estoques[1])*self.custo_de_backlog
         # A quantidade que tem estoque será então transportada: adiciona-se a fila
         self.entrega_retailer.append(self.estoques[1])
         self.estoques[1] = 0 # E o estoque é zerado

       # Decisão Distributor:
       if (self.estoques[2] >= action[1]):
         # O número de unidades estocadas é subtraído pelo pedido quando o estoque é maior ou igual ao pedido
         self.estoques[2] = self.estoques[2] - action[1]
         # Como Distributor não lida diretamente com cliente ele manda transportar as unidades de cerveja: o número pedido é adicionado a fila
         self.entrega_wholesaler.append(action[1])
       else:
         # Quando pedido é maior cobra-se um custo de backlog a cada unidade de cerveja não atendida, esse custo é armazenado como recompensa
         custo += (action[1] - self.estoques[2])*self.custo_de_backlog
         # A quantidade que tem estoque será então transportada: adiciona-se a fila
         self.entrega_wholesaler.append(self.estoques[2])
         self.estoques[2] = 0 # E o estoque é zerado
       
       # Decisão Factory:
       if (self.estoques[3] >= action[2]):
         # O número de unidades estocadas é subtraído pelo pedido quando o estoque é maior ou igual ao pedido
         self.estoques[3] = self.estoques[3] - action[2]
         # Como Factory não lida diretamente com cliente ele manda transportar as unidades de cerveja: o número pedido é adicionado a fila
         self.entrega_distributor.append(action[2])
       else:
         # Quando pedido é maior cobra-se um custo de backlog a cada unidade de cerveja não atendida, esse custo é armazenado como recompensa
         custo += (action[2] - self.estoques[3])*self.custo_de_backlog
         # A quantidade que tem estoque será então transportada: adiciona-se a fila
         self.entrega_distributor.append(self.estoques[3])
         self.estoques[3] = 0 # E o estoque é zerado
       
       # Da início a produção do número de unidades pedidas a fábrica
       self.producao_factory.append(action[3])
       
       # Além dos custos de Backlog é adicionado o custo de estoque por cada unidade em estoque em (R, W, D, F) -> Perceba que nessa situação a recompensa é negativa
       custo += sum(self.estoques)*self.custo_de_estoque
       # BeerGame só termina quando é atinjido o 50º período
       terminado = self.periodo_atual == self.periodo_max

       # A recompensa será o negativo do custo, pois o agente tenta maximizar a recompensa e, logo,
       # minimizará o custo
       recompensa = -custo

       return np.array(self._montar_estado()), recompensa, terminado, {} # Retorna-se a observação do estado, a recompensa, se é estado terminal  

    # Esse método imprime uma observação do estado
    def render(self, mode : str ="human") -> Any: # opcional
       print(self._montar_estado())

    # Esse método close fecha todos os recursos abertos que foram usados pelo ambiente. No nosso caso não foi preciso implementá-lo
    def close(self) -> None: # opcional
       pass
    
    #   Cria uma seed aleatória para iteração: afeta diretamente os pedidos aleatórios do suposto cliente
    # Dessa forma a seed é utilizada quando deseja repetir um certo cenário no ambiente e testá-lo de diferentes formas
    def seed(self, seed=None) -> None:
      self.rand_generator = np.random.RandomState(seed)
      self.action_space.seed(seed)

# %%
class BeerGameSimplificado(BeerGame):
  
  def __init__(self,  seed : Optional[int] = None) -> None:
        super().__init__()
        
        # Reescrevendo a variável 'action_space' para aceitar somente 1 ação:
        self.action_space = spaces.Discrete(20)

        # Reescrevendo a variável 'observation_space' para ter visão somente do retailer
        self.observation_space = spaces.MultiDiscrete(np.array(3*[self.capacidade_max]+[self.pedido_max]))

  def step(self, action : int) -> tuple[list[int], float, bool, bool, dict]:
        # Recebe uma única ação e cria uma lista com a ação replicada para então chamar o método da Classe Pai
        acao = [action, action, action, action]
        
        return super().step(acao)

  def _montar_estado(self) -> list[int]:
    estado = super()._montar_estado()
    
    # Retorna apenas as informações do retailer e a demanda do cliente
    estado = estado[:3] + [estado[-1]]

    # Usa np.clip para garantir que o estado não passa dos valores máximos do observation_space
    estado = np.clip(estado, 0, self.observation_space.nvec-1)
    
    return estado

# %% [markdown]
# ![Radiação.jpg](data:image/jpeg;base64,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Xl5MamGYRzfEqrtinPM3VsznmsWt/LOP2cwjUUAm5JkoLrEKq2vYlkzmjg8y5KqKeta1NGWb1trNzVh3dzfMnq8IuO+5yhV/DjMRyF3snzDDcY4xsD2/b7lcOQNIUsBUTzN/km3WJVXF3GH3aamZKBANg88kq1b2IUorTTjQIZ1j6g7mkshJacQyqvm3dDp56TRs3N3XzjK47+RcB1VKLUPwdZgVwjyOu7KKKgWSX0faBAaHS6ez42mueay53JNtFGpmROaImzwLgOqS7RP3mPQ/CGG18b0oiUVFJobTZjJGGM/mXIYXA/3fo5jm7y4Z+Y41/voFHsQCFNzX8sYJzZTb9N+D1KAqMz2FwBO8aCVNP+MJgZy39tZWiHqWs9+C/OKsSjKbkK65CsB0MzA+IiEDFZaFh8K8LXQ7L6JZ/wC58W3b/iFLQHsZypUDmebKCFp1W+8ooF4ThAGoMruESvUi7kwkJJJbZoO0SRTbG3copRrbs4DrtmGjtY6U5xMc4orOZlWNLT1I5GXC1CIfczofcGlHV6j/AJBKWyioytFGmskEouEYxmirMU5Mtx2rrlCu6XunZhlpyTFn7J/GJgD9s7oVBSdPD8wB2LKu8ppdKVojOilu+2UtyHCZSn+h9tU7DiRJVK4Vpc794G7FCxThOvyKTGr1npYK07uV9R8FVgNdm13tWZCNK3rHAf6PIp3h4fyI7f8AmP/aAAwDAQACAAMAAAAQ88888888848s888888888888aZpNeHO2488888888UEaE/8A/wD+6vmrzzzzzyp8sP8A/wD/AP8A/NHe688884cxks81/wD/AP1yw1BjzzwxaLzyjP8A/wDf/PPHKvPOLTlPPOOvvOEPPPFfHPLO8fPPHMpbdVfPPKFYfOM7PPPMrXPFGVPPLAXvONDzDj3PfPOJTjjDz1vFF/8A/wD/APibNOff/wD/APqe8s2//wD/AP8A9fd+/wD/AP8A/q288M+//wD/APEywTT/AP8A+WVPPOBfP/79/PKE/wD/APq48888wU099c888s1/0Fc88888yBm14w8ke76t88888888oSMxB3yzxIc888888888sccee+uM8888888888888888888888888//EACIRAQACAgIDAAIDAAAAAAAAAAEAESAhMWEQQTBxUUBQkf/aAAgBAwEBPxD+TwBF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)
# ## **Área de Testes**
# 
# ### **Como debugar no Google Colab?**
# * https://stackoverflow.com/questions/52656692/debugging-in-google-colabo-google-colab-94582
# 
# Command  Description
# 
# * list     Show the current location in the file
# * h(elp)   Show a list of commands, or find help on a specific command
# * q(uit)   Quit the debugger and the program
# * c(ontinue)  Quit the debugger, continue in the program
# * n(ext)   Go to the next step of the program
# * <enter>  Repeat the previous command
# * p(rint)  Print variables
# * s(tep)   Step into a subroutine
# * r(eturn)    Return out of a subroutine
# 
# ### **Pixie_debugger -> Problema com versão do jinja2**

# %%

cores = ["r", "g", "b", "c", "m", "y", "k", "w"]

# FUNÇÕES ÚTEIS:

# Função para gerar um gráfico da Curva de Aprendizado
def geraCurvaDeAprendizado(retornos, multiplo):
    if multiplo:
        tamanho = len(retornos)
        maior = 0
        for i in range(tamanho):
            eps = len(retornos[i])
            if eps > maior:
                maior = eps

        episodios = np.arange(1, tamanho+1, 1)

        plt.figure(figsize=(10,5))
        for i in range(tamanho):
            plt.plot(episodios, retornos[i], label = 'NYA', color = cores[i], lw = 2)
        plt.title('Curva de Aprendizado')
        plt.ylabel('Retornos')
        plt.xlabel('Episódios')
        plt.show()

    else:
        tamanho = len(retornos)
        episodios = np.arange(1, tamanho+1, 1)

        plt.figure(figsize=(10,5))
        plt.plot(episodios, retornos, label = 'NYA', color = 'g', lw = 2)
        plt.title('Curva de Aprendizado')
        plt.ylabel('Retornos')
        plt.xlabel('Episódios')
        plt.show()

# Função de avaliação de política
def avalia_politica(ambiente_par, politica, n_episodios):
    
    # Se foi passada uma string, assume-se que é um ambiente gym
    if isinstance(ambiente_par, str):
        ambiente_valid = gym.make(ambiente, render_mode="rgb_array")
        ambiente = ambiente_valid.env.P
    else: # caso contrário, assume-se que foi passado o MDP diretamete
        ambiente_valid = None
        ambiente = ambiente_par
    
    if ambiente:

        retornos = []

        # for episodio in range(n_episodios):
        for episodio in tqdm(range(n_episodios), leave=False):
            estado = ambiente.reset()            
            terminado = False
            retorno = 0
            while not terminado:
                acao = politica(estado)
                estado, recompensa, terminado, _ = ambiente.step(acao)
                retorno += recompensa            
            retornos.append(retorno)
        
        media_recompensa = np.sum(retornos)/len(retornos)

        return media_recompensa, retornos
    else:
        print("ATENÇÃO: o ambiente não foi configurado corretamente!")
        return -1, []

# Função que salva os retornos em um arquivo .npy
def salvaRetornos(retornos):
    if (len(retornos) > 0):
        print("Digite seu o nome do arquivo a ser salvo\nLembre-se não digite a extensão do arquivo (exemplo: .txt)\nNome:")
        nome = input()
        retornos = np.array(retornos)
        np.save(nome, retornos)
        print("O array foi salvo no arquivo " + nome + ".npy") 
    else:
        print("O array retornos está vazio")

# Função que carrega os retornos em um array para sua utilização
def carregaRetornos():
    print("Digite seu o nome do arquivo a ser caregado\nLembre-se é necessário que se digite a extensão do arquivo (exemplo: .txt)\nNome:")
    nome = input() + ".npy"
    retornos = np.load(nome)
    print("O arquivo " + nome + ".npy foi carregado com sucesso!")
    return retornos, nome

def comparaAvaliacoes():
    print("Digite a quantidade de arquivos a serem comparados")
    n = input()
    avaliacoes = []
    for i in range(n):
        retorno, nome = carregaRetornos()
        print("Avaliando dados de " + nome + ".npy...")
        media, retorno = avalia_politica(beer_game, pi, n_episodios = 5)
        print(nome + ".npy média de: " + media)
        avaliacoes.append(retorno)
    
    geraCurvaDeAprendizado(avaliacoes, True)

# AMBIENTE DE TESTES

# Variáveis bolleanas importantes:
usa_arquivo = False

# Testes realizados:
#beer_game : BeerGameSimplificado = BeerGameSimplificado(seed=10)
#estado : list[int] = beer_game.reset()

FrozenLake = old_gym.make('FrozenLake-v1')
estado = FrozenLake.reset()
print('Ambiente Configurado\n')
print(f'Estado inicial {estado}')

print('\n\n')
if (usa_arquivo):
    retornos, _ = carregaRetornos()
else:
    Q, V, pi, Q_historico, pi_historico, retornos = q_learning(FrozenLake, n_episodios=2000)
    print(f'Q = {Q}')
    print(f'V = {V}')
    print(f'pi = {pi}')
    salvaRetornos(retornos)

print('\n\nCriando gráfico:')
geraCurvaDeAprendizado(retornos, False)

print('\n\nTestando função de avaliação:')
#politica = lambda s : pi[indice_estado(s,beer_game.observation_space)]
media, retorno = avalia_politica(FrozenLake, pi, n_episodios = 2000)
print(f'media = {media}')
print(f'retorno = {retorno}')
