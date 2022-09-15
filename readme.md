# Rodar varias vezes e plotar a linha de media

# Optimizar as 2 estrategias (A2C, PPO) do mesmo autor com a mesma arq de NN

# Fixando 2 camadas, Definir LR, n de nos por camadas, 

# Comparar as estrategias

# CartPole, LunarLander

# Extrair um grafico com linhas recompensa/passos para cada vez que foi rodado

# Optimizar ES por 8h para cada ambiente

# 1 grafico com 3 curvas

# Rodar para cada curva multiplas vezes (3-5) e tirar a media

# Se estiver demorando muito focar no LunarLander

# 200 000 passos +/-

# Comparar com A2C e PPO SEM Optimizar os clássicos

# Descrição do Experimento:

    1. Algoritmos utilizados CMA-ES (Covariance Matrix Adaptation - Evolution Strategy) comparado ao A2C/PPO
    2. Cada algoritmo será aplicado nos ambientes CartPole e LunarLander
    // 3. A métrica a ser optimizada (minimizada) é o número de passos necessários para resolver o ambiente
    // Comparar a dinamica de aprendizado
    4. Ambos algoritmos utilizam redes neurais, a arquitetura seŕa sempre entrada -> camada linear de tamanho N com ReLU -> camada linear de tamanho N com ReLU -> saida
    5. Os parametros a serem variados e optimizados serão LR ([0.0005:0.002]) e o tamanho N das 2 camadas da rede ([2, 4, 8, ..., 256])
    6. Com um algoritmo e ambiente selecionados serão feitos treinamentos em batches, cada batch contendo 100 episódios limitados a 10000 passos
    7. O treinamento será dito finalizado com sucesso caso a recompensa média dos batches seja suficiente pra resolver o ambiente onde sua fitness é o número de passos que foram necessários para chegar nesse estado
    8. O treinamento será dito finalizado com falha caso após 10 minutos não tenha resolvido o ambiente
    9. Após 20 treinamentos (de sucesso?) serão extraidos os melhores hiperparametros para cada par algoritmo-ambiente

    study (8h){
        combinação de algoritmo e ambiente, ao fim, determina os melhores hyperparametros
        trial (20x) {
            uma opção de hyperparametros aplicadas no estudo, ao fim determina qual o fit daqueles hyperparametros
            batch {
                aqui ocorre o treinamento por reforço
                episodio (100x ou até solução) {
                    episódio de um ambiente
                    passos (10000 ou até solução) {
                        passo de um ambiente
                    }
                }
            }
        }
    }
