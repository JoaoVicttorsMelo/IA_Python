import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class AnaliseArquivos:
    def __init__(self, csv_file1, csv_file2):
        # Método de inicialização, carrega os dados e realiza a pré-processamento
        self.carregar_dados(csv_file1, csv_file2)

    def carregar_dados(self, csv_file1, csv_file2):
        # Carrega o arquivo CSV, remove linhas com valores ausentes e a coluna "id_cliente"
        self.tabela = pd.read_csv(csv_file1).dropna().drop(columns="id_cliente")

        # Carrega o arquivo CSV para novos clientes
        self.tabela_novo_cliente = pd.read_csv(csv_file2)

        # Aplica a codificação LabelEncoder aos atributos relevantes
        self.numerar_atributos()

    def numerar_atributos(self):
        # Aplica a codificação LabelEncoder aos atributos específicos
        codificador = LabelEncoder()
        atributos = ['profissao', 'mix_credito', 'comportamento_pagamento']

        for atributo in atributos:
            self.tabela[atributo] = codificador.fit_transform(self.tabela[atributo])

            # Para os novos clientes, use a transformação, não o ajuste
            self.tabela_novo_cliente[atributo] = codificador.transform(self.tabela_novo_cliente[atributo])

        # Exibe informações sobre o DataFrame após a codificação
        print(self.tabela.info())

    def treinar_e_avaliar_modelo(self):
        # Separa o alvo (y) dos recursos (x)
        y = self.tabela["score_credito"]
        x = self.tabela.drop(columns="score_credito")

        # Divide os dados em conjuntos de treinamento e teste
        x_treino, x_teste, y_treino, y_teste = train_test_split(x, y)

        # Cria um modelo RandomForestClassifier e o treina com os dados de treinamento
        modelo_arvore = RandomForestClassifier()
        modelo_arvore.fit(x_treino, y_treino)

        # Realiza previsões no conjunto de treinamento e exibe a acurácia
        previsao_arvore = modelo_arvore.predict(x_treino)
        print("Acurácia no treinamento:", accuracy_score(y_treino, previsao_arvore))

        # Realiza previsões para os novos clientes
        previsoes_novo_cliente = modelo_arvore.predict(self.tabela_novo_cliente)
        print("Previsões para novos clientes:", previsoes_novo_cliente)


if __name__ == '__main__':
    # Cria uma instância da classe AnaliseArquivos com os nomes dos arquivos CSV
    analise = AnaliseArquivos('clientes.csv', 'novos_clientes.csv')

    # Chama o método para treinar e avaliar o modelo
    analise.treinar_e_avaliar_modelo()
