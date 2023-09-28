import pandas as pd


class Analise_Arquivos():
    def __init__(self, csv_file):
        self.tabela = pd.read.csv(csv_file)
        self.filtrar_dados()

    def filtrar_dados(self):
        self.tabela = self.tabela.dropna()


if __name__ == '__main__':
    analise = Analise_Arquivos('clientes.csv')
