import pandas as pd
import re
import nltk
import spacy
import sys # Para sair em caso de erro de setup inicial
import os # Adicionado para manipulação de caminhos, caso necessário
import google.generativeai as genai
import pandas as pd
import os
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import markdown
import pdfplumber
import pandas as pd
import requests
import html
import json # Adicionado para usar o json.loads
from bs4 import BeautifulSoup
import unicodedata


def buscar_e_processar_diario_oficial(data=None):
    """
    Busca o Diário Oficial de SP em formato JSON (lista de objetos),
    processa o HTML e retorna um DataFrame.
    """
    if data is None:
        data = datetime.now()

    data_formatada = data.strftime("%d%m%Y")
    url = f"https://arquip.prefeitura.sp.gov.br/Assets/diario_aberto/json/{data_formatada}.json"
    
    print(f"Buscando dados em: {url}\n")
    
    try:
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        
        # --- LÓGICA CORRIGIDA E SIMPLIFICADA ---
        # Como o JSON agora é uma lista válida, podemos usar o método .json()
        # que é mais eficiente e seguro.
        lista_de_atos = response.json()
        # ######################################

        if not isinstance(lista_de_atos, list) or not lista_de_atos:
            print("AVISO: Nenhum ato (ou uma estrutura de dados inesperada) foi encontrado na edição de hoje.")
            return None
            
        print(f"✅ Sucesso! {len(lista_de_atos)} publicações encontradas no JSON.")
        
        df_diario = pd.DataFrame(lista_de_atos)
        
        print("Tratando o conteúdo HTML para extrair texto limpo...")
        
        def limpar_html(conteudo_html):
            if not isinstance(conteudo_html, str): return ""
            texto_decodificado = html.unescape(conteudo_html)
            soup = BeautifulSoup(texto_decodificado, 'html.parser')
            return soup.get_text(separator=' ', strip=True)

        # Aplica a limpeza na coluna 'conteudo' para criar a 'texto_limpo'
        df_diario['texto_limpo'] = df_diario['conteudo'].apply(limpar_html)
        
        print("✅ Coluna 'texto_limpo' criada com sucesso.")
        
        return df_diario

    except requests.exceptions.HTTPError as http_err:
        print(f"ERRO HTTP: Não foi possível encontrar o diário para a data de hoje. O servidor retornou: {http_err}")
        return None
    except json.JSONDecodeError as json_err:
        print(f"ERRO DE JSON. O formato pode estar corrompido: {json_err}")
        return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        return None
        
# --- Execução do Código ---
if __name__ == "__main__":
    
    df_diario_completo = buscar_e_processar_diario_oficial()
    
    if df_diario_completo is not None:
        print("\n--- ANÁLISE DO DATAFRAME GERADO ---")
        print(f"Total de publicações: {len(df_diario_completo)}")
        print(f"Colunas disponíveis: {df_diario_completo.columns.tolist()}")
        
        print("\n--- Exibindo as 5 primeiras linhas do DataFrame ---")
        with pd.option_context('display.max_rows', 5, 'display.max_columns', None, 'display.width', 200):
            print(df_diario_completo.head())
            
        print("\n--- Exemplo do conteúdo da primeira publicação ---")
        if not df_diario_completo.empty:
            print(df_diario_completo.loc[0, 'texto_limpo'])

df_diario_completo = df_diario_completo.rename(columns={'veiculo': 'Veículo',
                                                        'orgao': 'Órgão',
                                                        'unidade': 'Unidade',
                                                        'serie': 'Série',
                                                        'processo': 'Processo',
                                                        'documento': 'Documento',
                                                        'link': 'Link',
                                                        'texto_limpo': 'texto'})

# --- CONFIGURAÇÃO DA API DO GEMINI ---  <------ COLOQUE AQUI!
print("\n--- Configurando API do Gemini ---")
# COLE SUA API KEY AQUI
GEMINI_API_KEY = "AIzaSyAuXz9K9hE7MA7-7A3qaRvihTmt04e-DCk" # SUBSTITUA PELA SUA API KEY REAL
try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("API Key do Gemini configurada.")
except Exception as e_apikey:
    print(f"ERRO ao configurar API Key do Gemini: {e_apikey}")
    # Considere parar o script se a API Key for essencial e falhar
    # sys.exit()

MODELO_GEMINI = 'gemini-2.5-pro-preview-05-06'
print(f"Modelo Gemini definido como: {MODELO_GEMINI}")

print(f"--- Módulo de Filtragem de Pautas do Diário Oficial ---")
print(f"Iniciado em: {pd.Timestamp.now(tz='America/Sao_Paulo').strftime('%Y-%m-%d %H:%M:%S %Z')}")
#########################################################################################################


# --- 1. CONFIGURAÇÃO INICIAL ESSENCIAL (NLP e Stopwords) ---
print("\n--- 1. Configurando recursos de NLP e Stopwords ---")
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError as e_nltk:
    print(f"AVISO: Recurso do NLTK não encontrado ('{e_nltk.resource_name}'). Tentando baixar...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("Recursos 'punkt' e 'stopwords' do NLTK baixados/verificados.")
    except Exception as e_download:
        print(f"ERRO CRÍTICO ao baixar recursos do NLTK: {e_download}. O script pode não funcionar corretamente.")

nlp_spacy = None
try:
    nlp_spacy = spacy.load('pt_core_news_sm')
    print("Modelo 'pt_core_news_sm' do spaCy carregado.")
except OSError:
    print("ALERTA: Modelo 'pt_core_news_sm' do spaCy não encontrado.")
    print("  Execute: python -m spacy download pt_core_news_sm")
    print("  Lematização com spaCy estará desabilitada. Usando NLTK Stemmer se aplicável.")
usar_spacy = nlp_spacy is not None
##############################################################################################################


stopwords_pt = []
try:
    stopwords_pt = nltk.corpus.stopwords.words('portuguese')
    stopwords_pt.extend([ # MANTENHA SUA LISTA DE STOPWORDS ATUALIZADA
        'nº', 'secretaria', 'municipal', 'prefeitura', 'sei', 'processo',
        'despacho', 'portaria', 'resolução', 'decreto', 'art', 'gabinete',
        'prefeito', 'secretário', 'publicado', 'doc', 'pg', 'considerando',
        'lei', 'federal', 'estadual', 'inciso', 'parágrafo',
        'conforme', 'interessado', 'assunto', 'exercício',
        'sfa', 'ii', 'objeto', 'np', 'cnpj', 'ltda', 'silva', 'rf', 'smc', 'data', 'iii', 'cfoc',
        'sp', 'gov', 'br', 'paulo', 'cidade', 'município', 'documento', 'artigo', 'presente', 'dia',
        'rua', 'cpf', 'real', 'nome', 'santo', 'xxx.xxx', 'anexo'
    ])
    print(f"Stopwords carregadas e personalizadas. Total: {len(stopwords_pt)}")
except Exception as e:
    print(f"ALERTA: Erro ao carregar stopwords do NLTK: {e}. A lista de stopwords pode estar incompleta.")
###########################################################################################################################

# --- 2. FUNÇÃO DE PRÉ-PROCESSAMENTO DE TEXTO ---
print("\n--- 2. Definindo função de pré-processamento ---")
def preprocessar_texto_para_filtro(texto_original, nlp_model_spacy, lista_stopwords, usar_lematizador_spacy):
    if not isinstance(texto_original, str):
        texto_original = ""
    texto_limpo = texto_original.lower()
    # Regex de limpeza
    texto_limpo = re.sub(r'\d{4}\.\d{4}\/\d{7}-\d{1}', ' ', texto_limpo)
    texto_limpo = re.sub(r'\d{3}\.\*{3}\.\*{3}-\*{2}', ' ', texto_limpo)
    texto_limpo = re.sub(r'n[ºo\.\s]*\d{1,3}\.\d{3}\/\d{4}', ' ', texto_limpo, flags=re.IGNORECASE)
    texto_limpo = re.sub(r'n[ºo\.\s]*\d{2}\.\d{3}\.\d{3}\/\d{4}-\d{2}', ' ', texto_limpo, flags=re.IGNORECASE)
    texto_limpo = re.sub(r'\d{1,2}\/\d{1,2}\/\d{4}', ' ', texto_limpo)
    texto_limpo = re.sub(r'\bxxx\b', ' ', texto_limpo, flags=re.IGNORECASE)
    # Limpeza genérica
    texto_limpo = re.sub(r'[^\w\s]', ' ', texto_limpo)
    texto_limpo = re.sub(r'\d+', ' ', texto_limpo)
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()
    tokens_processados = []
    if usar_lematizador_spacy and nlp_model_spacy:
        doc_spacy = nlp_model_spacy(texto_limpo)
        for token_spacy in doc_spacy:
            if not token_spacy.is_stop and token_spacy.lemma_.lower() not in [sw.lower() for sw in lista_stopwords] and len(token_spacy.lemma_.strip()) > 1:
                tokens_processados.append(token_spacy.lemma_.lower())
    else:
        tokens_nltk = nltk.word_tokenize(texto_limpo, language='portuguese')
        stemmer = nltk.stem.RSLPStemmer()
        for token in tokens_nltk:
            token_lower = token.lower()
            if token_lower not in [sw.lower() for sw in lista_stopwords] and len(token_lower.strip()) > 1:
                tokens_processados.append(stemmer.stem(token_lower))
    return tokens_processados

# --- 2. FUNÇÕES DE PRÉ-PROCESSAMENTO E LIMPEZA DE TEXTO ---
print("\n--- 2. Definindo funções de pré-processamento e limpeza ---")

def remover_caracteres_de_controle(texto):
    """
    Remove caracteres de controle de uma string, que podem causar erros em APIs.
    Mantém caracteres de espaçamento padrão como quebra de linha e tabulação.
    """
    if not isinstance(texto, str):
        return texto
    
    # Mantém \n (newline), \r (carriage return), \t (tab)
    caracteres_a_manter = ['\n', '\r', '\t']
    
    return "".join(
        char for char in texto 
        if unicodedata.category(char)[0] != 'C' or char in caracteres_a_manter
    )


#########################################################################################
# --- 3. CARREGAMENTO DAS REGRAS DE FILTRAGEM (Seu "PASSO 8") ---
print("\n--- 3. Carregando regras de filtragem do CSV ---")


NOME_ARQUIVO_REGRAS = "regras_atualizadas.csv"



COLUNA_TERMO_FRASE_CSV = 'Termo/Frase/Padrão'
COLUNA_TIPO_CSV = 'Tipo'
COLUNA_FORCA_CSV = 'Força Estimada (1-5)'
df_regras_filtro = pd.DataFrame()
try:
    df_regras_filtro = pd.read_csv(NOME_ARQUIVO_REGRAS, sep=';', encoding='latin1')
    print(f"Regras carregadas de '{NOME_ARQUIVO_REGRAS}' com sucesso.")
    colunas_necessarias_csv = [COLUNA_TERMO_FRASE_CSV, COLUNA_TIPO_CSV, COLUNA_FORCA_CSV]
    for col_csv in colunas_necessarias_csv:
        if col_csv not in df_regras_filtro.columns:
            col_encontrada = None
            for actual_col in df_regras_filtro.columns:
                if actual_col.lower().replace(" ", "") == col_csv.lower().replace(" ", ""):
                    col_encontrada = actual_col; break
            if col_encontrada:
                print(f"AVISO: Coluna '{col_csv}' não encontrada. Usando similar: '{col_encontrada}'.")
                df_regras_filtro.rename(columns={col_encontrada: col_csv}, inplace=True)
            else:
                raise ValueError(f"Coluna '{col_csv}' não encontrada. Colunas no CSV: {df_regras_filtro.columns.tolist()}")
    if pd.api.types.is_string_dtype(df_regras_filtro[COLUNA_TERMO_FRASE_CSV]):
        df_regras_filtro[COLUNA_TERMO_FRASE_CSV] = df_regras_filtro[COLUNA_TERMO_FRASE_CSV].astype(str).str.lower().str.strip()
    df_regras_filtro[COLUNA_FORCA_CSV] = pd.to_numeric(df_regras_filtro[COLUNA_FORCA_CSV], errors='coerce')
    df_regras_filtro.dropna(subset=[COLUNA_FORCA_CSV, COLUNA_TERMO_FRASE_CSV], inplace=True)
    df_regras_filtro = df_regras_filtro[df_regras_filtro[COLUNA_TERMO_FRASE_CSV] != '']
    if df_regras_filtro.empty: print("ALERTA: Nenhuma regra válida carregada do CSV após limpeza.")
    else: print(f"{len(df_regras_filtro)} regras válidas carregadas.")
except FileNotFoundError: print(f"ERRO CRÍTICO: Arquivo de regras '{NOME_ARQUIVO_REGRAS}' não encontrado.")
except ValueError as ve: print(f"ERRO CRÍTICO ao carregar/validar regras: {ve}.")
except Exception as e: print(f"ERRO inesperado ao carregar regras: {e}.")

##############################################################################################################


# --- 4. FUNÇÃO DE APLICAÇÃO DO FILTRO E PONTUAÇÃO ---
print("\n--- 4. Definindo função de aplicação do filtro e pontuação ---")
LIMIAR_PONTUACAO_FILTRO = 15.0 # AJUSTE CONFORME SEUS TESTES
LIMIAR_PONTUACAO_NEGATIVA = -10.0 # Limiar negativo: se a SOMA dos pontos negativos for MENOR ou IGUAL a este valor, a decisão é eliminada.

def aplicar_filtro_e_pontuar(texto_decisao_original, df_regras, limiar_pontuacao_positivo, limiar_pontuacao_negativo,
                             nlp_model_spacy_local, lista_stopwords_local, usar_lematizador_spacy_local):
    """
    Aplica regras de filtragem, considerando um limiar negativo para descarte imediato.
    """
    if df_regras.empty: return False, 0

    tokens_decisao_processados = preprocessar_texto_para_filtro(
        texto_decisao_original, nlp_model_spacy_local,
        lista_stopwords_local, usar_lematizador_spacy_local
    )
    texto_decisao_processado_str = " ".join(tokens_decisao_processados)
    
    pontuacao_total = 0.0
    pontuacao_negativa_acumulada = 0.0 # Variável para somar apenas os pontos negativos

    for _, regra_atual in df_regras.iterrows():
        termo_da_regra = str(regra_atual[COLUNA_TERMO_FRASE_CSV])
        forca_da_regra = float(regra_atual[COLUNA_FORCA_CSV])
        
        match_encontrado = False
        if ' ' in termo_da_regra:
            if termo_da_regra in texto_decisao_processado_str: match_encontrado = True
        else:
            if termo_da_regra in tokens_decisao_processados: match_encontrado = True
            
        if match_encontrado:
            # Sempre soma na pontuação total
            pontuacao_total += forca_da_regra
            
            # Se a regra for negativa, também soma na pontuação negativa acumulada
            if forca_da_regra < 0:
                pontuacao_negativa_acumulada += forca_da_regra
                
                # VERIFICAÇÃO IMEDIATA: Se a soma negativa já atingiu o limiar, descarte e pare.
                if pontuacao_negativa_acumulada <= limiar_pontuacao_negativo:
                    # Retorna False imediatamente (não passou) e a pontuação calculada até o momento.
                    return False, pontuacao_total

    # Se o loop terminar sem atingir o limiar negativo, verifique o limiar positivo final.
    passou_no_filtro = pontuacao_total >= limiar_pontuacao_positivo
    
    return passou_no_filtro, pontuacao_total

############################################################################################## 

# --- 5. APLICAÇÃO DO FILTRO EM UMA NOVA AMOSTRA DO DIÁRIO OFICIAL ---
if True: 
    print("\n\n--- 5. Aplicação do Filtro em um Novo DataFrame ---")

    if df_regras_filtro.empty:
        print("ALERTA: Nenhuma regra de filtro carregada. Não é possível aplicar o filtro.")
    else:
        # --- A. CARREGUE SEU NOVO DATAFRAME DO DIÁRIO OFICIAL (BRUTO) AQUI ---
        # Substitua pelo carregamento do seu DataFrame real.
        # Exemplo:
        # caminho_novo_diario = input("Digite o caminho para o CSV da nova amostra do Diário Oficial (bruto): ")
        # try:
        #     df_nova_amostra_bruta = pd.read_csv(caminho_novo_diario)
        # except FileNotFoundError:
        #     print(f"Arquivo '{caminho_novo_diario}' não encontrado.")
        #     df_nova_amostra_bruta = pd.DataFrame()
        
        # Para demonstração, vamos criar um DataFrame de exemplo:
    
        df_nova_amostra_bruta = df_diario_completo
        # !!! AJUSTE O NOME DA COLUNA QUE CONTÉM O TEXTO PRINCIPAL DA DECISÃO !!!
        COLUNA_TEXTO_PRINCIPAL_NOVA_AMOSTRA = 'texto'
        # !!! AJUSTE O NOME DA COLUNA 'Veículo' E O VALOR A SER DESCARTADO !!!
        COLUNA_VEICULO = 'Veículo'
        VALOR_VEICULO_A_DESCARTAR = 'Servidores'
        # !!! AJUSTE O NOME DA COLUNA 'Órgão' E O VALOR PARA INCLUSÃO OBRIGATÓRIA !!!
        COLUNA_ORGAO_PARAMETRO = 'Órgão' # Renomeado para não confundir com nome de variável
        VALOR_ORGAO_INCLUSAO_OBRIGATORIA = 'Gabinete do Prefeito'
        
        print(f"DataFrame de nova amostra (exemplo) carregado com {len(df_nova_amostra_bruta)} publicações.")
        # -------------------------------------------------------------------

        if not df_nova_amostra_bruta.empty and COLUNA_TEXTO_PRINCIPAL_NOVA_AMOSTRA in df_nova_amostra_bruta.columns:
            print(f"\nAplicando pré-filtros e filtro de pontuação à coluna '{COLUNA_TEXTO_PRINCIPAL_NOVA_AMOSTRA}'...")
            print(f"Usando Limiar de Pontuação: {LIMIAR_PONTUACAO_FILTRO}")

            # --- B. APLICAR NOVAS REGRAS DE FILTRAGEM ---

            # B.1. Descartar linhas com 'Veículo' == 'Servidores'
            df_apos_descarte_veiculo = df_nova_amostra_bruta.copy() # Começar com uma cópia
            if COLUNA_VEICULO in df_apos_descarte_veiculo.columns:
                total_antes_descarte = len(df_apos_descarte_veiculo)
                df_apos_descarte_veiculo = df_apos_descarte_veiculo[df_apos_descarte_veiculo[COLUNA_VEICULO] != VALOR_VEICULO_A_DESCARTAR]
                descartados_veiculo = total_antes_descarte - len(df_apos_descarte_veiculo)
                print(f"  {descartados_veiculo} publicações descartadas (Veículo='{VALOR_VEICULO_A_DESCARTAR}'). Restam: {len(df_apos_descarte_veiculo)}")
            else:
                print(f"  AVISO: Coluna '{COLUNA_VEICULO}' não encontrada. Nenhum descarte por veículo realizado.")

            # B.2. Separar linhas de 'Gabinete do Prefeito' para inclusão obrigatória
            df_inclusao_obrigatoria = pd.DataFrame()
            df_para_filtro_pontuacao = df_apos_descarte_veiculo.copy() # O que sobrou após descarte por veículo

            if COLUNA_ORGAO_PARAMETRO in df_apos_descarte_veiculo.columns:
                filtro_gabinete = df_apos_descarte_veiculo[COLUNA_ORGAO_PARAMETRO] == VALOR_ORGAO_INCLUSAO_OBRIGATORIA
                df_inclusao_obrigatoria = df_apos_descarte_veiculo[filtro_gabinete].copy()
                # Adicionar colunas de pontuação e status para consistência, mesmo que não pontuadas
                df_inclusao_obrigatoria['passou_no_filtro'] = True
                df_inclusao_obrigatoria['pontuacao_obtida'] = float('inf') # ou um valor alto como 9999
                
                df_para_filtro_pontuacao = df_apos_descarte_veiculo[~filtro_gabinete].copy()
                print(f"  {len(df_inclusao_obrigatoria)} publicações do '{VALOR_ORGAO_INCLUSAO_OBRIGATORIA}' separadas para inclusão obrigatória.")
                print(f"  {len(df_para_filtro_pontuacao)} publicações restantes para filtro de pontuação.")
            else:
                print(f"  AVISO: Coluna '{COLUNA_ORGAO_PARAMETRO}' não encontrada. Nenhuma inclusão obrigatória por órgão realizada.")


            # B.3. Aplicar filtro de pontuação ao restante
            df_resultados_pontuacao_temp = pd.DataFrame()
            if not df_para_filtro_pontuacao.empty:
                resultados_pontuacao = []
                for texto_original_pub in df_para_filtro_pontuacao[COLUNA_TEXTO_PRINCIPAL_NOVA_AMOSTRA]:
                    passou, pontuacao = aplicar_filtro_e_pontuar(
                            texto_original_pub,
                            df_regras_filtro,
                            LIMIAR_PONTUACAO_FILTRO, # Passando o limiar positivo
                            LIMIAR_PONTUACAO_NEGATIVA, # Passando o novo limiar negativo
                            nlp_spacy,
                            stopwords_pt,
                            usar_spacy
                        )
                    resultados_pontuacao.append({'passou_no_filtro': passou, 'pontuacao_obtida': pontuacao})
                
                if resultados_pontuacao: # Adiciona colunas somente se houve resultados
                    df_resultados_pontuacao_temp = pd.DataFrame(resultados_pontuacao, index=df_para_filtro_pontuacao.index)
                    df_para_filtro_pontuacao = pd.concat([df_para_filtro_pontuacao, df_resultados_pontuacao_temp], axis=1)
                
                df_passaram_na_pontuacao = df_para_filtro_pontuacao[df_para_filtro_pontuacao['passou_no_filtro'] == True]
                print(f"  {len(df_passaram_na_pontuacao)} publicações passaram no filtro de pontuação.")
            else:
                print("  Nenhuma publicação restante para aplicar o filtro de pontuação.")
                df_passaram_na_pontuacao = pd.DataFrame()


            # B.4. Combinar os resultados: inclusão obrigatória + os que passaram na pontuação
            # Certificar que as colunas 'passou_no_filtro' e 'pontuacao_obtida' existem em df_passaram_na_pontuacao
            # antes de tentar concatenar, para evitar erros se df_passaram_na_pontuacao estiver vazio sem essas colunas.
            if 'passou_no_filtro' not in df_passaram_na_pontuacao.columns and not df_passaram_na_pontuacao.empty:
                 # Se df_passaram_na_pontuacao não estiver vazio mas não tiver as colunas, algo deu errado antes.
                 # Para segurança, podemos adicionar colunas vazias ou com valores padrão se for o caso,
                 # mas o ideal é que elas já existam. No nosso fluxo, elas são criadas.
                 print("ALERTA: df_passaram_na_pontuacao não tem as colunas de filtro. Isso não deveria acontecer.")


            # Concatenar os DataFrames. Se um estiver vazio, o concat ainda funciona.
            df_file_final_para_gemini = pd.concat([df_inclusao_obrigatoria, df_passaram_na_pontuacao], ignore_index=True)
            
            # Remover duplicatas caso um item do Gabinete do Prefeito também passasse na pontuação e fosse processado duas vezes
            # (o fluxo atual evita isso, mas é uma boa prática se os fluxos fossem diferentes)
            # Se você tiver uma coluna de ID única, use-a: df_file_final_para_gemini.drop_duplicates(subset=['id_unico_da_publicacao'], keep='first', inplace=True)
            # Se não, e os textos são o identificador:
            if COLUNA_TEXTO_PRINCIPAL_NOVA_AMOSTRA in df_file_final_para_gemini.columns:
                 df_file_final_para_gemini.drop_duplicates(subset=[COLUNA_TEXTO_PRINCIPAL_NOVA_AMOSTRA], keep='first', inplace=True)


            print("\n--- Resultados Finais do Filtro na Nova Amostra ---")
            if not df_file_final_para_gemini.empty:
                print(f"Total de publicações selecionadas para o Gemini (filé final): {len(df_file_final_para_gemini)}")
                print("Exemplos de publicações selecionadas:")
                for _, row_selecionada in df_file_final_para_gemini.head().iterrows():
                    tipo_selecao = "Obrigatória (Gabinete)" if row_selecionada.get(COLUNA_ORGAO_PARAMETRO) == VALOR_ORGAO_INCLUSAO_OBRIGATORIA else "Pontuação"
                    pont_obtida = row_selecionada.get('pontuacao_obtida', 'N/A')
                    if isinstance(pont_obtida, float) and pont_obtida != float('inf'):
                        pont_formatada = f"{pont_obtida:.2f}"
                    else:
                        pont_formatada = str(pont_obtida)

                    print(f"  Seleção: {tipo_selecao}, Pontuação: {pont_formatada} - Texto: '{str(row_selecionada[COLUNA_TEXTO_PRINCIPAL_NOVA_AMOSTRA])[:100]}...'")

                    print("\n--- Preparando Texto Combinado para Gemini ---")
            if not df_file_final_para_gemini.empty:

                textos_gabinete_lista = []
                textos_outras_lista = []

                # Separar os textos com base na coluna do órgão
                for index, row in df_file_final_para_gemini.iterrows():
                    texto_atual = str(row[COLUNA_TEXTO_PRINCIPAL_NOVA_AMOSTRA])
                    # Verifica se a coluna Órgão existe e se o valor corresponde
                    if COLUNA_ORGAO_PARAMETRO in row and row[COLUNA_ORGAO_PARAMETRO] == VALOR_ORGAO_INCLUSAO_OBRIGATORIA:
                        textos_gabinete_lista.append(texto_atual)
                    else:
                        # Todos os outros (que passaram na pontuação) vão para esta lista
                        textos_outras_lista.append(texto_atual)

                # Montar a string final com os headers solicitados
                partes_do_contexto = []

                if textos_gabinete_lista:
                    partes_do_contexto.append("--- Textos Gabinete do Prefeito (Análise Obrigatória) ---")
                    # Numerar e adicionar cada texto do gabinete
                    for i, texto_gab in enumerate(textos_gabinete_lista):
                        partes_do_contexto.append(f"GABINETE {i+1}:\n{texto_gab}")
                    # Adiciona um separador visual no final da seção (opcional)
                    partes_do_contexto.append("--- Fim Textos Gabinete ---")

                if textos_outras_lista:
                    # Adiciona espaço extra se ambas as seções existirem
                    if textos_gabinete_lista:
                        partes_do_contexto.append("\n") # Adiciona uma linha extra de espaço

                    partes_do_contexto.append("--- Outras Decisões (Selecionadas pelo Filtro) ---")
                    # Numerar e adicionar cada uma das outras decisões
                    for i, texto_outras in enumerate(textos_outras_lista):
                        partes_do_contexto.append(f"OUTRAS {i+1}:\n{texto_outras}")
                     # Adiciona um separador visual no final da seção (opcional)
                    partes_do_contexto.append("--- Fim Outras Decisões ---")

                # Junta todas as partes em uma única string, separadas por duas quebras de linha
                contexto_final_formatado_para_gemini = "\n\n".join(partes_do_contexto)
                # ================================================================= #
                # ### CORREÇÃO APLICADA EXATAMENTE AQUI ###
                # ================================================================= #
                print("\nLimpando texto final para remover caracteres de controle problemáticos...")
                # Aplica a função de limpeza na variável final
                contexto_pronto_para_api = remover_caracteres_de_controle(contexto_final_formatado_para_gemini)
        
                # Opcional: verifica se algo foi removido
                if len(contexto_pronto_para_api) < len(contexto_final_formatado_para_gemini):
                    removidos = len(contexto_final_formatado_para_gemini) - len(contexto_pronto_para_api)
                    print(f"✅ Limpeza concluída. {removidos} caracteres de controle foram removidos.")
                else:
                    print("✅ Limpeza concluída. Nenhum caractere de controle problemático encontrado.")
        # ================================================================= #

                print("\n--- Contexto Formatado e Limpo para Enviar ao Gemini (Prévia) ---")
        # Note que agora estamos imprimindo a prévia da variável limpa
                print(contexto_pronto_para_api[:2000] + ("..." if len(contexto_pronto_para_api) > 2000 else ""))
                print("--- Fim da Prévia ---")

                print(f"\nA variável 'contexto_pronto_para_api' está pronta.")
                print("Use esta nova variável no seu prompt para a chamada da API do Gemini.")

                print("\n--- Contexto Formatado para Enviar ao Gemini (Prévia dos primeiros 2000 caracteres) ---")
                print(contexto_final_formatado_para_gemini[:2000] + ("..." if len(contexto_final_formatado_para_gemini) > 2000 else ""))
                print("--- Fim da Prévia ---")

                print(f"\nA variável 'contexto_final_formatado_para_gemini' está pronta.")
                print("Use esta variável no seu prompt para a chamada da API do Gemini.")

                # ------------------------------------------------------------------
                # PRÓXIMO PASSO (Fora deste bloco, mas para você fazer a seguir):
                # 1. Definir seu PROMPT_TEMPLATE final, garantindo que ele instrua
                #    a IA sobre como tratar os textos de "Gabinete" e "Outras".
                # 2. Formatar o prompt:
                #    prompt_para_api = PROMPT_TEMPLATE.format(contexto_combinado=contexto_final_formatado_para_gemini)
                # 3. Contar tokens (opcional, mas recomendado):
                #    contagem = modelo.count_tokens(prompt_para_api)
                #    print(f"Tokens: {contagem.total_tokens}")
                # 4. Chamar a API:
                #    resposta = modelo.generate_content(prompt_para_api)
                # 5. Imprimir a resposta:
                #    print(resposta.text)
                # ------------------------------------------------------------------

            else:
                print("DataFrame final 'df_file_final_para_gemini' está vazio. Nada a formatar para o Gemini.")        

        elif df_nova_amostra_bruta.empty:
            print("O DataFrame da nova amostra (bruto) está vazio.")
        else:
            print(f"Coluna '{COLUNA_TEXTO_PRINCIPAL_NOVA_AMOSTRA}' não encontrada no DataFrame da nova amostra (bruto).")
#############################################################################################################################

        # === PASSO 6: CHAMAR GEMINI E ENVIAR RESULTADO POR E-MAIL ===

print("\n--- 6. Chamando Gemini e Enviando Resultado por E-mail ---")



# Prompt Template (com instruções sobre headers obrigatórios)
# (Usando o mesmo que definimos antes)
PROMPT_FINAL_PAUTAS = """
Você é um editor-chefe experiente e altamente seletivo da editoria de "Cidades" ou "Política" de um dos maiores veículos de imprensa do Brasil (como g1, Estadão, Folha de S.Paulo ou Brazil Journal). Sua especialidade é identificar no Diário Oficial de São Paulo pautas de ALTO IMPACTO, com potencial de grande repercussão e que revelem aspectos POSITIVOS e SIGNIFICATIVOS da gestão municipal para os cidadãos. Você tem um olhar crítico e descarta rapidamente o que é apenas burocracia ou de interesse restrito.

Sua tarefa é dividida em duas partes:

**TAREFA 1: INFORMATIVO INTERNO ESTRATÉGICO (Gabinete do Prefeito)**
A partir dos textos que tiverem com o header "Textos Gabinete do Prefeito (Análise Obrigatória)" no cabeçalho, sua função é diferente: prepare um **informativo interno conciso** para a equipe de comunicação da Prefeitura. 
Destaque as principais decisões, sejam elas percebidos como positivos ou negativos externamente. O objetivo aqui é o conhecimento estratégico interno. Podem interessar: abertura de créditos adicionais, nomeações/exonerações de alto escalão (secretários, presidentes de autarquias), reestruturações em secretarias, vetos, sanções de leis importantes e outras decisões de alto nível que o núcleo da comunicação precisa estar ciente. Seja direto e objetivo.
Coloque como intertítulo dessa parte "INFORMATIVO INTERNO - SECRETARIA DE COMUNICAÇÃO". 

**TAREFA 2: GARIMPAGEM DE PAUTAS DE ALTO IMPACTO (Outras Decisões)**
Para os textos restantes (aqueles com o header "Outras Decisões (Selecionadas pelo Filtro)"), sua missão é identificar NO MÁXIMO 10  pautas, que devem:
    a. Mostrar APENAS aspectos positivos e concretos do trabalho da gestão.
    b. Ter potencial para virar notícia de impacto em grandes veículos de imprensa.
    c. Sejam interessantes para um público amplo; ou tenham relevância e atendam a critérios de noticiabilidade. 

A SUA RESPOSTA VAI SER DIRETAMENTE ENVIADA POR E-MAIL POR COLEGAS, ENTÃO NÃO COLOQUE COISAS COMO "[NOME DO EDITOR-CHEFE]",
ESPAÇOS PARA EU PREENCHER, COMO SE EU -UMA PESSOA- ESTIVESSE FINGINDO QUE ESCREVI O SEU OUTPUT.  


**Para CADA PAUTA selecionada na TAREFA 2, fornecer:**
    - Um **Título Jornalístico:** Curto e direto.
    - A **Fonte Exata:** Número do despacho/portaria, nome da secretaria, etc., conforme presente no texto original.
    - Uma **Explicação Curta e Precisa sobre o que exatamente o trecho do Diário Oficial prevê. 

**Critérios para PRIORIZAR pautas na TAREFA 2 (busque por excelência e impacto):**
- **NOVIDADE E RELEVÂNCIA ESTRATÉGICA:** Lançamento de novos programas públicos transformadores, projetos inovadores com soluções criativas para problemas urbanos, ou investimentos substanciais em programas existentes de grande alcance.
- **GRANDE IMPACTO FINANCEIRO E SOCIAL EM PROJETOS ESTRUTURANTES:** Editais de concorrência, licitações e contratos de cifras realmente elevadas (milhões de reais) destinados a projetos que mudam a paisagem da cidade ou a qualidade de vida de cidadãos (novas grandes escolas, hospitais de referência, obras de mobilidade urbana de grande porte, programas sociais de larga escala, soluções ambientais significativas).
- **MELHORIA SENSÍVEL EM SERVIÇOS PÚBLICOS ESSENCIAIS:** Decisões que resultem em melhorias claras, perceptíveis e de grande escala em transporte, segurança pública, habitação/moradia, ou assistência social.
- **PLANEJAMENTO ESTRATÉGICO DE ALTO NÍVEL COM RESULTADOS VISÍVEIS A CAMINHO:** Planos plurianuais, programas de metas ou resultados de consultas públicas que definam rumos importantes e com potencial de transformação para a cidade, *especialmente se já houver marcos ou primeiros resultados a serem destacados*.
- **IMPACTO EM ÁREAS EMBLEMÁTICAS OU PARA GRANDE NÚMERO DE PESSOAS:** Ações em locais simbólicos, avenidas famosas e/ou monumentos  da cidade ou que afetem positivamente um número muito grande de munícipes.
- **PLANEJAMENTO ESTRATÉGICO DE ALTO NÍVEL COM RESULTADOS VISÍVEIS A CAMINHO:** Criações de grupos de trabalho, atas de reunião com planejamentos e estudos de viabilidade.


**Para a TAREFA 2, seja SELETIVO. Analise criticamente o impacto real e a novidade. Na dúvida, se não for claramente uma pauta de grande alcance e impacto positivo, NÃO inclua. A qualidade e o potencial de repercussão são mais importantes que a quantidade.**

**Critérios para ELIMINAR pautas na TAREFA 2 (seja rigoroso):**
- Qualquer decisão relacionada à TAREFA 1 (Gabinete do Prefeito) já  tratada.
- Nomeações (exceto cargos de altíssimo escalão como Secretários, que seriam da Tarefa 1), exonerações de baixo/médio escalão, aposentadorias, progressões de carreira, férias, licenças, homenagens, multas individuais, advertências, eventos internos ou de rotina administrativa.
- Contratação de oficineiros para escolas.
- Questões tributárias rotineiras, pequenas isenções ou imunidades que não representem uma mudança de política fiscal de grande impacto.
- Aprovação de regimentos internos, criação de comissões internas com escopo limitado, credenciamentos rotineiros, atas de reuniões sem deliberações de grande impacto externo.
- Contratos ou licitações de baixo valor, ou para aquisição de bens/serviços para manutenção burocrática e funcionamento interno da máquina pública (ex: compra de material de escritório, serviços de limpeza rotineiros, manutenção predial comum, comissões para compras pequenas como uniformes, medicamentos para postos específicos sem ser um grande programa, etc.). Foque no que é investimento estratégico e de grande escala.
- Pequenas revitalizações de espaços públicos pouco conhecidos ou de impacto local muito restrito.
- Concessão ou renovação de alvarás comerciais ou termos de permissão de uso rotineiros e individuais.
- Qualquer item que seja apenas "business as usual", sem um claro gancho de novidade, relevância e impacto positivo para o cidadão em larga escala.

Coloque como intertítulo dessa parte "Pautas Positivas do Diário Oficial". 



Envie esta mensagem como e-mail final, não coloque um campo a ser preenchido com o meu nome. 
--- OS TEXTOS DO DIÁRIO OFICIAL PARA ANÁLISE ESTÃO ABAIXO ---

{contexto_combinado}
"""
###################################################################################################################################################

# --- Configuração do E-mail (Como você forneceu) ---
EMAIL_REMETENTE = "gcroquer@PREFEITURA.SP.GOV.BR"
EMAIL_SENHA = "Poneialado2499!" #"Prodam10"  # !! ATENÇÃO COM A SENHA NO CÓDIGO !!
EMAIL_DESTINATARIO = ["gabriel.croquer@fsbcomunicacao.com.br",
                      "vitor.marques@fsbcomunicacao.com.br",
                      "jacqueline.matoso@fsbcomunicacao.com.br",
                      "vital.neto@fsbcomunicacao.com.br",
                      "newton.palma@fsb.com.br",
                      "caroline.valente@fsbcomunicacao.com.br",
                      "brsouza@prefeitura.sp.gov.br",
                      "acsouza@prefeitura.sp.gov.br",
                      "fbonadirman@prefeitura.sp.gov.br",
                      "jessica.mendes@fsbcomunicacao.com.br",
                      "silvia.amorim@fsbcomunicacao.com.br"]
# ])
SMTP_SERVIDOR = "smtp.office365.com"
SMTP_PORTA = 587

################################################################################################################################################

# --- Sua função de envio de e-mail ---
# (Certifique-se que as bibliotecas smtplib, email.mime... e markdown estão importadas no início do script)
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import markdown
from datetime import datetime # Adicionado para usar datetime.now() aqui

##############################################################################################################################################

def send_email(subject, body):
    """
    Envia um e-mail com um assunto e corpo definidos para múltiplos destinatários.
    Converte a formatação Markdown para HTML.
    """
    msg = MIMEMultipart('alternative')
    msg['From'] = EMAIL_REMETENTE
    msg['To'] = ", ".join(EMAIL_DESTINATARIO)
    msg['Subject'] = subject

    # Versão em texto plano como fallback
    text_part = MIMEText(body, 'plain', 'utf-8')

    # Conversão de Markdown para HTML usando a biblioteca markdown
    # Adicionando extensões para melhor formatação (como quebras de linha automáticas)
    html_content = f"""
    <html>
      <head>
        <style>
          body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
          h1 {{ color: #333366; }}
          h2 {{ color: #333366; margin-top: 20px; }}
          strong {{ font-weight: bold; }}
          pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; font-family: monospace;}}
          ul {{ margin-top: 0; padding-left: 20px;}}
          li {{ margin-bottom: 5px; }}
        </style>
      </head>
      <body>
        {markdown.markdown(body, extensions=['nl2br', 'fenced_code'])}
      </body>
    </html>
    """
    html_part = MIMEText(html_content, 'html', 'utf-8')

    msg.attach(text_part)
    msg.attach(html_part)

    try:
        server = smtplib.SMTP(SMTP_SERVIDOR, SMTP_PORTA)
        server.starttls()
        server.login(EMAIL_REMETENTE, EMAIL_SENHA)
        server.sendmail(EMAIL_REMETENTE, EMAIL_DESTINATARIO, msg.as_string())
        server.quit()
        print("✅ E-mail enviado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao enviar e-mail: {e}")
        # Para debug mais detalhado:
        # import traceback
        # print(traceback.format_exc())
#############################################################################################################################################

# --- LÓGICA PRINCIPAL DESTE BLOCO ---

# Verifica se a variável de contexto existe e não está vazia
if 'contexto_final_formatado_para_gemini' in locals() and isinstance(contexto_final_formatado_para_gemini, str) and contexto_final_formatado_para_gemini.strip():

    prompt_para_api = PROMPT_FINAL_PAUTAS.format(contexto_combinado=contexto_final_formatado_para_gemini)

    resposta_gemini_texto = "ERRO: Análise do Gemini não foi executada ou falhou." # Valor padrão

    # Chama Gemini
    try:
        modelo = genai.GenerativeModel(MODELO_GEMINI)

        print(f"\nContando tokens para o modelo {MODELO_GEMINI}...")
        contagem_tokens = modelo.count_tokens(prompt_para_api)
        print(f"Total de tokens a serem enviados: {contagem_tokens.total_tokens}")

        # Adicionar uma checagem simples para tokens muito altos
        if contagem_tokens.total_tokens > 950000: # Exemplo de limite
             print(f"AVISO: Contagem de tokens ({contagem_tokens.total_tokens}) está MUITO ALTA!")
             # Você pode adicionar uma confirmação aqui ou parar se necessário

        print(f"Enviando prompt ao modelo {MODELO_GEMINI}...")
        resposta_gemini = modelo.generate_content(prompt_para_api)

        print("\n--- Resposta do Gemini recebida (imprimindo prévia) ---")
        # Tenta extrair o texto da resposta
        if hasattr(resposta_gemini, 'text'):
            resposta_gemini_texto = resposta_gemini.text
        elif hasattr(resposta_gemini, 'parts') and resposta_gemini.parts:
            resposta_gemini_texto = "".join([part.text for part in resposta_gemini.parts if hasattr(part, 'text')])
        else:
            resposta_gemini_texto = f"ERRO: Formato de resposta inesperado.\n{resposta_gemini}"

        print(resposta_gemini_texto[:1500] + ("..." if len(resposta_gemini_texto) > 1500 else ""))
        print("--- Fim da prévia ---")

    except Exception as e:
        print(f"ERRO AO CHAMAR A API DO GEMINI: {str(e)}")
        resposta_gemini_texto = f"ERRO NA CHAMADA DA API DO GEMINI: {str(e)}"


    # Enviar resultado por e-mail
    print("\n--- Enviando resultado por e-mail ---")
    data_hoje = datetime.now().strftime("%d/%m/%Y")
    assunto_email = f"Análise de Pautas do Diário Oficial - {data_hoje}"
    
    # Prepara corpo do e-mail em Markdown
    corpo_email = f"# Análise de Pautas do Diário Oficial ({data_hoje})\n\n"
    corpo_email += f"Resultado da análise do Gemini para os textos selecionados (filtrados por pontuação e/ou obrigatórios):\n\n"
    corpo_email += "---\n\n"
    # Adiciona a resposta do Gemini. O Markdown será processado pela função send_email
    corpo_email += resposta_gemini_texto
    corpo_email += "\n\n---\n"
    corpo_email += f"\n*Este e-mail foi enviado automaticamente pelo script de análise.*\n"
    if 'contagem_tokens' in locals():
        corpo_email += f"*Total de tokens enviados ao Gemini (prompt+contexto): {contagem_tokens.total_tokens}*"
    
    # Chama a função de envio (que já está definida acima neste bloco)
    send_email(assunto_email, corpo_email)

else:
    print("\nERRO ou ALERTA: A variável 'contexto_final_formatado_para_gemini' está vazia ou não foi definida.")
    print("Não foi possível chamar o Gemini ou enviar o e-mail.")

# === FIM DO BLOCO DE CHAMADA GEMINI E ENVIO DE EMAIL ===
