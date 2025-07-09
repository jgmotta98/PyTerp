#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <sstream>
#include <vector>
#include <string_view>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <charconv>
#include <iostream>
#include <chrono>

#include <omp.h>

namespace py = pybind11;

// Função para casos com menos de 2 Variáveis
void extrairValoresVariaveis(
    std::string_view &texto,
    const std::vector<std::string> &variaveis_pedidas,
    const std::unordered_set<std::string_view> &set_variaveis_pedidas,
    std::unordered_map<std::string, unsigned int> &indices_variaveis_desejadas, std::unordered_map<std::string, std::vector<double>> &dict_resultado,
    unsigned int &total_variaveis_ativas,
    const unsigned int quant_x,
    const unsigned int quant_y,
    const unsigned int quant_z)
{

    const unsigned int valores_por_linha = 6;                    // Quantidade máximo de valores por linha
    const unsigned int valores_por_plano_xy = quant_x * quant_y; // Quantidade de valores por Z

    const unsigned int linhas_por_plano_xy = (valores_por_plano_xy + valores_por_linha - 1) / valores_por_linha; // Divisão com arredondamento para cima para encontrar linhas por Z

    // Listas de verificação para acesso rápido
    std::vector<bool> indice_e_desejado(total_variaveis_ativas, false);
    std::vector<std::string> indice_para_nome(total_variaveis_ativas);
    for (const auto &[nome, idx] : indices_variaveis_desejadas)
    {
        if (idx < total_variaveis_ativas)
        {
            indice_e_desejado[idx] = true;
            indice_para_nome[idx] = nome;
        }
    }

    // Dicionário temporário para armazenar os dados na ordem em que são lidos (separados por plano Z)
    std::unordered_map<std::string, std::vector<std::vector<double>>> dados_por_z;
    // Pré-aloca espaço para as variáveis desejadas para evitar realocações
    for (const auto &var_nome : variaveis_pedidas)
    {
        dados_por_z[var_nome].reserve(quant_z);
    }

    const char *p_current = texto.data();                           // Ponteiro que indica para o início do texto com os valores
    const char *const p_stream_end = texto.data() + texto.length(); // Ponteiro que indica para o fim do arquivo

    // Loop principal: itera sobre a quantidade de Z
    for (unsigned int z = 0; z < quant_z; ++z)
    {
        // Loop intero: itera sobre cada variável presente no Arquivo
        for (unsigned int var_idx_no_arquivo = 0; var_idx_no_arquivo < total_variaveis_ativas; ++var_idx_no_arquivo)
        {
            // Se for uma variável desejada, processa
            if (indice_e_desejado[var_idx_no_arquivo])
            {
                const std::string &nome_var_atual = indice_para_nome[var_idx_no_arquivo]; // Nome da variável
                std::vector<double> valores_plano_xy;                                     // Vetor para armazenar os valores de XY desse Z
                valores_plano_xy.reserve(valores_por_plano_xy);                           // Reserva espaço com tamanho exato de valores de XY

                // Lê apenas as linhas de um plano XY
                for (unsigned int i = 0; i < linhas_por_plano_xy && p_current < p_stream_end; ++i)
                {
                    const char *p_start_line = p_current;                                                           // Ponteiro que aponta para o início da linha
                    const char *p_end_line = (const char *)memchr(p_start_line, '\n', p_stream_end - p_start_line); // Ponteiro que pega o primeira quebra de linha (fim da linha)
                    if (!p_end_line)
                        p_end_line = p_stream_end; // Se não tiver quebra de linha então o fim da linha é o fim do arquivo

                    const char *p_num_parser = p_start_line; // Ponteiro para fazer a leitura dos valores presentes na linha
                    while (p_num_parser < p_end_line)        // Enquanto o ponteiro de leitura não chegar no fim, processa
                    {
                        while (p_num_parser < p_end_line && std::isspace(static_cast<unsigned char>(*p_num_parser))) // Enquanto o ponteiro apontar para um caracter vazio, passa para o próximo
                        {
                            ++p_num_parser;
                        }
                        if (p_num_parser >= p_end_line)
                            break;

                        const char *p_start_num = p_num_parser;  // Ponteiro que aponta para o início de um número
                        const char *p_end_num = p_start_num + 1; // Ponteiro que aponta para o fim de um número (inicializado como o próximo caracter)

                        while (p_end_num < p_end_line) // Enquanto o fim do número não for igual ao fim da linha, processa
                        {
                            char current_char = *p_end_num; // Caracter atual é igual ao fim do número

                            if (std::isspace(static_cast<unsigned char>(current_char))) // Se caracter for espaço, acabou o número, passa para o próximo
                                break;

                            if ((current_char == '+' || current_char == '-') && (*(p_end_num - 1) != 'E' && *(p_end_num - 1) != 'e')) // Se caracter for igual a um "+" ou "-" e não for da notação ciêntifica,
                                break;                                                                                                // acabou o número, passa para o próximo

                            ++p_end_num; // Se não for o fim do número, pega o próximo caracter para análise
                        }

                        if (p_start_num < p_end_num) // Se o número existe, processa
                        {
                            double valor;
                            if (auto [ptr, ec] = std::from_chars(p_start_num, p_end_num, valor); ec == std::errc()) // Converte a string para double a partir dos ponteiros de início e fim do número
                            {
                                valores_plano_xy.push_back(valor); // Coloca no fim da lista de valores de XY correspondestes a essa variável e Z
                            }
                        }
                        p_num_parser = p_end_num; // Ponteiro para leitura da linha passa a ser o que termina o número
                    }
                    p_current = p_end_line + 1; // Passa para a próxima linha
                }
                // Adiciona o plano lido ao dicionário temporário
                dados_por_z.at(nome_var_atual).push_back(std::move(valores_plano_xy));
            }
            else
            {
                // Variável não desejada, pula as linhas de um plano XY
                for (unsigned int i = 0; i < linhas_por_plano_xy && p_current < p_stream_end; ++i)
                {
                    const char *p_end_line = (const char *)memchr(p_current, '\n', p_stream_end - p_current);
                    if (!p_end_line)
                    {
                        p_current = p_stream_end;
                        break;
                    }
                    p_current = p_end_line + 1;
                }
            }
        }
    }

    const unsigned int quant_valores_variavel = quant_x * quant_y * quant_z; // Valores presentes em uma variável

// Reordenação Z-pencil de forma paralela
#pragma omp parallel for
    // Loop principal: itera sobre as variáveis pedidas
    for (int i = 0; i < variaveis_pedidas.size(); ++i)
    {
        const std::string &var_nome = variaveis_pedidas[i];

        // Se variável existe no arquivo, processa
        if (dados_por_z.count(var_nome))
        {
            const auto &z_list = dados_por_z.at(var_nome);    // Pega valor correspondete a variável no dicionário temporário
            std::vector<double> valores_variavel;             // Vetor para armazenar valores da variável em Z-pencil
            valores_variavel.reserve(quant_valores_variavel); // Reserva espaço exato para o vetor

            // Reordena de planos (Z-major) para colunas (Z-pencil)
            for (unsigned int xy = 0; xy < valores_por_plano_xy; ++xy)
            {
                for (unsigned int z = 0; z < quant_z; ++z)
                {
                    // Adiciona validação para o caso de o arquivo terminar inesperadamente
                    if (z < z_list.size() && xy < z_list[z].size())
                    {
                        valores_variavel.push_back(z_list[z][xy]);
                    }
                }
            }
#pragma omp critical
            {
                dict_resultado[var_nome] = std::move(valores_variavel); // Adiciona no dicionário de resultado os valores em ordem
            }
        }
    }
}

std::string_view variablesSetup(const std::string &texto, const std::unordered_set<std::string_view> &set_variaveis_pedidas,
                                std::unordered_map<std::string, unsigned int> &indices_variaveis_desejadas, unsigned int &total_variaveis_ativas,
                                std::unordered_map<std::string, std::vector<double>> &dict_resultado)
{
    std::vector<std::string> ordem_variaveis = {
        "P1", "P2", "U1", "nul", "V1", "nul", "W1", "nul", "R1", "R2", "nul", "KE", "EP", "nul", "nul", "C1", "nul", "nul", "nul",
        "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul",
        "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul",
        "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul",
        "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul",
        "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul",
        "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul",
        "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "PRPS", "LII", "EPKE", "DEN1", "EL1", "ENUT"};

    std::string linha_tf_completa; // Linha Completa dos True or False
    size_t pos_inicio_dados = 0;   // Posição referente ao começo dos dados no arquiov

    size_t pos_atual = 0;       // Posição atual na leitura do arquivo
    int contador_linhas_tf = 0; // Contador de linhas encontradas com True or False

    // Loop para leitura da sequência de True or False
    while (pos_atual < texto.length())
    {

        size_t pos_fim_linha = texto.find('\n', pos_atual);                          // Posição do fim da linha (Lugar onde tem a quebra de linha)
        std::string_view linha(texto.data() + pos_atual, pos_fim_linha - pos_atual); // Linha do texto (Posição inicial do texto + Posição atual : Posição Fim Linha - Posição Atual)
        size_t first = linha.find_first_not_of(" \t\r");                             // Encontra primeiro valor que não seja espaço
        std::string_view linha_view = linha.substr(first);                           // Simula .trim() (Pega toda a linha após os espaços)

        bool linha_apenas_tf = !linha_view.empty();
        // Loop para verificar se a linha tem apenas T e F
        for (char c : linha_view)
        {
            if (c != 'T' && c != 'F')
            {
                linha_apenas_tf = false;
                break;
            }
        }
        // Se linha é valida e não for a primeira, se concatena
        if (linha_apenas_tf)
        {
            if (contador_linhas_tf > 0)
                linha_tf_completa.append(linha_view);
            contador_linhas_tf++;
            if (contador_linhas_tf > 2)
            {
                pos_inicio_dados = pos_fim_linha + 1; // Posição do início dos dados é a próxima após as linhas TFFT...
                break;
            }
        }
        pos_atual = pos_fim_linha + 1; // Passa para a próxima linha
    }

    if (pos_inicio_dados == 0)
        throw std::runtime_error("Bloco de dados não encontrado.");

    // Loop para fazer 'conversão' de índices das variáveis desejadas
    for (unsigned int i = 0; i < linha_tf_completa.size() && i < ordem_variaveis.size(); ++i)
    {
        // Se a variável for marcada com T, está presente no arquivo
        if (linha_tf_completa[i] == 'T')
        {
            const std::string &var = ordem_variaveis[i];
            // Se a variável está entre as desejadas, adiciona nos dicionários
            if (set_variaveis_pedidas.count(var))
            {
                indices_variaveis_desejadas[var] = total_variaveis_ativas;
                dict_resultado[var] = {};
            }
            total_variaveis_ativas++;
        }
    }

    return std::string_view(texto).substr(pos_inicio_dados);
}

std::unordered_map<std::string, std::vector<double>> variablesExtraction(const std::vector<std::string> &variaveis,
                                                                         const std::string &texto, const unsigned int quant_x, const unsigned int quant_y, const unsigned int quant_z)
{
    const std::unordered_set<std::string_view> set_variaveis_pedidas(variaveis.begin(), variaveis.end()); // Conversão da lista para um set por ser mais rápido a pesquisa O(1)
    std::unordered_map<std::string, unsigned int> indices_variaveis_desejadas;                            // Dicionário com os índices das variáveis no arquiov
    unsigned int total_variaveis_ativas = 0;

    std::unordered_map<std::string, std::vector<double>> dict_resultado; // Dicionário para os resultados de cada variável

    std::string_view texto_dados = variablesSetup(texto, set_variaveis_pedidas, indices_variaveis_desejadas, total_variaveis_ativas, dict_resultado);

    // Pega os valores das variáveis
    extrairValoresVariaveis(texto_dados, variaveis, set_variaveis_pedidas, indices_variaveis_desejadas, dict_resultado, total_variaveis_ativas, quant_x, quant_y, quant_z);

    return dict_resultado;
}