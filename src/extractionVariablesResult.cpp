#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <string_view>
#include <unordered_map>

#include <omp.h>

namespace py = pybind11;

std::vector<double> ordanamentoY(const std::string_view &texto_bloco, const std::unordered_map<std::string, double> &mapping_dict)
{
    // Vetor para armazenar os blocos
    std::vector<std::vector<std::vector<std::string>>> lista_blocos;

    // Vetor correspondente ao bloco atual
    std::vector<std::vector<std::string>> bloco_atual;

    std::string texto_bloco_str = std::string(texto_bloco);

    std::istringstream stream(texto_bloco_str);
    std::string linha;

    while (std::getline(stream, linha)) {
        const auto first = linha.find_first_not_of(" \t\r");
        if (first == std::string::npos) continue;

        // Realiza um .trim() no texto
        std::string_view linha_trim = std::string_view(linha).substr(first);

        // Se linha começa com IX=, ou seja, fim de bloco, adiciona o bloco na lista e limpa o bloco atual
        if (linha_trim.rfind("IX=", 0) == 0) {
            if (!bloco_atual.empty()) {
                lista_blocos.push_back(bloco_atual);
                bloco_atual.clear();
            }
        // Se linha começa com IY=, linha de valores da varíavel
        } else if (linha_trim.rfind("IY=", 0) == 0) {
            std::vector<std::string> valores;


            size_t igual_pos = linha_trim.find('=');
            if (igual_pos == std::string::npos) continue;

            std::string_view valores_view = linha_trim.substr(igual_pos + 1);

            // Ignora o primeiro valor (coordenada Y)
            size_t i = 0;
            int valores_encontrados = 0;

            // Faz a leitura dos valores na linha
            while (i < valores_view.size()) {
                while (i < valores_view.size() && std::isspace(valores_view[i])) i++;

                size_t start = i;

                // Sinal opcional
                if (i < valores_view.size() && (valores_view[i] == '+' || valores_view[i] == '-')) i++;

                bool tem_numero = false;

                // Parte inteira
                while (i < valores_view.size() && std::isdigit(valores_view[i])) {
                    i++;
                    tem_numero = true;
                }

                // Parte decimal
                if (i < valores_view.size() && valores_view[i] == '.') {
                    i++;
                    while (i < valores_view.size() && std::isdigit(valores_view[i])) {
                        i++;
                        tem_numero = true;
                    }
                }

                // Notação científica
                if (tem_numero && i < valores_view.size() && (valores_view[i] == 'E' || valores_view[i] == 'e')) {
                    i++;
                    if (i < valores_view.size() && (valores_view[i] == '+' || valores_view[i] == '-')) i++;
                    while (i < valores_view.size() && std::isdigit(valores_view[i])) i++;
                }

                if (tem_numero) {
                    if (++valores_encontrados > 1)
                        valores.emplace_back(std::string(valores_view.substr(start, i - start)));
                    continue;
                }

                std::string_view r = valores_view.substr(i);
                if (r.size() >= 4 && r.substr(0, 4) == "none") {
                    if (++valores_encontrados > 1) valores.emplace_back("none");
                    i += 4;
                } else if (r.size() >= 8 && r.substr(0, 8) == "pil prop") {
                    if (++valores_encontrados > 1) valores.emplace_back("pil prop");
                    i += 8;
                } else if (r.size() >= 8 && r.substr(0, 8) == "blockage") {
                    if (++valores_encontrados > 1) valores.emplace_back("blockage");
                    i += 8;
                } else {
                    // pula tokens inválidos para evitar travamento
                    while (i < valores_view.size() && !std::isspace(valores_view[i])) i++;
                }
            }
            // Adiciona os valores ao bloco atual
            if (!valores.empty()) {
                bloco_atual.insert(bloco_atual.begin(), valores);
            }
        }
    }
    // Se terminar em um bloco, coloca ele na lista de blocos
    if (!bloco_atual.empty())
    {
        lista_blocos.push_back(bloco_atual);
    }

    // Calcular o tamanho total e os deslocamentos de cada bloco serialmente antes do loop paralelo.
    size_t total_size = 0;
    std::vector<size_t> offsets;
    offsets.push_back(0);

    for (const auto& bloco : lista_blocos) {
        if (bloco.empty()) continue;

        size_t expected_cols = bloco[0].size();
        total_size += bloco.size() * expected_cols;
        offsets.push_back(total_size);
    }

    std::vector<double> valores_ordenados(total_size);

    // Loop para ordendar valores de Y referentes ao X
    #pragma omp parallel for
    for (size_t b = 0; b < lista_blocos.size(); b++) {
        const auto &bloco = lista_blocos[b];
        if (bloco.empty())
            continue;

        size_t num_linhas = bloco.size();      // Número de Ys
        size_t num_colunas = bloco[0].size();  // Número de Xs
        size_t bloco_offset = offsets[b];      // Pega o deslocamento pré-calculado     

        // Loop para pegar cada X
        for (size_t y = 0; y < num_colunas; ++y)
        {
            // Loop para pegar cada Y
            for (int x = 0; x < num_linhas; ++x)
            {
                size_t idx = bloco_offset + y * num_linhas + x;

                // Valor de X para aquele Y
                const std::string &valor_str = bloco[x][y];

                // Tenta transformar em float, se não traduz usando o dicionário, no fim adiciona no fim di vetor
                try
                {

                    valores_ordenados[idx] = (std::stod(valor_str));
                }
                catch (...)
                {
                    auto it = mapping_dict.find(valor_str);
                    if (it != mapping_dict.end())
                    {
                        double valor = it->second;
                        valores_ordenados[idx] = (valor);
                    }
                    
                }
            }
        }
    }

    // Retorna os valores ordenados em relação a X
    return valores_ordenados;
}

std::vector<double> extrairValoresVariaveisResult(const std::string &texto, const std::string &variavel, std::unordered_map<std::string, double> &mapping_dict)
{
    // Vetor para armzenar os blocos (Z) em forma de texto
    std::vector<std::string_view> bloco_list;

    const std::string target_header = "Field Values of " + variavel;
    const std::string generic_header = "Field Values of ";
    size_t search_pos = 0;

    // Indentifica todos os blocos (Z) da variável presentes no arquivo
    while ((search_pos = texto.find(target_header, search_pos)) != std::string::npos) {
        size_t block_data_start = texto.find('\n', search_pos);
        if (block_data_start == std::string::npos) break;
        block_data_start++;

        size_t next_header_pos = texto.find(generic_header, block_data_start);
        
        std::string_view block_content = (next_header_pos == std::string::npos)
            ? std::string_view(texto).substr(block_data_start)
            : std::string_view(texto).substr(block_data_start, next_header_pos - block_data_start);

        if (!block_content.empty()) {
            bloco_list.push_back(block_content);
        }
        if (next_header_pos == std::string::npos) {
            break; 
        }
        search_pos = next_header_pos;
    }
    
    // Se nenhuma camada Z for encontrada, retorna um vetor vazio.
    if (bloco_list.empty())
    {
        return {};
    }

    // Vetor para armzenar os valores das variáveis na coordenada (X, Y) referente ao seu Z
    std::vector<std::vector<double>> z_list(bloco_list.size());

    // Faz a conversão de texto para double e armazena corretamente no z_list
    #pragma omp parallel for
    for (size_t i = 0; i < bloco_list.size(); ++i) {
        // Cada thread trabalha em um bloco diferente e escreve em uma posição única do vetor.
        z_list[i] = ordanamentoY(bloco_list[i], mapping_dict);
    }

    // Se o Z for vazio, retorna vazio
    if (z_list[0].empty()) return {};

    // Número de pontos em um Z
    size_t num_xy_points = z_list[0].size();

    // Número de Zs
    size_t num_z_points = z_list.size();

    // Vetor de retorno na ordem correta
    std::vector<double> return_list(num_xy_points * num_z_points);

    // Loop para colocar os valores em ordem referentes a cada Z
    #pragma omp parallel for collapse(2)
    for (size_t n = 0; n < num_xy_points; ++n) // Itera sobre cada ponto no plano XY
    {
        for (size_t z = 0; z < num_z_points; ++z) // Itera sobre cada plano Z
        {
            size_t idx = n * num_z_points + z;
            return_list[idx] = z_list[z][n];
        }
    }

    // Retorna os valores na ordem correta
    return return_list;
}

std::unordered_map<std::string, std::vector<double>> variablesExtractionResult(const std::vector<std::string> &variaveis,
                                                                 const std::string &texto,
                                                                 std::unordered_map<std::string, double> &mapping_dict)
{

    std::unordered_map<std::string, std::vector<double>> dict_valores;

    // Loop para leitura de todas as variáveis
    for (const auto& variavel : variaveis)
    {
        // Pega os valores das variáveis
        auto variavelCoords = extrairValoresVariaveisResult(texto, variavel, mapping_dict);
        
        // Se o variavelCoords não for vazio, adiciona no dicionário
        if (!variavelCoords.empty())
        {
            // Zona crítica para paralelismo, pode ter acesso atravessado de threads, por isso o critical, garantindo acesso uma thread por vez
            {
                dict_valores[variavel] = std::move(variavelCoords);
            }
        }
    }

    return dict_valores;
}