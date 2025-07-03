#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <sstream>
#include <vector>
#include <string_view>
#include <algorithm>
#include <unordered_map>

#include <omp.h>

namespace py = pybind11;

std::unordered_map<std::string, std::vector<double>> extrairValoresVariaveis(
    const std::string &texto,
    const std::vector<std::string> &variaveis_pedidas,
    const unsigned int quant_x,
    const unsigned int quant_y,
    const unsigned int quant_z)
{
    std::vector<std::string> ordem_variaveis = {
        "P1", "P2", "U1", "nul", "V1", "nul", "W1", "nul", "R1", "R2", "nul", "KE", "EP", "nul", "nul", "C1", "nul", "nul", "nul",
        "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul",
        "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul",
        "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul",
        "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul",
        "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul",
        "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "nul", "PRPS", "LII", "EPKE", "DEN1", "EL1", "ENUT"
    };

    std::unordered_map<std::string, unsigned int> indices_variaveis_desejadas;
    std::unordered_map<std::string, std::vector<double>> dict_resultado;

    std::istringstream stream(texto);
    std::string linha;
    bool encontrou_tf = false;

    unsigned int total_variaveis_ativas = 0;
    std::string linha_tf_completa;

    unsigned int contador_linhas = 0;

    // Localiza linha TF e determina os índices das variáveis desejadas
    while (std::getline(stream, linha)) {
        // Simula o .trim()
        size_t first = linha.find_first_not_of(" \t\r");
        if (first == std::string::npos) continue;

        std::string_view linha_view = std::string_view(linha).substr(first);

        // Se a linha só tem T e F e não for a primeira, concatena
        bool linha_apenas_tf = !linha_view.empty();
        for (char c : linha_view) {
            if (c != 'T' && c != 'F') {
                linha_apenas_tf = false;
                break;
            }
        }

        if (linha_apenas_tf) {
            if (contador_linhas > 0)
            {
            linha_tf_completa += std::string(linha_view);
            }
            contador_linhas ++;
            if (contador_linhas > 2)
            {
                break;
            }
            continue;
        }
    }

    // Agora processamos a linha TF completa
    unsigned int idx_ativa = 0;
    for (unsigned int i = 0; i < linha_tf_completa.size(); ++i) {
        if (linha_tf_completa[i] == 'T') {
            const std::string &var = ordem_variaveis[i];
            if (std::find(variaveis_pedidas.begin(), variaveis_pedidas.end(), var) != variaveis_pedidas.end()) {
                indices_variaveis_desejadas[var] = idx_ativa;
                dict_resultado[var] = {};
            }
            idx_ativa++;
        }
    }

    total_variaveis_ativas = idx_ativa;

    // Calcula quantos valores por variável
    const unsigned int quant_valores_variavel = quant_x * quant_y * quant_z;

    // Lê os valores numéricos uma única vez
    std::vector<double> todos_valores;
    std::string token;
    while (std::getline(stream, linha)) {
        std::istringstream linha_stream(linha);
        while (linha_stream >> token) {
            size_t i = 0;
            size_t len = token.length();
            while (i < len) {
                size_t inicio = i;
                // Anda até próximo início de número
                ++i;
                while (i < len) {
                    if ((token[i] == '-' || token[i] == '+') &&
                        token[i - 1] != 'E' && token[i - 1] != 'e') {
                        break;
                    }
                    ++i;
                }
                std::string sub = token.substr(inicio, i - inicio);
                try {
                    todos_valores.push_back(std::stod(sub));
                } catch (...) {
                    // Ignora erro
                }
            }
        }
    }

    if (todos_valores.size() < quant_valores_variavel * total_variaveis_ativas) {
        throw std::runtime_error("Arquivo não tem valores suficientes.");
    }

    // Para cada variável desejada, extrai diretamente os blocos
    for (const auto &[variavel, idx] : indices_variaveis_desejadas) {
        size_t inicio = idx * quant_x * quant_y;

        std::vector<std::vector<double>> z_list;
        for (unsigned int z = 0; z < quant_z; ++z) {
            size_t offset = inicio + z * total_variaveis_ativas * (quant_x * quant_y);
            z_list.emplace_back(
                todos_valores.begin() + offset,
                todos_valores.begin() + offset + (quant_x * quant_y));
        }

        std::vector<double> valores_variavel;
        valores_variavel.reserve(quant_valores_variavel);

        for (unsigned int xy = 0; xy < quant_x * quant_y; xy++) {
            for (unsigned int z = 0; z < quant_z; z++) {
                valores_variavel.push_back(z_list[z][xy]);
            }
        }

        dict_resultado[variavel] = std::move(valores_variavel);
    }

    return dict_resultado;
}

std::unordered_map<std::string, std::vector<double>> variablesExtraction(const std::vector<std::string> &variaveis,
                                                                 const std::string &texto, const unsigned int quant_x, const unsigned int quant_y, const unsigned int quant_z)
{
    // Pega os valores das variáveis
    auto dict_variaveis_valaroes = extrairValoresVariaveis(texto, variaveis, quant_x, quant_y, quant_z);

    return dict_variaveis_valaroes;
}