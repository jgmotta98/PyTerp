

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>
#include <string_view>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <charconv>
#include <iostream>
#include <tuple>
#include <omp.h>

namespace py = pybind11;

std::tuple<py::array_t<uint16_t>, py::array_t<uint16_t>> calculate_allocation_scores(
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> &idx_candidatos,
    py::array_t<bool, py::array::c_style | py::array::forcecast> &plumas,
    uint16_t quant_plumas,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> &offsets,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> &vizinhos_flat,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> &cobertura_atual,
    uint8_t k,
    bool usar_raio_detector)
{
    uint32_t num_candidatos = idx_candidatos.size();

    py::array_t<uint16_t> scores_adicionais(num_candidatos);
    py::buffer_info scores_adicionais_buf = scores_adicionais.request();
    std::memset(scores_adicionais_buf.ptr, 0, num_candidatos * sizeof(uint16_t));
    auto scores_adicionais_ptr = static_cast<uint16_t *>(scores_adicionais_buf.ptr);

    py::array_t<uint16_t> scores_total(num_candidatos);
    py::buffer_info scores_total_buf = scores_total.request();
    std::memset(scores_total_buf.ptr, 0, num_candidatos * sizeof(uint16_t));
    auto scores_total_ptr = static_cast<uint16_t *>(scores_total_buf.ptr);

    py::array_t<bool> plumas_abaixo_k(cobertura_atual.size());
    py::buffer_info plumas_abaixo_k_buf = plumas_abaixo_k.request();
    auto plumas_abaixo_k_ptr = static_cast<uint8_t *>(plumas_abaixo_k_buf.ptr);

    py::buffer_info cobertura_atual_buf = cobertura_atual.request();
    auto cobertura_atual_ptr = static_cast<uint8_t *>(cobertura_atual_buf.ptr);

#pragma omp parallel for
    for (int i = 0; i < cobertura_atual.size(); i++)
    {
        plumas_abaixo_k_ptr[i] = cobertura_atual_ptr[i] < k;
    }

    py::buffer_info idx_candidatos_buf = idx_candidatos.request();
    auto idx_candidatos_ptr = static_cast<uint32_t *>(idx_candidatos_buf.ptr);

    py::buffer_info offsets_buf = offsets.request();
    auto offsets_ptr = static_cast<uint32_t *>(offsets_buf.ptr);

    py::buffer_info vizinhos_flat_buf = vizinhos_flat.request();
    auto vizinhos_flat_ptr = static_cast<uint32_t *>(vizinhos_flat_buf.ptr);

    py::buffer_info plumas_buf = plumas.request();
    auto plumas_ptr = static_cast<bool *>(plumas_buf.ptr);

    int max_threads = omp_get_max_threads();
    std::vector<uint8_t> thread_buffers(max_threads * quant_plumas);

    py::gil_scoped_release release;

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_candidatos; i++)
    {
        uint32_t pos_original = idx_candidatos_ptr[i];

        uint32_t start_idx = offsets_ptr[pos_original];
        uint32_t end_idx = offsets_ptr[pos_original + 1];

        uint32_t vizinhos_size = end_idx - start_idx;

        uint32_t *vizinhos = vizinhos_flat_ptr + start_idx;

        bool *plumas_do_sensor;

        if (usar_raio_detector && vizinhos_size > 0)
        {
            int thread_id = omp_get_thread_num();

            bool *temp_ptr = reinterpret_cast<bool *>(thread_buffers.data() + (thread_id * quant_plumas));

            bool *primeiro_vizinho = plumas_ptr + (vizinhos[0] * quant_plumas);
#pragma omp simd
            for (int n = 0; n < quant_plumas; n++)
            {
                temp_ptr[n] = primeiro_vizinho[n];
            }

            for (int j = 1; j < vizinhos_size; j++)
            {
                uint32_t idx_v = vizinhos[j];
                bool *plumas_vizinho_ptr = plumas_ptr + idx_v * quant_plumas;

                for (int n = 0; n < quant_plumas; n++)
                {
                    temp_ptr[n] |= plumas_vizinho_ptr[n];
                }
            }

            plumas_do_sensor = temp_ptr;
        }
        else
        {
            plumas_do_sensor = plumas_ptr + (pos_original * quant_plumas);
        }

        bool is_valid = true;

        if (vizinhos_size > 0 && !(usar_raio_detector))
        {
            for (int j = 0; j < vizinhos_size; j++)
            {
                uint32_t idx_v = vizinhos[j];

                bool *pluma_vizinhanca = plumas_ptr + idx_v * quant_plumas;

                for (int n = 0; n < quant_plumas; n++)
                {
                    if (pluma_vizinhanca[n] < plumas_do_sensor[n])
                    {
                        is_valid = false;
                        break;
                    }
                }

                if (!is_valid)
                {
                    break;
                }
            }
        }

        if (is_valid)
        {

            uint16_t score_k = 0;
            uint16_t score_basico = 0;

            uint16_t score_k_total = 0;
            uint16_t score_basico_total = 0;
#pragma omp simd reduction(+ : score_k, score_basico, score_k_total, score_basico_total)
            for (int j = 0; j < quant_plumas; j++)
            {
                bool plumas_cobertas_abaixo_k = (plumas_do_sensor[j] & plumas_abaixo_k_ptr[j]);
                uint8_t nova_cobertura = cobertura_atual_ptr[j] + plumas_do_sensor[j];

                score_k = score_k + ((nova_cobertura >= k) && plumas_cobertas_abaixo_k);
                score_basico = score_basico + plumas_cobertas_abaixo_k;

                score_k_total = score_k_total + ((nova_cobertura >= k) && plumas_do_sensor[j]);
                score_basico_total = score_basico_total + plumas_do_sensor[j];
            }

            if (score_k > 0)
            {
                scores_adicionais_ptr[i] = score_k;
                scores_total_ptr[i] = score_k_total;
            }
            else
            {
                scores_adicionais_ptr[i] = score_basico;
                scores_total_ptr[i] = score_basico_total;
            }
        }
    }

    std::tuple<py::array_t<uint16_t>, py::array_t<uint16_t>> response(scores_adicionais, scores_total);

    return response;
}
