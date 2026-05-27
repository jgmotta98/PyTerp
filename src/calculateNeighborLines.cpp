#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <tuple>
#include <omp.h>
#include <cmath>

namespace py = pybind11;

std::tuple<int32_t, int32_t> calculate_neighbor_lines(
    py::array_t<int64_t> vizinhos,
    py::array_t<double> coords,
    py::array_t<double> coords_vizinhos_x,
    py::array_t<double> coords_vizinhos_y,
    py::array_t<double> coord_central,
    py::array_t<double> plumas,
    py::array_t<bool> plumas_ainda_nao_detectadas,
    double valor_minimo_lii_metro,
    double tolerancia = 1e-6)
{
    uint32_t vizinhos_size = vizinhos.shape(0);
    uint32_t num_plumas = plumas.shape(1);
    uint32_t num_coords_vizinhos = coords_vizinhos_x.shape(0);

    auto vizinhos_ptr = vizinhos.unchecked<1>();

    auto coords_ptr = coords.unchecked<2>();

    auto coords_vizinhos_x_ptr = coords_vizinhos_x.unchecked<1>();

    auto coords_vizinhos_y_ptr = coords_vizinhos_y.unchecked<1>();

    auto coord_central_ptr = coord_central.unchecked<1>();

    auto plumas_ptr = plumas.unchecked<2>();

    auto plumas_ainda_nao_detectadas_ptr = plumas_ainda_nao_detectadas.unchecked<1>();

    double cx = coord_central_ptr(0);
    double cy = coord_central_ptr(1);
    double cz = coord_central_ptr(2);

    std::vector<int32_t> plumas_ativas;
    for (uint32_t p = 0; p < num_plumas; ++p)
    {
        if (plumas_ainda_nao_detectadas_ptr(p))
        {
            plumas_ativas.push_back(p);
        }
    }

    if (plumas_ativas.empty())
    {
        return std::make_tuple(0, -1);
    }

    int32_t global_melhor_score = -1;
    int32_t global_melhor_vizinho = -1;

#pragma omp parallel
    {
        int32_t local_melhor_score = -1;
        int32_t local_melhor_vizinho = -1;

#pragma omp for schedule(dynamic)
        for (int i = 0; i < vizinhos_size; i++)
        {
            int64_t v = vizinhos_ptr(i);

            double vx = coords_ptr(v, 0);
            double vy = coords_ptr(v, 1);
            double vz = coords_ptr(v, 2);

            if (std::abs(cz - vz) > tolerancia)
                continue;

            double diff_x = std::abs(cx - vx);
            double diff_y = std::abs(cy - vy);

            bool eixo_x = (diff_y < tolerancia) && (diff_x > tolerancia);
            bool eixo_y = (diff_x < tolerancia) && (diff_y > tolerancia);

            if (!(eixo_x || eixo_y))
                continue;

            double tamanho_reta = eixo_x ? diff_x : diff_y;

            if (tamanho_reta < tolerancia)
                continue;

            double x_min = (vx < cx) ? vx : cx;
            double x_max = (vx < cx) ? cx : vx;
            double y_min = (vy < cy) ? vy : cy;
            double y_max = (vy < cy) ? cy : vy;

            int32_t pontos_na_reta = 0;
            std::vector<double> soma_plumas(plumas_ativas.size(), 0.0);

            for (uint32_t j = 0; j < num_coords_vizinhos; j++)
            {
                double cv_x = coords_vizinhos_x_ptr(j);
                double cv_y = coords_vizinhos_y_ptr(j);

                bool na_reta = false;
                if (eixo_x)
                {
                    if (std::abs(cv_y - cy) < tolerancia && cv_x >= x_min && cv_x <= x_max)
                    {
                        na_reta = true;
                    }
                }
                else
                {
                    if (std::abs(cv_x - cx) < tolerancia && cv_y >= y_min && cv_y <= y_max)
                    {
                        na_reta = true;
                    }
                }

                if (na_reta)
                {
                    pontos_na_reta++;
                    int64_t idx_original = vizinhos_ptr(j);

                    for (size_t p = 0; p < plumas_ativas.size(); p++)
                    {
                        soma_plumas[p] += plumas_ptr(idx_original, plumas_ativas[p]);
                    }
                }
            }

            if (pontos_na_reta == 0)
                continue;

            int32_t score_atual = 0;
            for (size_t p = 0; p < plumas_ativas.size(); p++)
            {
                double lii_metro = (soma_plumas[p] / pontos_na_reta) * tamanho_reta;
                if (lii_metro >= valor_minimo_lii_metro)
                {
                    score_atual++;
                }
            }

            if (score_atual > local_melhor_score)
            {
                local_melhor_score = score_atual;
                local_melhor_vizinho = v;
            }
        }

#pragma omp critical
        {
            if (local_melhor_score > global_melhor_score)
            {
                global_melhor_score = local_melhor_score;
                global_melhor_vizinho = local_melhor_vizinho;
            }
        }
    }

    return std::make_tuple(global_melhor_score, global_melhor_vizinho);
}