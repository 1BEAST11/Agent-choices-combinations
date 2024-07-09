#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <random>
#include <mpi.h>
#include <iomanip>
#include <fstream>

using namespace std;

// заполнение вектора конфигураций
vector<int> get_configuration(int index, int size)
{
    vector<int> combination(size, 0);
    for (int i = size - 1; i >= 0; --i) 
    {
        combination[i] = (index & 1);
        index >>= 1;
    }

    // 0 заменяется на -1, 1 остается 1
    transform(combination.begin(), combination.end(), combination.begin(), [](int x) { return (x == 0) ? -1 : 1; });
    return combination;
}

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int agent_amount = 20; // число агентов
    int configuration_amount = pow(2, agent_amount); // число конфигураций

    vector<double> agent_coeff_list(agent_amount); // ОДИНОЧНЫЕ коэффициенты агентов
    vector<double> agent_pair_coeff_list(((agent_amount * agent_amount) - agent_amount) / 2); // ПАРНЫЕ коэффициенты агентов

    vector<double> E_list; // список Е, рассчитанных 1 процессом
    vector<int> A_max_list; // список А, найденных одним процессом

    vector<double> all_E_list; // список из Е, собранных со всех процессов
    vector<int> all_A_max_list; // cписок из A_max, собранных со всех процессов

    double parallel_end_time = 0.0;
    double parallel_start_time = 0.0;

    if (world_rank == 0)
    {
        // генератор случайных чисел
        random_device rd;
        mt19937 generator(rd());

        uniform_real_distribution<double> distribution(-1.0, 1.0); // равномерное распределение

        // генерация случайных значений коэффициентов
        generate(agent_coeff_list.begin(), agent_coeff_list.end(), [&]() { return distribution(generator); });
        generate(agent_pair_coeff_list.begin(), agent_pair_coeff_list.end(), [&]() { return distribution(generator); });

        ofstream coeff_list("coeff_list.csv");
        ofstream pair_coeff_list("pair_coeff_list.csv");

        for (int i = 0; i < agent_coeff_list.size(); ++i)
        {
            coeff_list << agent_coeff_list[i] << "\n";
            
        }

        for (int i = 0; i < agent_pair_coeff_list.size(); ++i)
        {
            pair_coeff_list << agent_pair_coeff_list[i] << "\n";

        }

        coeff_list.close();
        pair_coeff_list.close();

        parallel_start_time = MPI_Wtime(); // время начала параллельной части
    }
    
    // рассылка коэффициентов всем процессам
    MPI_Bcast(agent_coeff_list.data(), agent_coeff_list.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(agent_pair_coeff_list.data(), agent_pair_coeff_list.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // поиск индекса максимального коэффициента a_max
    auto max_coeff_iterator = max_element(agent_coeff_list.begin(), agent_coeff_list.end());
    int max_coeff_index = distance(agent_coeff_list.begin(), max_coeff_iterator);

    // расчет E + поиск A_max
    for (int configuration_id = world_rank; configuration_id < configuration_amount; configuration_id += world_size)
    {
        vector<int> configuration = get_configuration(configuration_id, agent_coeff_list.size());

        // для одиночных коэффициентов
        double agent_sum = inner_product(configuration.begin(), configuration.end(), agent_coeff_list.begin(), 0.0);
        int A_max = configuration[max_coeff_index]; // A_max в текущей конфигурации

        // для парных коэффициентов
        double pair_sum = 0.0;
        for (int i = 0; i < agent_coeff_list.size(); ++i)
        {
            for (int j = i + 1; j < agent_coeff_list.size(); ++j)
            {
                pair_sum += agent_pair_coeff_list[i] * configuration[i] * configuration[j];
            }
        }

        double E = agent_sum + pair_sum; // вычисленное E
        A_max_list.push_back(A_max);
        E_list.push_back(E);
    }

    // Собираем данные со всех процессов
    int local_size = E_list.size();
    vector<int> recvcounts(world_size);

    MPI_Gather(&local_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_size = 0;
    vector<int> displs(world_size);

    if (world_rank == 0) 
    {
        displs[0] = 0;
        for (int i = 0; i < world_size; ++i) {
            displs[i] = total_size;
            total_size += recvcounts[i];
        }

        // Выделение памяти на нулевом процессе
        all_E_list.resize(total_size);
        all_A_max_list.resize(total_size);
    }

    // Собираем E и A_max
    MPI_Gatherv(E_list.data(), local_size, MPI_DOUBLE, all_E_list.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(A_max_list.data(), local_size, MPI_INT, all_A_max_list.data(), recvcounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);


    if (world_rank == 0)
    {
        parallel_end_time = MPI_Wtime(); // время окончания параллельной части
        cout << "Parallel Execution Time: " << parallel_end_time - parallel_start_time << " seconds\n"; // время параллельной части

        // мера неопределенности (хаоса)
        vector<double> chaos(10000);
        for (int i = 0; i < 10000; ++i)
        {
            chaos[i] = (i + 1) * 0.1;
        }

        // функция z(T)
        vector<double> z(chaos.size());
        for (int i = 0; i < chaos.size(); ++i)
        {
            for (int j = 0; j < all_E_list.size(); ++j)
            {
                z[i] += exp(all_E_list[j] / chaos[i]);
            }
        }

        // запись в файл рассчитанных значений <A_max>
        locale::global(locale("ru"));
        ofstream out_A("A_max(T).csv");

        for (int i = 0; i < chaos.size(); ++i)
        {
            double sum_A_max = 0;
            for (int j = 0; j < all_E_list.size(); ++j)
            {
                double ro = exp(all_E_list[j] / chaos[i]) / z[i];
                sum_A_max += all_A_max_list[j] * ro;
            }
            // запись значений A_max в файл
            out_A << fixed << setprecision(6) << sum_A_max << ";" << chaos[i] << "\n";
        }
        out_A.close();

        // сортировка списка значений E  впорядке возрастания
        sort(all_E_list.rbegin(), all_E_list.rend());

        // запись в файл отсортированных значений E
        ofstream out_E("E.csv");

        for (int i = 0; i < all_E_list.size(); ++i)
        {
            out_E << all_E_list[i] << "\n";
        }

        out_E.close();
    }

    MPI_Finalize();
    return 0;
}