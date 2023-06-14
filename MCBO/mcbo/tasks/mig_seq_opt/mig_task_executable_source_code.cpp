#include <cmath>
#include <fstream>
#include <iostream>
#include <lorina/aiger.hpp>
#include <mockturtle/mockturtle.hpp>

using namespace mockturtle;
using std::cout; using std::cerr;
using std::endl; using std::string;
using std::ifstream;


struct graph_info {
    float size;
    float depth;
    float num_pis;
    float num_pos;
};

template<class graph_type>
static inline graph_type load_graph(string path) {
    graph_type graph;
    auto const result = lorina::read_aiger(path, aiger_reader(graph));
    assert(result == lorina::return_code::success);
    return graph;
}

template<class graph_type>
static inline graph_info compute_klut_info(graph_type graph, int cut_size = 6) {
    lut_mapping_params ps;
    ps.cut_enumeration_ps.cut_size = cut_size;
    ps.rounds = 2u; // default is 2u
    ps.rounds_ela = 1u; // default is 1u
    ps.verbose = false;

    mapping_view<graph_type, true> mapped_graph(graph);
    lut_mapping<mapping_view<graph_type, true>, true>(mapped_graph, ps);

    const auto klut = *collapse_mapped_network<klut_network>(mapped_graph);

    depth_view klut_depth_view(klut);

    graph_info stats = {(float) klut_depth_view.num_gates(), (float) klut_depth_view.depth(),
                        (float) klut_depth_view.num_pis(), (float) klut.num_pos()};
    return stats;
}

void read_sequence(string sequence_path, int *sequence) {

    string filename(sequence_path);
    int number, i = 0;

    ifstream input_file(filename);
    if (!input_file.is_open()) {
        cerr << "Could not open the file - '"
             << filename << "'" << endl;
    }

    while (input_file >> number) {
        sequence[i] = number;
        i++;
    }
    cout << endl;
    input_file.close();

}

// 0 = balance, 1 = cut rewriting, 2 = cut rewriting -z, 3 = refactoring, 4 = refactoring -z, 5 = resubstitution, 6 = functional reduction
mig_network apply_operator(mig_network ntk, int operation) {
    // balancing parameters
    balancing_params balance_ps;
    balance_ps.cut_enumeration_ps.cut_size = 6u;
    balance_ps.only_on_critical_path = false;
    balance_ps.verbose = false;
    balance_ps.progress = false;
    sop_rebalancing<mig_network> balance_fn;

    // cut rewriting parameters
    cut_rewriting_params cut_ps;
    cut_ps.cut_enumeration_ps.cut_size = 4u;
    cut_ps.use_dont_cares = true;
    cut_ps.preserve_depth = false;
    cut_ps.allow_zero_gain = false;
    cut_ps.candidate_selection_strategy = cut_ps.minimize_weight;
    cut_ps.verbose = false;
    cut_ps.very_verbose = false;
    cut_ps.progress = false;
    mig_npn_resynthesis cut_resyn;

    // refactoring parameters
    refactoring_params refactor_ps;
    refactor_ps.max_pis = 4u;
    refactor_ps.use_dont_cares = true;
    refactor_ps.allow_zero_gain = false;
    refactor_ps.verbose = false;
    refactor_ps.progress = false;
    mig_npn_resynthesis refactor_resyn;

    // resubstitution parameters
    resubstitution_params resub_ps;
    resub_ps.max_pis = 8u;
    resub_ps.window_size = 12u;
    resub_ps.use_dont_cares = true;
    resub_ps.preserve_depth = false;
    resub_ps.verbose = false;
    resub_ps.progress = false;

    // functional reduction parameters
    functional_reduction_params func_red_ps;
    func_red_ps.saturation = false;
    func_red_ps.verbose = false;

    // views required for resubstitution
    depth_view ntk_depth_view(ntk);
    fanout_view ntk_depth_fanout_view(ntk_depth_view);

    switch (operation) {
        case 0:
            ntk = balancing<mig_network>(ntk, balance_fn, balance_ps);
            break;
        case 1:
            // Cut rewriting
            cut_ps.allow_zero_gain = false;
            ntk = cut_rewriting(ntk, cut_resyn, cut_ps);
            break;
        case 2:
            // Cut rewriting -z
            cut_ps.allow_zero_gain = true;
            ntk = cut_rewriting(ntk, cut_resyn, cut_ps);
            break;
        case 3:
            // refactoring
            refactor_ps.allow_zero_gain = false;
            refactoring(ntk, refactor_resyn, refactor_ps);
            ntk = cleanup_dangling(ntk);
            break;
        case 4:
            // refactoring -z
            refactor_ps.allow_zero_gain = true;
            refactoring(ntk, refactor_resyn, refactor_ps);
            ntk = cleanup_dangling(ntk);
            break;
        case 5:
            mig_resubstitution(ntk_depth_fanout_view);
            ntk = cleanup_dangling(ntk);
            break;
        case 6:
            functional_reduction(ntk, func_red_ps);
            ntk = cleanup_dangling(ntk);
        default:
            break;
    }
    return ntk;
}

mig_network apply_sequence(mig_network ntk, int sequence[], int seq_length) {
    int i;

    for (i = 0; i < seq_length; ++i) {
        ntk = apply_operator(ntk, sequence[i]);
    }

    return ntk;
}


int main(int argc, char *argv[]) {

    // Arguments: read_dir, network_name, path_to_sequence_txt_file, sequence_length
    string read_path = argv[1];
    string sequence_path = argv[2];
    const int sequence_length = atoi(argv[3]);

    // Read the sequence
    int *sequence;
    sequence = new int[sequence_length];
    read_sequence(sequence_path, sequence);

    // Load the network
    mig_network init_ntk = load_graph<mig_network>(read_path);
    mig_network ntk = load_graph<mig_network>(read_path);

    // Apply sequence
    mig_network final_ntk = apply_sequence(ntk, sequence, sequence_length);

    // Get initial and final size and depth
    graph_info init_k_lut_info = compute_klut_info(init_ntk, 6);
    graph_info final_k_lut_info = compute_klut_info(final_ntk, 6);

    // Write results to the console
    cout << ", " << init_k_lut_info.size << ", " << init_k_lut_info.depth << ", " << final_k_lut_info.size << ", "
         << final_k_lut_info.depth << ", \n";

    return 0;
}