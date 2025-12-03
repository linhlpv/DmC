# import all algorithms this benchmark implement

def call_algo(algo_name, config, mode, device):
    if mode == 12:
        # genplan approaches
        algo_name = algo_name.lower()
        assert algo_name in ['iql', 'iql_score_weight_cvae_filter_inside'']
        from offline_offline.iql import IQL
        from offline_offline.iql_score_weight_cvae_filter_wi import IQL_score_weight_cvae_filter_inside

        algo_to_call = {
            'iql': IQL,
            'iql_score_weight_cvae_filter_inside': IQL_score_weight_cvae_filter_inside,
        }
        algo = algo_to_call[algo_name]
        policy = algo(config, device)

    return policy