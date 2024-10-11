import torch
from .similarity import modularity, conductance, intra_inter_sim, calc_adjacent_similarity_with_offset

def events_with_similarity_adjustment(events_base, A, similarity_metric='modularity'
                                      , min_size=0, offset=0):
    events_temp = [0]
    events_base = [i+offset for i in events_base]
    # remove any events too close together
    events_base_ = [events_base[0]] if events_base[0] >= min_size or offset==0 else []
    events_base = events_base_ + [events_base[i] for i in range(1, len(events_base)) if events_base[i] - events_base[i-1] >= min_size]
    if similarity_metric == 'modularity':
        sim_func = modularity
    elif similarity_metric == 'conductance':
        sim_func = lambda a,c: conductance(a,c)[0]
    elif similarity_metric == 'intra_inter_sim':
        sim_func = lambda a,c: intra_inter_sim(a,c)[0]
    else:
        raise NotImplementedError(f'Similarity metric {similarity_metric} not implemented')
    for event in events_base:
        if event - events_temp[-1] > min_size:
            if event-offset > min_size:
                # Allow a window (when possible) of twice as much as the original event size 
                original_event_size = event - events_temp[-1]
                half_size = int(original_event_size/2) 
                start_from = max(0, events_temp[-1] - half_size)
                end_to = min(A.shape[0], event + half_size)
                first_indx_to_check = max(offset-start_from, events_temp[-1]-start_from)  # either (relative) offset or (half_size or 0)
                last_indx_to_check = event - start_from
                
                TI_LES = torch.clone(A[start_from:end_to, :][:, start_from:end_to])
                adj_mod = calc_adjacent_similarity_with_offset(TI_LES, first_indx_to_check, last_indx_to_check
                                                                , sim_func=sim_func)
                if similarity_metric == 'conductance':
                    arg_mod = torch.argmin(adj_mod[min_size:])
                else:
                    arg_mod = torch.argmax(adj_mod[min_size:])
                events_temp.append(start_from + first_indx_to_check + min_size + arg_mod)
            else:
                events_temp.append(event)
        elif event - events_temp[-1] == min_size or offset == 0:
            events_temp.append(event)
        else:
            print(len(events_base))
            print(len(events_temp))
            raise Exception(f'Problem with event size: {event}')
        

    events_temp = [i.item()-offset.item() for i in events_temp[1:]]
    assert len(events_temp) == len(events_base), f'Problem with refinement does not have the same number of events: {len(events_temp)}, {len(events_base)}'
    return events_temp
