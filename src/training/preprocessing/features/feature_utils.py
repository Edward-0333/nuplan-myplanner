import numpy as np
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


def find_candidate_lanes(data: dict, map_api, hist_steps):

    # cand_mask: [N, T, K] - 候选车道有效的 mask (1:有效，0:无效)。
    K = 256
    N = data['agent']['position'].shape[0]
    T = data['agent']['position'].shape[1]# [:,hist_steps:].shape[1]
    cand_mask = np.zeros((N, T, K), dtype=bool)
    all_lane_id = data['map']['polygon_road_lane_id']
    dict_all_lane_id = {int(lid): idx for idx, lid in enumerate(all_lane_id) if int(lid) > 0}
    target_roadblock_id = data['agent']['roadblock_id']#[:,hist_steps:]
    target_lane_id = data['agent']['lane_id']#[:,hist_steps:]
    assert len(dict_all_lane_id.keys())< K, f"too many lanes in the scene! > {K}"
    for i in range(N):
        for t in range(T):
            if target_roadblock_id[i][t] == 0:
                continue
            roadblock_id = str(target_roadblock_id[i][t])

            block = map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
            block = block or map_api.get_map_object(
                roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR
            )
            assert block is not None, f"roadblock {roadblock_id} not found in map!， 不对劲!"
            for lane in block.interior_edges:
                lane_id = int(lane.id)
                if lane_id in dict_all_lane_id:
                    lane_idx = dict_all_lane_id[lane_id]
                    cand_mask[i, t, lane_idx] = True
            for block in block.outgoing_edges:
                for lane in block.interior_edges:
                    lane_id = int(lane.id)
                    if lane_id in dict_all_lane_id:
                        lane_idx = dict_all_lane_id[lane_id]
                        cand_mask[i, t, lane_idx] = True
    return cand_mask, dict_all_lane_id


def transform_lane_id_to_probability(data: dict, hist_steps=21,max_lanes = 128):
    lane_id = data["map"]["polygon_road_lane_id"]
    lane_in_route = data["map"]["polygon_on_route"]
    lane_id = lane_id[lane_in_route]
    assert lane_id.shape[0] < max_lanes, f"too many lanes in the scene! > {max_lanes}"
    valid_mask = np.zeros(max_lanes, dtype=bool)
    valid_mask[: lane_id.shape[0]] = True

    agent_lane_id = data["agent"]["lane_id"]
    # target_lane_id: (N, T_future)
    target_lane_id = agent_lane_id[:, hist_steps:]

    N = target_lane_id.shape[0]
    # M = lane_id.shape[0]

    # Initialize probability matrix (agents x map_lanes)
    probability_matix = np.zeros((N, max_lanes), dtype=np.float64)

    # Map lane id value -> column index in lane_id array
    lane_index = {}
    for idx, lid in enumerate(lane_id):
        try:
            lid_int = int(lid)
        except Exception:
            continue
        if lid_int > 0:
            lane_index[lid_int] = idx

    # Inverse weighting with slower decay:
    # w_t = -1/6 * ln(t+1) + 1, t = 0..T_future-1
    # This decays slower than 1/(t+1). Clip at 0 to avoid negatives for very large T.
    T_future = target_lane_id.shape[1]
    weights = - (1.0 / 6.0) * np.log(np.arange(T_future, dtype=np.float64) + 1.0) + 1.0
    weights = np.clip(weights, a_min=0.0, a_max=None)

    for n in range(N):
        for t in range(T_future):
            lid = target_lane_id[n, t]
            try:
                lid_int = int(lid)
            except Exception:
                continue
            if lid_int <= 0:
                continue
            col = lane_index.get(lid_int, None)
            if col is None:
                continue
            if probability_matix[n, col] <= weights[t]:
                probability_matix[n, col] = weights[t]

    # Normalize per agent by row max (scale to [0, 1]) where possible
    row_max = probability_matix.max(axis=1, keepdims=True)
    nonzero_mask = row_max.squeeze(-1) > 0
    if np.any(nonzero_mask):
        probability_matix[nonzero_mask] = probability_matix[nonzero_mask] / row_max[nonzero_mask]
    return probability_matix, valid_mask


def transform_lane_id_to_target(data: dict, hist_steps=21, max_lanes=128):
    lane_id = data["map"]["polygon_road_lane_id"]
    lane_in_route = data["map"]["polygon_on_route"]
    lane_id = lane_id[lane_in_route]
    assert lane_id.shape[0] < max_lanes, f"too many lanes in the scene! > {max_lanes}"
    valid_mask = np.zeros(max_lanes, dtype=bool)
    valid_mask[: lane_id.shape[0]] = True

    agent_lane_id = data["agent"]["lane_id"]
    # target_lane_id: (N, T_future)
    target_lane_id = agent_lane_id[:, hist_steps:]
    veh_mask = data["agent"]["valid_mask"][:, hist_steps - 1]
    N = target_lane_id.shape[0]
    # M = lane_id.shape[0]
    test_loss(target_lane_id, veh_mask, max_lanes=max_lanes)
    return 0,0


def test_loss(target_lane_id,veh_mask, max_lanes=128):
    import torch
    import torch.nn.functional as F
    # logits: [B, N, T, K] - 网络输出的 logits。
    logits = torch.randn(1, target_lane_id.shape[0], target_lane_id.shape[1], max_lanes, requires_grad=True)
    # targets: [B, N, T] - 真实的车道ID（类标签）。
    targets = torch.from_numpy(target_lane_id).unsqueeze(0)
    # veh_mask: [B, N] - 有效车辆的 mask (1:有效，0:无效)。
    veh_mask = torch.from_numpy(veh_mask).unsqueeze(0)
    print(1)





def filter_on_route_map(data):

    map_info = data['map']
    polygon_on_route = map_info["polygon_on_route"]
    route_lane_point_position = map_info["point_position"][polygon_on_route]
    route_lane_point_vector = map_info["point_vector"][polygon_on_route]
    route_lane_point_orientation = map_info["point_orientation"][polygon_on_route]
    route_lane_polygon_center = map_info["polygon_center"][polygon_on_route]
    route_lane_polygon_type = map_info["polygon_type"][polygon_on_route]
    route_lane_speed_limit = map_info["polygon_speed_limit"][polygon_on_route]
    route_lane_has_speed_limit = map_info["polygon_has_speed_limit"][polygon_on_route]

    # transform lane_id to probability
    target_lane_matrix, target_lane_matrix_mask = transform_lane_id_to_probability(data)
    # 获取仅在route上的traffic light
    tl_status = data["map"]["polygon_tl_status"]
    tl_on_route = data["map"]["polygon_on_route"]
    tl_status = tl_status[tl_on_route]
    valid_tl_status = np.ones(tl_status.shape[0], dtype=bool)
    on_route_tl_status_mask = valid_tl_status
    route_map = {}
    route_map['point_position'] = route_lane_point_position
    route_map['point_vector'] = route_lane_point_vector
    route_map['point_orientation'] = route_lane_point_orientation
    route_map['polygon_center'] = route_lane_polygon_center
    route_map['polygon_type'] = route_lane_polygon_type
    route_map['polygon_speed_limit'] = route_lane_speed_limit
    route_map['polygon_has_speed_limit'] = route_lane_has_speed_limit
    route_map['on_route_tl_status'] = tl_status
    route_map['on_route_tl_status_mask'] = on_route_tl_status_mask
    route_map['target_lane_matrix_mask'] = target_lane_matrix_mask
    route_map['target_lane_matrix'] = target_lane_matrix
    route_map['valid_mask'] = np.ones((tl_status.shape[0], route_lane_point_position.shape[2]), dtype=bool)

    return route_map