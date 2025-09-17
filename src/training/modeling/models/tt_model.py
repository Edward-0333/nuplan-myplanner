from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F
import math
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)
from torch.fx import map_arg

from src.training.preprocessing.feature_builders.tt_feature_builder import TTFeatureBuilder
from .layers.fourier_embedding import FourierEmbedding
from .layers.transformer import TransformerEncoderLayer
from .modules.agent_encoder import AgentEncoder
from .modules.agent_predictor import AgentPredictor
from .modules.map_encoder import MapEncoder
from .modules.map_route_encoder import MapRouteEncoder
from .modules.static_objects_encoder import StaticObjectsEncoder
from .modules.planning_decoder import PlanningDecoder
from .layers.mlp_layer import MLPLayer

trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)


class PlanningModel(TorchModuleWrapper):
    def __init__(
        self,
        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        history_steps=21,
        future_steps=80,
        encoder_depth=4,
        decoder_depth=4,
        drop_path=0.2,
        dropout=0.1,
        num_heads=8,
        num_modes=6,
        use_ego_history=False,
        state_attn_encoder=True,
        state_dropout=0.75,
        use_hidden_proj=False,
        cat_x=False,
        ref_free_traj=False,
        feature_builder: TTFeatureBuilder = TTFeatureBuilder(),
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.use_hidden_proj = use_hidden_proj
        self.num_modes = num_modes
        self.radius = feature_builder.radius
        self.ref_free_traj = ref_free_traj

        self.pos_emb = FourierEmbedding(3, dim, 64)
        self.location_emb = FourierEmbedding(3, dim, 64)
        self.agent_info_emb = FourierEmbedding(5, dim, 64)
        self.agent_encoder = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
        )

        self.map_encoder = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
            use_lane_boundary=True,
        )
        self.map_route_encoder = MapRouteEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
            use_lane_boundary=True,
        )

        self.static_objects_encoder = StaticObjectsEncoder(dim=dim)

        self.agent_encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )

        self.map_encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )

        self.scene_encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )

        self.route_encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

        self.traffic_light_emb = nn.Embedding(4, dim)

        self.agent_predictor = AgentPredictor(dim=dim, future_steps=future_steps)
        self.planning_decoder = PlanningDecoder(
            num_mode=num_modes,
            decoder_depth=decoder_depth,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4,
            dropout=dropout,
            cat_x=cat_x,
            future_steps=future_steps,
        )

        if use_hidden_proj:
            self.hidden_proj = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )

        if self.ref_free_traj:
            self.ref_free_decoder = MLPLayer(dim, 2 * dim, future_steps * 4)

        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, data):
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_vel = data["agent"]["velocity"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]
        on_route_tl_status = data["route_map"]["on_route_tl_status"].long()
        on_route_tl_status_mask = data["route_map"]["on_route_tl_status_mask"]
        target_lane_matrix = data["route_map"]["target_lane_matrix"]
        test_target_lane_matrix = data["route_map"]["target_lane_matrix"][0].cpu().numpy()
        test_target_lane_matrix1 = data["route_map"]["target_lane_matrix"][1].cpu().numpy()

        bs, A = agent_pos.shape[0:2]
        # agent_pos单独emb
        agent_heading = (agent_heading + math.pi) % (2 * math.pi) - math.pi
        agent_location = torch.cat([agent_pos, agent_heading.unsqueeze(-1)], dim=-1)
        location_embed = self.location_emb(agent_location)

        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)
        agent_key_padding = ~(agent_mask.any(-1))

        for blk in self.agent_encoder_blocks:
            location_embed, x_agent = blk(location_embed, x_agent, key_padding_mask=agent_key_padding, return_attn_weights=False)
        location_embed = self.norm1(location_embed)
        map_key_padding = ~(polygon_mask.any(-1))
        for blk in self.map_encoder_blocks:
            location_embed, x_polygon = blk(location_embed, x_polygon, key_padding_mask=map_key_padding, return_attn_weights=False)
        location_embed = self.norm2(location_embed)

        x_static, static_pos, static_key_padding = self.static_objects_encoder(data)
        x_tl_status = self.traffic_light_emb(on_route_tl_status)
        x_tl_status[~on_route_tl_status_mask] = 0
        # test = x_tl_status.clone()[0].cpu().numpy()
        # test1 = x_tl_status.clone()[1].cpu().numpy()

        agent_location_copy = agent_location.clone()
        agent_information = torch.cat([agent_location_copy, agent_vel], dim=-1)
        x_agent_info = self.agent_info_emb(agent_information)
        x_scene = torch.cat((x_agent_info, x_static, x_tl_status), dim=1)
        scene_key_padding = torch.cat((agent_key_padding, static_key_padding, ~on_route_tl_status_mask), dim=-1)
        # test = ~on_route_tl_status_mask[0].cpu().numpy()
        # test2 = ~on_route_tl_status_mask[1].cpu().numpy()
        for blk in self.scene_encoder_blocks:
            location_embed, x_scene, attn = blk(location_embed, x_scene, key_padding_mask=scene_key_padding, return_attn_weights=True)
        location_embed = self.norm3(location_embed)

        x_route_polygon = self.map_route_encoder(data)
        route_key_padding = data['route_map']['on_route_tl_status_mask']
        for blk in self.route_encoder_blocks:
            location_embed, x_route_polygon = blk(location_embed, x_route_polygon, key_padding_mask=~route_key_padding, return_attn_weights=False)
        location_embed = self.norm4(location_embed) # 可以直接用作后续的车道分类，与target_lane_matrix
        route_key_padding = F.pad(route_key_padding, (0, location_embed.shape[2] - route_key_padding.shape[1]), value=False)

        # test1 = route_key_padding[0].cpu().numpy()
        # test2 = route_key_padding[1].cpu().numpy()
        temp = route_key_padding.unsqueeze(1).expand(-1, location_embed.shape[1], -1)
        location_embed[~temp] = 0
        # test3 = location_embed[0].cpu().numpy()
        # test4 = location_embed[1].cpu().numpy()
        return location_embed
