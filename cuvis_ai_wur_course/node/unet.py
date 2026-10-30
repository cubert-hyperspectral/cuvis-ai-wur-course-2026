"""The end-to-end U-Net chemical-map ensemble (Engstrom et al., 2025).

Wraps the study's vendored ``Conv3Don2D + UNet`` compound model as a
cuvis-ai node, run ``n_members`` times (one compound model per cross-
validation fold) and averaged, exactly as ``EnsembleModel`` does in
``cuvis_ai_wur_course.upstream.unet_chemical_map.models``.
"""

from __future__ import annotations

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_wur_course.upstream.unet_chemical_map.models.compound_model import (
    CompoundModel,
)
from cuvis_ai_wur_course.upstream.unet_chemical_map.models.conv3don2d import Conv3Don2D
from cuvis_ai_wur_course.upstream.unet_chemical_map.models.unet import UNet

_DEFAULT_MEMBER_IDS = (1, 2, 3, 4, 5)


class UNetChemicalMapEnsemble(Node):
    """Ensemble of ``Conv3Don2D + UNet`` compound models predicting a chemical map.

    Each member is the study's ``CompoundModel``: a learnable 3D-convolution
    front-end that compresses the spectral axis (stride 2 over bands and
    stride 2 over both spatial axes), followed by a valid-convolution U-Net
    that outputs one chemical-concentration channel at half the front-end's
    spatial resolution. The ensemble prediction is the mean of its members,
    run one after another (``EnsembleModel.forward`` in the vendored code),
    never stacked into one batched call.

    Member count and memory. One member holds the front-end (29 parameters)
    plus the U-Net (28,973,313 parameters for the study's channel widths).
    A full course slice is ``[1, 62, 2360, 1272]`` float32 (about 744 MB);
    holding one member and one slice at a time is what the study's own
    12 GB-class GPU setup assumes. Running ``n_members`` sequentially (never
    loading all five at once unless the hardware allows it) is why members
    are iterated in a fixed-length Python loop inside :meth:`forward`
    instead of batched: each iteration can be issued, consumed and freed
    before the next model's weights are needed. ``n_members`` below the
    study's 5 trades ensemble accuracy for memory and runtime, matching the
    course's own CPU fallback (``EVAL_MEMBERS_CPU``).

    Pretrained weights are the study's CC BY-NC 4.0 checkpoints on Hugging
    Face (``Sm00thix/unet_chemical_map``), downloaded through
    ``huggingface_hub.hf_hub_download`` at construction time when
    ``weights="hf"``, pinned to ``hf_revision``. Tests use ``weights=None``
    (random initialization) and never touch the network.
    """

    _category = NodeCategory.MODEL
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.REGRESSION, NodeTag.LEARNABLE, NodeTag.TORCH})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, "input_channels"),
            description="Dequantized hyperspectral slice, BHWC float32.",
        ),
    }
    OUTPUT_SPECS = {
        "map": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Predicted chemical-concentration map (percent), mean over the members.",
        ),
    }

    def __init__(
        self,
        n_members: int = 5,
        member_ids: list[int] | None = None,
        weights: str | None = None,
        hf_repo: str = "Sm00thix/unet_chemical_map",
        hf_revision: str = "9683a7ae6019c17e0e03f38a5e0d3c7dee9210e2",
        input_channels: int = 62,
        conv3d_kernel: tuple[int, int, int] = (7, 2, 2),
        conv3d_filters: int = 1,
        out_channels: int = 1,
        pad: bool = False,
        bilinear: bool = True,
        normalization: str | None = None,
        **kwargs,
    ) -> None:
        if not 1 <= n_members <= 5:
            raise ValueError(f"n_members must be between 1 and 5, got {n_members!r}")
        if member_ids is None:
            member_ids = list(_DEFAULT_MEMBER_IDS[:n_members])
        if len(member_ids) != n_members:
            raise ValueError(
                f"member_ids has {len(member_ids)} entries, expected n_members={n_members}"
            )
        if weights not in (None, "hf"):
            raise ValueError(f"weights must be 'hf' or None, got {weights!r}")

        super().__init__(
            n_members=n_members,
            member_ids=list(member_ids),
            weights=weights,
            hf_repo=hf_repo,
            hf_revision=hf_revision,
            input_channels=input_channels,
            conv3d_kernel=list(conv3d_kernel),
            conv3d_filters=conv3d_filters,
            out_channels=out_channels,
            pad=pad,
            bilinear=bilinear,
            normalization=normalization,
            **kwargs,
        )
        self.n_members = n_members
        self.member_ids = list(member_ids)
        self.weights = weights
        self.hf_repo = hf_repo
        self.hf_revision = hf_revision
        self.input_channels = input_channels
        self.conv3d_kernel = tuple(conv3d_kernel)
        self.conv3d_filters = conv3d_filters
        self.out_channels = out_channels
        self.pad = pad
        self.bilinear = bilinear
        self.normalization = normalization

        unet_input_channels = (input_channels - (self.conv3d_kernel[0] - 1)) * conv3d_filters

        self.members = torch.nn.ModuleList(
            [self._build_member(unet_input_channels) for _ in member_ids]
        )

        if weights == "hf":
            self._load_hf_weights()

    def _build_member(self, unet_input_channels: int) -> CompoundModel:
        front_end = Conv3Don2D(
            self.input_channels,
            self.conv3d_filters,
            self.conv3d_kernel,
            normalization=self.normalization,
        )
        unet = UNet(
            unet_input_channels,
            self.out_channels,
            pad=self.pad,
            bilinear=self.bilinear,
            normalization=self.normalization,
        )
        return CompoundModel(front_end, unet)

    def _load_hf_weights(self) -> None:
        """Download each member's checkpoint from Hugging Face and load it."""
        from huggingface_hub import hf_hub_download

        for member, member_id in zip(self.members, self.member_ids, strict=True):
            path = hf_hub_download(
                self.hf_repo,
                f"model_weights/{member_id}/best_weights.pt",
                revision=self.hf_revision,
            )
            state = torch.load(path, map_location="cpu", weights_only=True)
            missing, unexpected = member.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"UNetChemicalMapEnsemble: checkpoint for member {member_id} "
                    f"has {len(missing)} missing and {len(unexpected)} unexpected keys"
                )

    def forward(self, cube: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        """Run each member in turn on BHWC input, return the mean map (BHWC)."""
        x = cube.permute(0, 3, 1, 2)  # BHWC -> BCHW for the vendored models
        outputs = [member(x) for member in self.members]
        mean = torch.stack(outputs, dim=0).mean(dim=0)
        return {"map": mean.permute(0, 2, 3, 1)}  # BCHW -> BHWC
