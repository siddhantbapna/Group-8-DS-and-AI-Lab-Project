from typing import Literal

import monai
from monai.networks.nets import UNet, AttentionUnet, DynUNet, SegResNet, VNet


def create_model(
	name: Literal["unet2d", "unet3d", "attenunet", "nnunet", "resunet", "vnet"],
	in_channels: int,
	out_channels: int,
	feature_sizes_2d: list[int] | None = None,
	feature_sizes_3d: list[int] | None = None,
):
	# if name == "unet2d":
	# 	return UNet(
	# 		spatial_dims=2,
	# 		in_channels=in_channels,
	# 		out_channels=out_channels,
	# 		channels=feature_sizes_2d or [32, 64, 128, 256],
	# 		strides=(2, 2, 2, 2),
	# 		num_res_units=2,
	# 	)
	if name == "unet3d":
		return UNet(
			spatial_dims=3,
			in_channels=in_channels,
			out_channels=out_channels,
			channels=feature_sizes_3d or [16, 32, 64, 128],
			strides=(2, 2, 2, 2),
			num_res_units=2,
		)
	elif name == "attenunet":
		return AttentionUnet(
			spatial_dims=3,
			in_channels=in_channels,
			out_channels=out_channels,
			channels=feature_sizes_3d or [16, 32, 64, 128],
			strides=(2, 2, 2, 2),
		)
	elif name == "nnunet":
		# DynUNet with nnUNet-like config
		# Standard configuration with proper encoder/decoder structure
		filters = feature_sizes_3d or [16, 32, 64, 128]
		num_layers = len(filters)
		
		return DynUNet(
			spatial_dims=3,
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=[[3, 3, 3]] * num_layers,
			strides=[[1, 1, 1]] + [[2, 2, 2]] * (num_layers - 1),  # First layer stride=1, rest stride=2
			filters=filters,
			deep_supervision=False,  # Disable deep supervision for simplicity
			norm_name="instance",
			act_name="leakyrelu",
		)
	elif name == "resunet":
		return SegResNet(
			spatial_dims=3,
			in_channels=in_channels,
			out_channels=out_channels,
			init_filters=16,
			blocks_down=(1, 2, 2, 4),
			blocks_up=(1, 1, 1),
		)
	elif name == "vnet":
		return VNet(
			spatial_dims=3,
			in_channels=in_channels,
			out_channels=out_channels,
		)
	else:
		raise ValueError(f"Unknown model name: {name}")
